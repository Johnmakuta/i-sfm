#!/usr/bin/env python3
"""
measure_ply_object.py

Reads a .ply point cloud (ASCII or binary-little-endian with vertex x,y,z) and
prints object height, width, and their ratio. Two measurement modes:

1) Axis-aligned (assumes Z is "up"):
   - height = range along Z
   - width  = max(range along X, range along Y)

2) PCA-oriented (orientation-agnostic):
   - height = largest principal extent
   - width  = second-largest principal extent

By default, shows both so you can choose which matches your capture setup.
Usage:
    python measure_ply_object.py path/to/file.ply [--mode z|pca|both]

Notes:
- Units are whatever your point cloud uses (often meters).
- The script tolerates common ASCII PLY files and most binary little-endian PLYs.
- Only vertex positions are required; color/other attributes are ignored.
"""

import sys
import struct
import argparse
import numpy as np

def _read_ply_header(fobj):
    """
    Reads the PLY header and returns:
      format: "ascii" or "binary_little_endian"
      vertex_count: int
      properties: list of (name, type) in the order declared for 'vertex'
      header_end_offset: byte offset where data rows start
    Raises ValueError if unsupported format or missing vertex element.
    """
    # Ensure we start at file beginning
    fobj.seek(0)
    first = fobj.readline().decode('latin1').strip()
    if first != "ply":
        raise ValueError("Not a PLY file (missing 'ply' magic).")

    fmt = None
    vertex_count = None
    properties = []
    in_vertex = False

    header_lines = [first]

    while True:
        line = fobj.readline()
        if not line:
            raise ValueError("Unexpected EOF while reading PLY header.")
        s = line.decode('latin1').rstrip("\r\n")
        header_lines.append(s)

        if s.startswith("format "):
            if "ascii" in s:
                fmt = "ascii"
            elif "binary_little_endian" in s:
                fmt = "binary_little_endian"
            else:
                raise ValueError(f"Unsupported PLY format: {s}")

        elif s.startswith("element "):
            tokens = s.split()
            if len(tokens) == 3 and tokens[1] == "vertex":
                in_vertex = True
                vertex_count = int(tokens[2])
            else:
                # Leaving vertex element; ignore other elements
                in_vertex = False

        elif s.startswith("property ") and in_vertex:
            # property <type> <name>  (assume scalar)
            tokens = s.split()
            if len(tokens) < 3:
                raise ValueError(f"Malformed property line: {s}")
            ptype, pname = tokens[1], tokens[2]
            properties.append((pname, ptype))

        elif s == "end_header":
            header_end = fobj.tell()
            break

    if fmt is None or vertex_count is None or not properties:
        raise ValueError("Invalid PLY: missing format/vertex element/properties.")

    return fmt, vertex_count, properties, header_end

# Map PLY scalar type to struct format and numpy dtype
PLY_TO_STRUCT = {
    'char':  'b', 'uchar': 'B', 'int8': 'b', 'uint8': 'B',
    'short': 'h', 'ushort': 'H', 'int16': 'h', 'uint16': 'H',
    'int':   'i', 'uint':  'I', 'int32': 'i','uint32': 'I',
    'float': 'f', 'float32':'f', 'double':'d', 'float64':'d'
}
PLY_TO_DTYPE = {
    'char':  np.int8,  'uchar': np.uint8,  'int8': np.int8,  'uint8': np.uint8,
    'short': np.int16, 'ushort':np.uint16, 'int16':np.int16, 'uint16':np.uint16,
    'int':   np.int32, 'uint':  np.uint32, 'int32':np.int32, 'uint32':np.uint32,
    'float': np.float32,'float32':np.float32,'double':np.float64,'float64':np.float64
}

def load_ply_vertices(path):
    """
    Load vertices (x, y, z) from a PLY file.
    Supports:
      - ASCII PLY
      - Binary little-endian PLY (scalar properties; lists not supported)
    Returns: numpy array of shape (N, 3)
    """
    with open(path, 'rb') as f:
        fmt, nverts, props, data_start = _read_ply_header(f)

        # Find indices of x, y, z
        name_to_idx = {name: i for i, (name, _) in enumerate(props)}
        if not all(k in name_to_idx for k in ('x','y','z')):
            raise ValueError("PLY must contain x, y, z properties in vertex element.")
        x_i, y_i, z_i = name_to_idx['x'], name_to_idx['y'], name_to_idx['z']

        if fmt == "ascii":
            f.seek(data_start)
            pts = np.empty((nverts, 3), dtype=np.float64)
            for i in range(nverts):
                line = f.readline()
                if not line:
                    raise ValueError("Unexpected EOF while reading ASCII vertex data.")
                parts = line.decode('latin1').strip().split()
                try:
                    pts[i, 0] = float(parts[x_i])
                    pts[i, 1] = float(parts[y_i])
                    pts[i, 2] = float(parts[z_i])
                except (IndexError, ValueError):
                    raise ValueError(f"Malformed vertex line at index {i}: {parts}")
            return pts

        elif fmt == "binary_little_endian":
            # Build a struct format string per-vertex
            fmts = []
            dtypes = []
            for _, t in props:
                if t not in PLY_TO_STRUCT:
                    raise ValueError(f"Unsupported PLY property type: {t}")
                fmts.append(PLY_TO_STRUCT[t])
                dtypes.append(PLY_TO_DTYPE[t])
            per_vtx_struct = struct.Struct('<' + ''.join(fmts))  # little-endian

            f.seek(data_start)
            pts = np.empty((nverts, 3), dtype=np.float64)
            for i in range(nverts):
                data = f.read(per_vtx_struct.size)
                if len(data) != per_vtx_struct.size:
                    raise ValueError("Unexpected EOF while reading binary vertex data.")
                values = per_vtx_struct.unpack(data)
                pts[i, 0] = float(values[x_i])
                pts[i, 1] = float(values[y_i])
                pts[i, 2] = float(values[z_i])
            return pts
        else:
            raise ValueError(f"Unsupported format: {fmt}")

def axis_aligned_dims(points):
    """
    Axis-aligned ranges along x,y,z. Returns (dx, dy, dz).
    """
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    return tuple((maxs - mins).tolist())

def pca_extents(points):
    """
    PCA extents along principal axes. Returns sorted extents (e1>=e2>=e3) and eigenvectors.
    """
    # Center
    P = points - points.mean(axis=0, keepdims=True)
    # Covariance and eigen-decomposition
    C = np.cov(P.T)
    evals, evecs = np.linalg.eigh(C)  # returns ascending evals
    order = np.argsort(evals)[::-1]
    evecs = evecs[:, order]
    # Project and compute ranges along principal axes
    Proj = P @ evecs  # shape (N,3)
    ranges = Proj.max(axis=0) - Proj.min(axis=0)
    # Sort ranges largest->smallest to define height/width/depth
    sort_idx = np.argsort(ranges)[::-1]
    sorted_ranges = ranges[sort_idx]
    # Reorder eigenvectors accordingly (optional return)
    evecs_sorted = evecs[:, sort_idx]
    return sorted_ranges, evecs_sorted

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ply_path", help="Path to input .ply point cloud")
    ap.add_argument("--mode", choices=["z","pca","both"], default="both",
                    help="Measurement mode: z (assume Z-up), pca (orientation-agnostic), both (default).")
    args = ap.parse_args()

    pts = load_ply_vertices(args.ply_path)

    # Axis-aligned measurements (Z-up assumption)
    dx, dy, dz = axis_aligned_dims(pts)
    axis_height = dz
    axis_width  = max(dx, dy)
    axis_ratio  = axis_height / axis_width if axis_width > 0 else float('inf')

    # PCA-oriented measurements
    pca_ranges, _ = pca_extents(pts)
    pca_height = pca_ranges[0]
    pca_width  = pca_ranges[1] if len(pca_ranges) > 1 else 0.0
    pca_ratio  = pca_height / pca_width if pca_width > 0 else float('inf')

    # Print results
    units = "(units: same as the .ply file)"
    if args.mode in ("z","both"):
        print("== Axis-aligned (assumes Z is up) ==")
        print(f"  Height (ΔZ): {axis_height:.6f} {units}")
        print(f"  Width  (max(ΔX, ΔY)): {axis_width:.6f} {units}")
        print(f"  Height/Width ratio: {axis_ratio:.6f}")
        print("")

    if args.mode in ("pca","both"):
        print("== PCA-oriented (orientation agnostic) ==")
        print(f"  Height (largest extent): {pca_height:.6f} {units}")
        print(f"  Width  (2nd-largest):   {pca_width:.6f} {units}")
        print(f"  Height/Width ratio:     {pca_ratio:.6f}")

if __name__ == "__main__":
    main()
