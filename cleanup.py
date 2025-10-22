"""
Script for cleaning up generated .ply files.
Example usage:
python3 cleanup.py Tiamat_Dragon.ply Tiamat_Dragon_Cleaned.ply --min_points_per_voxel 3

"""


import sys
import numpy as np
from collections import deque, defaultdict
from pathlib import Path

def read_ply_ascii_points_with_rgb(path):
    header = []
    with open(path, "r", encoding="ascii", errors="ignore") as f:
      for line in f:
        header.append(line.rstrip("\n"))
        if line.strip() == "end_header":
          break
      data = np.loadtxt(f, dtype=float)
    if data.ndim == 1:
      data = data.reshape(1, -1)
    if data.shape[1] == 6:
      pts = data[:, :3].astype(float)
      colors = data[:, 3:].astype(np.uint8)
    elif data.shape[1] == 3:
      pts = data[:, :3].astype(float)
      colors = None
    else:
      raise ValueError(f"Unsupported vertex format with {data.shape[1]} columns. Expected 3 or 6.")
    return header, pts, colors

def write_ply_ascii_points_with_rgb(path, points, colors=None):
    n = points.shape[0]
    header_lines = ["ply","format ascii 1.0",f"element vertex {n}",
                    "property float x","property float y","property float z"]
    if colors is not None:
      header_lines += ["property uchar red","property uchar green","property uchar blue"]
    header_lines.append("end_header")
    with open(path, "w", encoding="ascii") as f:
      f.write("\n".join(header_lines) + "\n")
      if colors is None:
        np.savetxt(f, points, fmt="%.6f %.6f %.6f")
      else:
        arr = np.hstack([points.astype(float), colors.astype(np.uint8)])
        np.savetxt(f, arr, fmt="%.6f %.6f %.6f %d %d %d")

def auto_voxel_size(pts, target_steps=200):
    ranges = pts.max(axis=0) - pts.min(axis=0)
    diag = np.linalg.norm(ranges)
    return float(max(diag / target_steps, 1e-5)) if diag > 0 else 1e-3

def build_voxel_map(pts, voxel_size):
    mins = pts.min(axis=0)
    ijk = np.floor((pts - mins) / voxel_size).astype(np.int32)
    voxel_to_indices = defaultdict(list)
    for idx, key in enumerate(map(tuple, ijk)):
      voxel_to_indices[key].append(idx)
    return voxel_to_indices, ijk, mins

def largest_connected_component(voxel_to_indices):
    neighbor_offsets = [(dx,dy,dz) for dx in (-1,0,1) for dy in (-1,0,1) for dz in (-1,0,1)
                        if not (dx==0 and dy==0 and dz==0)]
    voxels = set(voxel_to_indices.keys())
    visited, best_component = set(), set()
    for v in voxels:
      if v in visited: continue
      comp = {v}; visited.add(v); q = deque([v])
      while q:
        cx,cy,cz = q.popleft()
        for dx,dy,dz in neighbor_offsets:
          nb = (cx+dx, cy+dy, cz+dz)
          if nb in voxels and nb not in visited:
            visited.add(nb); comp.add(nb); q.append(nb)
      if len(comp) > len(best_component):
        best_component = comp
    return best_component

def filter_sparse_voxels(voxel_to_indices, min_points=3):
    return {v: idxs for v, idxs in voxel_to_indices.items() if len(idxs) >= min_points}

def clean_point_cloud(pts, colors=None, voxel_size=None, min_points_per_voxel=3):
    if voxel_size is None or voxel_size <= 0:
      voxel_size = auto_voxel_size(pts, target_steps=200)
    voxel_to_indices, ijk, mins = build_voxel_map(pts, voxel_size)
    voxel_to_indices = filter_sparse_voxels(voxel_to_indices, min_points=min_points_per_voxel)
    if not voxel_to_indices:
      voxel_to_indices, ijk, mins = build_voxel_map(pts, voxel_size)
    lcc_voxels = largest_connected_component(voxel_to_indices)
    keep_mask = np.zeros(len(pts), dtype=bool)
    for v in lcc_voxels:
      for idx in voxel_to_indices.get(v, []):
        keep_mask[idx] = True
    cleaned_pts = pts[keep_mask]
    cleaned_colors = colors[keep_mask] if colors is not None else None
    return cleaned_pts, cleaned_colors, keep_mask, voxel_size

def main():
    if len(sys.argv) < 3:
      print("Usage: python cleanup.py <input.ply> <output.ply> "
            "[--voxel_size <float>] [--min_points_per_voxel <int>]")
      sys.exit(1)
    in_path = Path(sys.argv[1]); out_path = Path(sys.argv[2])
    voxel_size = None; min_points_per_voxel = 3
    i = 3
    while i < len(sys.argv):
      if sys.argv[i] == "--voxel_size" and i+1 < len(sys.argv):
        voxel_size = float(sys.argv[i+1]); i += 2
      elif sys.argv[i] == "--min_points_per_voxel" and i+1 < len(sys.argv):
        min_points_per_voxel = int(sys.argv[i+1]); i += 2
      else:
        i += 1

    header, pts, colors = read_ply_ascii_points_with_rgb(in_path)
    cleaned_pts, cleaned_colors, keep_mask, vs = clean_point_cloud(
      pts, colors, voxel_size, min_points_per_voxel
    )
    write_ply_ascii_points_with_rgb(out_path, cleaned_pts, cleaned_colors)

    # Report & save indices of kept points
    print(f"Input points: {len(pts)}")
    print(f"Kept points:  {len(cleaned_pts)}")
    print(f"Removed:      {len(pts) - len(cleaned_pts)}")
    print(f"Voxel size used: {vs:.6f}")
    mask_path = out_path.with_suffix(".keepidx.txt")
    np.savetxt(mask_path, np.where(keep_mask)[0], fmt="%d")
    print(f"Saved kept-point indices to: {mask_path}")
    print(f"Output written to: {out_path}")

if __name__ == "__main__":
    main()
