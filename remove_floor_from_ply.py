#!/usr/bin/env python3
"""
remove_floor_from_ply.py

Remove floor / lowest points from a .ply point cloud (e.g., from SfM) so that
only the object remains.

Two strategies:
  1) Plane segmentation (recommended): detect the largest plane (floor) with RANSAC,
     drop those inliers, and keep points above that plane (with a small margin).
  2) PCA lower-slice (no plane fit required): find the "vertical" direction using PCA,
     then remove the lowest percentile along that axis.

Usage:
  python remove_floor_from_ply.py input.ply -o output.ply
  # With custom parameters:
  python remove_floor_from_ply.py input.ply -o output.ply \
      --method plane --plane-distance 0.008 --above-margin 0.005 \
      --dbscan-eps 0.02 --min-cluster-size 50 --denoise

  # PCA fallback:
  python remove_floor_from_ply.py input.ply -o output.ply \
      --method pca --lower-percentile 0.08 --above-margin 0.003 --denoise

Install:
  pip install open3d numpy
"""

import argparse
import sys
import os
import numpy as np

try:
    import open3d as o3d
except ImportError as e:
    o3d = None
    print("Warning: open3d not found. Install with `pip install open3d` for best results.", file=sys.stderr)


def load_point_cloud(path: str):
    if o3d is None:
        raise RuntimeError("open3d is required to read/write .ply files. Install with `pip install open3d`.")
    pcd = o3d.io.read_point_cloud(path)
    if pcd.is_empty():
        raise ValueError(f"No points loaded from {path}. Is it a valid .ply point cloud?")
    return pcd


def save_point_cloud(pcd: "o3d.geometry.PointCloud", path: str):
    ok = o3d.io.write_point_cloud(path, pcd, write_ascii=False, compressed=False)
    if not ok:
        raise IOError(f"Failed to write point cloud to {path}")


def optional_statistical_denoise(pcd, nb_neighbors=20, std_ratio=2.0):
    """Optionally remove isolated outliers before/after floor removal."""
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    return pcd.select_by_index(ind)


def keep_largest_cluster(pcd, eps=0.02, min_points=50):
    """Keep the largest DBSCAN cluster to isolate the main object."""
    if len(pcd.points) == 0:
        return pcd
    labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False))
    if labels.size == 0 or labels.max() < 0:
        return pcd  # No clusters found; return as-is
    # Keep the largest non-noise cluster
    unique, counts = np.unique(labels[labels >= 0], return_counts=True)
    if unique.size == 0:
        return pcd
    keep_label = unique[np.argmax(counts)]
    idx = np.where(labels == keep_label)[0]
    return pcd.select_by_index(idx)


def plane_floor_removal(pcd,
                        plane_distance=0.008,
                        ransac_n=3,
                        num_iterations=1000,
                        above_margin=0.003):
    """
    Segment the largest plane (assumed floor), remove its inliers,
    then keep only points 'above' that plane by a small margin.
    """
    pts = np.asarray(pcd.points)
    model, inliers = pcd.segment_plane(distance_threshold=plane_distance,
                                       ransac_n=ransac_n,
                                       num_iterations=num_iterations)
    if len(inliers) == 0:
        # No plane detected; return original
        return pcd
    a, b, c, d = model
    n = np.array([a, b, c], dtype=np.float64)
    n = n / (np.linalg.norm(n) + 1e-12)

    mask = np.ones(len(pts), dtype=bool)
    mask[inliers] = False  # drop plane inliers

    # Signed distance of all points to plane: s = n·x + d
    s = pts @ n + d

    # Orient normal so that remaining points are mostly "above" (positive)
    if np.mean(s[mask]) < 0:
        n = -n
        d = -d
        s = -s

    # Keep only points above the plane by a small margin
    keep = mask & (s >= above_margin)
    kept_idx = np.where(keep)[0]
    return pcd.select_by_index(kept_idx)


def pca_lower_slice_removal(pcd, lower_percentile=0.07, above_margin=0.003):
    """
    Find 'vertical' via PCA (smallest-variance axis ~ floor normal).
    Remove points below a percentile along that axis.
    """
    pts = np.asarray(pcd.points)
    center = pts.mean(axis=0, keepdims=True)
    X = pts - center

    # PCA via eigen-decomposition of covariance
    cov = np.cov(X, rowvar=False)
    w, V = np.linalg.eigh(cov)  # ascending eigenvalues
    v_vertical = V[:, 0]        # smallest variance eigenvector ~ floor normal

    # Project on vertical axis; 'lower' = small projection values
    t = (X @ v_vertical)
    thresh = np.quantile(t, lower_percentile) + above_margin
    keep = t > thresh
    kept_idx = np.where(keep)[0]
    return pcd.select_by_index(kept_idx)


def main():
    parser = argparse.ArgumentParser(description="Remove floor/lowest points from a .ply point cloud.")
    parser.add_argument("input", help="Path to input .ply")
    parser.add_argument("-o", "--output", required=True, help="Path to write cleaned .ply")
    parser.add_argument("--method", choices=["plane", "pca", "auto"], default="auto",
                        help="Removal strategy: 'plane' (RANSAC), 'pca' (lower-slice), or 'auto' (try plane, fallback to pca).")
    parser.add_argument("--denoise", action="store_true", help="Apply statistical outlier removal before and after.")
    # Plane params
    parser.add_argument("--plane-distance", type=float, default=0.008,
                        help="Max point-to-plane distance (meters) for plane inliers.")
    parser.add_argument("--ransac-n", type=int, default=3, help="RANSAC plane model points.")
    parser.add_argument("--ransac-iters", type=int, default=1000, help="RANSAC iterations.")
    # PCA params
    parser.add_argument("--lower-percentile", type=float, default=0.07,
                        help="Percentile of lowest points to remove in PCA mode (0..1).")
    # Common
    parser.add_argument("--above-margin", type=float, default=0.003,
                        help="Extra margin above the detected floor/lower threshold to keep (meters).")
    parser.add_argument("--dbscan-eps", type=float, default=0.02,
                        help="DBSCAN neighborhood radius (meters) to keep largest cluster.")
    parser.add_argument("--min-cluster-size", type=int, default=50,
                        help="DBSCAN minimum points per cluster to be considered valid.")
    args = parser.parse_args()

    # Load
    pcd = load_point_cloud(args.input)

    # Optional pre-denoise
    if args.denoise:
        pcd = optional_statistical_denoise(pcd, nb_neighbors=20, std_ratio=2.0)

    original_count = len(pcd.points)

    # Remove floor / lower slice
    cleaned = None
    if args.method == "plane":
        cleaned = plane_floor_removal(
            pcd,
            plane_distance=args.plane_distance,
            ransac_n=args.ransac_n,
            num_iterations=args.ransac_iters,
            above_margin=args.above_margin
        )
    elif args.method == "pca":
        cleaned = pca_lower_slice_removal(
            pcd,
            lower_percentile=args.lower_percentile,
            above_margin=args.above_margin
        )
    else:  # auto
        try:
            cleaned = plane_floor_removal(
                pcd,
                plane_distance=args.plane_distance,
                ransac_n=args.ransac_n,
                num_iterations=args.ransac_iters,
                above_margin=args.above_margin
            )
            # If plane strategy removed nothing, fall back to PCA
            if len(cleaned.points) == original_count:
                cleaned = pca_lower_slice_removal(
                    pcd,
                    lower_percentile=args.lower_percentile,
                    above_margin=args.above_margin
                )
        except Exception:
            # If plane segmentation fails, use PCA
            cleaned = pca_lower_slice_removal(
                pcd,
                lower_percentile=args.lower_percentile,
                above_margin=args.above_margin
            )

    # Optional post-denoise + keep largest cluster (object)
    cleaned = keep_largest_cluster(cleaned, eps=args.dbscan_eps, min_points=args.min_cluster_size)
    if args.denoise:
        cleaned = optional_statistical_denoise(cleaned, nb_neighbors=20, std_ratio=2.0)

    # Save
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    save_point_cloud(cleaned, args.output)

    print(f"Input points:   {original_count}")
    print(f"Output points:  {len(cleaned.points)}")
    print(f"Saved cleaned point cloud to: {args.output}")


if __name__ == "__main__":
    main()
