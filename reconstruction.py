import logging
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

@dataclass
class Point3DView:
    coords: np.ndarray
    rgb: Optional[Tuple[int, int, int]] = None  # RGB color of the point
    observations: Dict[int, int] = field(default_factory=dict)

@dataclass
class ReconstructionCfg:
    K: np.ndarray
    min_inliers_baseline: int = 10
    essential_ransac_thresh: float = 2.0
    pnp_reproj_thresh: float = 4.0
    pnp_iterations: int = 100  # Reduced from 1000
    pnp_method: int = cv2.SOLVEPNP_EPNP  # Faster than P3P
    min_pnp_correspondences: int = 10
    max_failed_attempts: int = 3
    bundle_every: int = 5  # Less frequent bundle adjustment
    verbose: bool = True
    num_threads: int = 12  # For multithreading

class Reconstruction:
    def __init__(
        self,
        keypoints: List[List[cv2.KeyPoint]],
        matches: Dict[Tuple[int, int], List[cv2.DMatch]],
        img_adjacency: np.ndarray,
        cfg: ReconstructionCfg,
        images: List[np.ndarray]
    ):
        self.keypoints = keypoints
        self.matches = matches
        self.adjacency = img_adjacency
        self.cfg = cfg
        self.images = images
        self.poses: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        self.points3d: List[Point3DView] = []
        self.placed: List[int] = []
        self.unplaced: List[int] = list(range(img_adjacency.shape[0]))
        self.failed_attempts: Dict[int, int] = {}
        self.add_count = 0

    def select_baseline(self, top_percent: float = 0.3,
                    min_parallax_deg: float = 2.0) -> Tuple[int, int]:
        """
        Select a baseline image pair for initialization.
    
        Args:
            top_percent: optional, unused here but could be used if you want
                         to keep the top-k% pairs.
            min_parallax_deg: minimum median parallax (in degrees)
                              required to accept a pair.
        Returns:
            (i, j): indices of selected baseline pair
        """
    
        def compute_parallax(pts_i, pts_j, K, R, t, mask):
            pts_i = pts_i[mask.ravel() == 255]
            pts_j = pts_j[mask.ravel() == 255]
            if len(pts_i) == 0:
                return 0.0
            # undistort & normalize to bearing vectors
            v1 = cv2.undistortPoints(pts_i.reshape(-1,1,2), K, None).reshape(-1,2)
            v2 = cv2.undistortPoints(pts_j.reshape(-1,1,2), K, None).reshape(-1,2)
            v1 = np.hstack([v1, np.ones((v1.shape[0],1))])
            v2 = np.hstack([v2, np.ones((v2.shape[0],1))])
            v2_rot = (R @ v2.T).T
            cos_angle = np.sum(v1 * v2_rot, axis=1) / (
                np.linalg.norm(v1, axis=1) * np.linalg.norm(v2_rot, axis=1)
            )
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            return np.median(np.arccos(cos_angle))  # radians
    
        scores = []
        for (i, j), mlist in self.matches.items():
            if len(mlist) < self.cfg.min_inliers_baseline:
                continue
            pts_i, pts_j = self._aligned_points(i, j)
            E, mask = cv2.findEssentialMat(
                pts_i, pts_j, self.cfg.K,
                method=cv2.FM_RANSAC,
                threshold=self.cfg.essential_ransac_thresh
            )
            if mask is None or mask.sum() < self.cfg.min_inliers_baseline:
                continue
            _, R, t, out_mask = cv2.recoverPose(E, pts_i, pts_j, self.cfg.K)
            parallax = compute_parallax(pts_i, pts_j, self.cfg.K, R, t, out_mask)
            parallax_deg = np.degrees(parallax)
    
            # filter by parallax
            if parallax_deg < min_parallax_deg:
                logger.debug(f"Rejected pair {(i, j)}: parallax {parallax_deg:.2f}° < {min_parallax_deg}°")
                continue
    
            scores.append(((i, j), len(mlist), int(out_mask.sum()), parallax_deg))
    
        if not scores:
            raise RuntimeError("No valid baseline pair found (all had too little parallax).")
    
        # sort by inliers, then by parallax
        scores.sort(key=lambda x: (x[2], x[3]), reverse=True)
        best = scores[0]
        logger.info(f"Baseline pair: {best[0]} with {best[1]} matches, "
                    f"{best[2]} inliers, median parallax {best[3]:.2f}°")
        return best[0]


    def initialize(self, baseline: Tuple[int, int]) -> None:
        i, j = baseline
        pts_i, pts_j, idxs_i, idxs_j = self._aligned_points(i, j, return_idxs=True)
        E, mask = cv2.findEssentialMat(
            pts_i, pts_j, self.cfg.K,
            method=cv2.FM_RANSAC,
            threshold=self.cfg.essential_ransac_thresh
        )
        if mask is None or mask.sum() < 15:
            raise ValueError("Baseline failed due to insufficient inliers")
        _, R, t, pose_mask = cv2.recoverPose(E, pts_i, pts_j, self.cfg.K)
        inliers = pose_mask.ravel().astype(bool)
        if inliers.sum() < 15:
            raise ValueError("Pose recovery failed with low inliers")
        self.poses[i] = (np.eye(3), np.zeros((3, 1)))
        self.poses[j] = (R, t)
        idxs_i = np.array(idxs_i)[inliers]
        idxs_j = np.array(idxs_j)[inliers]
        self._triangulate_and_add(i, j, idxs_i, idxs_j)
        self.placed = [i, j]
        self.unplaced.remove(i)
        self.unplaced.remove(j)
        logger.info("Initialized reconstruction with baseline image pair.")

    def grow(self, bundle_adjust_fn=None, pbar=None):
        last_pose_count = len(self.poses)
        while self.unplaced:
            valid_unplaced = [
                u for u in self.unplaced
                if self.failed_attempts.get(u, 0) < self.cfg.max_failed_attempts
            ]
            if not valid_unplaced:
                logger.warning("All unplaced images have exceeded failure limit.")
                break
            try:
                best_img, best_corr = max(
                    ((u, self._count_correspondences(u)) for u in valid_unplaced),
                    key=lambda x: x[1]
                )
            except ValueError:
                logger.warning("No valid images with correspondences.")
                break
            if best_corr < self.cfg.min_pnp_correspondences:
                logger.warning(f"Image {best_img} has only {best_corr} correspondence(s); skipping.")
                self.unplaced.remove(best_img)
                continue
            success = False
            try:
                self._add_image(best_img)
                success = True
                self.add_count += 1
                if pbar:
                    pbar.update(1)
                if bundle_adjust_fn and self.add_count % self.cfg.bundle_every == 0:
                    bundle_adjust_fn()
                if best_img in self.failed_attempts:
                    del self.failed_attempts[best_img]
            except Exception as e:
                logger.error(f"Failed to add image {best_img}: {str(e)}")
                self.failed_attempts[best_img] = self.failed_attempts.get(best_img, 0) + 1
                if self.failed_attempts[best_img] >= self.cfg.max_failed_attempts:
                    logger.warning(f"Permanently removing image {best_img}")
                    self.unplaced.remove(best_img)
            if len(self.poses) == last_pose_count:
                logger.warning("No progress made in this iteration. Stopping to prevent infinite loop.")
                break
            last_pose_count = len(self.poses)

    def _count_correspondences(self, img_idx: int) -> int:
        count = 0
        for pt in self.points3d:
            for p in self.placed:
                pair = tuple(sorted((p, img_idx)))
                if pair not in self.matches:
                    continue
                matches = self.matches[pair]
                for m in matches:
                    q, t = (m.queryIdx, m.trainIdx) if p < img_idx else (m.trainIdx, m.queryIdx)
                    if q < len(self.keypoints[p]) and t < len(self.keypoints[img_idx]):
                        if pt.observations.get(p) == q:
                            count += 1
                            break
        return count

    def _add_image(self, img_idx: int, pbar=None) -> None:
        pts3d, pts2d, pt_objs = [], [], []
        for pt3d in self.points3d:
            for p in self.placed:
                pair = tuple(sorted((p, img_idx)))
                if pair not in self.matches:
                    continue
                matches = self.matches[pair]
                for m in matches:
                    q, t = (m.queryIdx, m.trainIdx) if p < img_idx else (m.trainIdx, m.queryIdx)
                    if q < len(self.keypoints[p]) and t < len(self.keypoints[img_idx]):
                        if pt3d.observations.get(p) == q:
                            pts3d.append(pt3d.coords)
                            pts2d.append(self.keypoints[img_idx][t].pt)
                            pt_objs.append((pt3d, t))
                            break

        if len(pts3d) < self.cfg.min_pnp_correspondences:
            candidates = sorted(
                [(p, len(self.matches.get(tuple(sorted((p, img_idx))), []))) for p in self.placed],
                key=lambda x: x[1], reverse=True
            )
            if not candidates:
                raise ValueError("No placed image has matches with this one")
            best_p, _ = candidates[0]
            R_bp, t_bp = self.poses[best_p]
            if np.linalg.norm(t_bp) < 1e-6 and len(self.placed) > 2:
                raise ValueError("Neighbor has invalid identity pose")
            raw_matches = self.matches.get(tuple(sorted((best_p, img_idx))), [])
            match_list = [
                m for m in raw_matches
                if m.queryIdx < len(self.keypoints[best_p]) and
                m.trainIdx < len(self.keypoints[img_idx])
            ]
            if len(match_list) >= 8:
                idxs_p = [m.queryIdx if best_p < img_idx else m.trainIdx for m in match_list]
                idxs_n = [m.trainIdx if best_p < img_idx else m.queryIdx for m in match_list]
                self._triangulate_and_add(best_p, img_idx, idxs_p, idxs_n)
                R_rel = np.eye(3)
                t_rel = np.array([[0.1], [0], [0]])
                R_abs = R_rel @ self.poses[best_p][0]
                t_abs = self.poses[best_p][1] + self.poses[best_p][0] @ t_rel
                self.poses[img_idx] = (R_abs, t_abs)
                self.placed.append(img_idx)
                self.unplaced.remove(img_idx)
                logger.info(f"Added {img_idx} via fallback with {len(match_list)} matches")
                if pbar:
                    pbar.update(1)
                return

        if len(pts3d) >= self.cfg.min_pnp_correspondences:
            success, rvec, tvec, inliers = cv2.solvePnPRansac(
                np.array(pts3d).reshape(-1, 3),
                np.array(pts2d).reshape(-1, 2),
                self.cfg.K,
                distCoeffs=None,
                iterationsCount=self.cfg.pnp_iterations,
                reprojectionError=self.cfg.pnp_reproj_thresh,
                flags=self.cfg.pnp_method  # Optimized solver
            )
            if success and inliers is not None and len(inliers) >= 4:
                R, _ = cv2.Rodrigues(rvec)
                self.poses[img_idx] = (R, tvec.reshape(3, 1))
                for idx in inliers.ravel():
                    pt_objs[idx][0].observations[img_idx] = pt_objs[idx][1]
                self.placed.append(img_idx)
                self.unplaced.remove(img_idx)
                logger.info(f"Added image {img_idx} with {len(inliers)} PnP inliers")
                self._triangulate_new_matches(img_idx)
                if pbar:
                    pbar.update(1)
            else:
                logger.warning(f"PnP failed for image {img_idx}")
                raise RuntimeError("PnP failed or insufficient inliers")
        else:
            raise ValueError("Insufficient correspondences for PnP")

    def _aligned_points(self, i: int, j: int, return_idxs: bool = False):
        mlist = self.matches.get(tuple(sorted((i, j))), [])
        pts_i = np.array([self.keypoints[i][m.queryIdx].pt for m in mlist], dtype=np.float32)
        pts_j = np.array([self.keypoints[j][m.trainIdx].pt for m in mlist], dtype=np.float32)
        if return_idxs:
            idxs_i = np.array([m.queryIdx for m in mlist])
            idxs_j = np.array([m.trainIdx for m in mlist])
            return pts_i, pts_j, idxs_i, idxs_j
        return pts_i, pts_j

    def _process_pair(self, p: int, img_idx: int, P_i: np.ndarray, img_n: np.ndarray) -> List[Point3DView]:
        R_j, t_j = self.poses[p]
        P_j = self.cfg.K @ np.hstack((R_j, t_j))
        pair = tuple(sorted((p, img_idx)))
        matches = self.matches.get(pair, [])
        idxs_p, idxs_n = [], []
        for m in matches:
            q, t = (m.queryIdx, m.trainIdx) if p < img_idx else (m.trainIdx, m.queryIdx)
            if q < len(self.keypoints[p]) and t < len(self.keypoints[img_idx]):
                has_observation = False
                for pt3d in self.points3d:
                    if pt3d.observations.get(p) == q or pt3d.observations.get(img_idx) == t:
                        has_observation = True
                        break
                if not has_observation:
                    idxs_p.append(q)
                    idxs_n.append(t)
        if len(idxs_p) < 4:
            return []
        pts_p = np.array([self.keypoints[p][i].pt for i in idxs_p], dtype=np.float64).T.reshape(2, -1)
        pts_n = np.array([self.keypoints[img_idx][i].pt for i in idxs_n], dtype=np.float64).T.reshape(2, -1)
        pts4d = cv2.triangulatePoints(P_j, P_i, pts_p, pts_n)
        pts3d = cv2.convertPointsFromHomogeneous(pts4d.T).squeeze()
        depths = pts3d[:, 2]
        valid = depths > 0.1
        if valid.sum() < 2:
            return []
        img_p = self.images[p]
        new_points = []
        for k in range(len(valid)):
            if not valid[k]:
                continue
            coords = pts3d[k]
            x_p, y_p = map(int, self.keypoints[p][idxs_p[k]].pt)
            x_n, y_n = map(int, self.keypoints[img_idx][idxs_n[k]].pt)
            rgb_p = tuple(reversed(img_p[y_p, x_p]))
            rgb_n = tuple(reversed(img_n[y_n, x_n]))
            avg_rgb = tuple((np.array(rgb_p) + np.array(rgb_n)) // 2)
            pt = Point3DView(
                coords=coords,
                rgb=avg_rgb,
                observations={p: idxs_p[k], img_idx: idxs_n[k]}
            )
            new_points.append(pt)
        return new_points

    def _triangulate_new_matches(self, img_idx: int):
        logger.debug(f"Triangulating new points for image {img_idx}")
        R_i, t_i = self.poses[img_idx]
        P_i = self.cfg.K @ np.hstack((R_i, t_i))
        img_n = self.images[img_idx]
        new_points_list = []
        
        # Multithreaded triangulation
        with ThreadPoolExecutor(max_workers=self.cfg.num_threads) as executor:
            futures = []
            for p in self.placed:
                if p != img_idx:
                    futures.append(executor.submit(self._process_pair, p, img_idx, P_i, img_n))
            for future in futures:
                new_points_list.extend(future.result())

        # Batch update points3d
        self.points3d.extend(new_points_list)
        if new_points_list:
            logger.info(f"Added {len(new_points_list)} new colored 3D points during addition of image {img_idx}")
        else:
            logger.warning(f"No new 3D points could be triangulated for image {img_idx}")

    def _triangulate_and_add(self, i: int, j: int, idxs_i: List[int], idxs_j: List[int]) -> None:
        logger.info(f"Triangulating {len(idxs_i)} points between images {i} and {j}")
        R_i, t_i = self.poses[i]
        R_j, t_j = self.poses[j]
        P_i = self.cfg.K @ np.hstack((R_i, t_i))
        P_j = self.cfg.K @ np.hstack((R_j, t_j))
        pts_i = np.array([self.keypoints[i][idx].pt for idx in idxs_i], dtype=np.float64).T.reshape(2, -1)
        pts_j = np.array([self.keypoints[j][idx].pt for idx in idxs_j], dtype=np.float64).T.reshape(2, -1)
        pts4d = cv2.triangulatePoints(P_i.astype(np.float64), P_j.astype(np.float64), pts_i, pts_j)
        pts3d = cv2.convertPointsFromHomogeneous(pts4d.T).squeeze()
        depths = pts3d[:, 2]
        valid_mask = depths > 0.1
        if valid_mask.sum() < 2:
            raise ValueError("Not enough valid 3D points after triangulation")
        img_i = self.images[i]
        img_j = self.images[j]
        for k in range(len(valid_mask)):
            if not valid_mask[k]:
                continue
            coords = pts3d[k]
            x_i, y_i = map(int, self.keypoints[i][idxs_i[k]].pt)
            x_j, y_j = map(int, self.keypoints[j][idxs_j[k]].pt)
            rgb_i = tuple(reversed(img_i[y_i, x_i]))
            rgb_j = tuple(reversed(img_j[y_j, x_j]))
            avg_rgb = tuple((np.array(rgb_i) + np.array(rgb_j)) // 2)
            pt = Point3DView(
                coords=coords,
                rgb=avg_rgb,
                observations={i: idxs_i[k], j: idxs_j[k]}
            )
            self.points3d.append(pt)
        logger.info(f"Successfully added {valid_mask.sum()} new colored 3D points between images {i} and {j}")
