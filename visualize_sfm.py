import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt



def visualize_reconstruction(poses: dict, points3d: list, cam_size=0.1, max_dist=50.0):
    import math
    
    def create_camera_frustum(R, t, scale=0.1, fov=60):
        h = 2 * scale * math.tan(np.radians(fov / 2))
        w = h * 4/3
        pts = np.array([
            [0, 0, 0],
            [w/2,  h/2, scale],
            [-w/2,  h/2, scale],
            [-w/2, -h/2, scale],
            [w/2, -h/2, scale]
        ])
        pts_world = (R @ pts.T).T + t.flatten()
        lines = [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [2, 3], [3, 4], [4, 1]]
        colors = [[1, 0, 0] for _ in lines]
        line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(pts_world),
            lines=o3d.utility.Vector2iVector(lines)
        )
        line_set.colors = o3d.utility.Vector3dVector(colors)
        return line_set

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='SFM Reconstruction', width=1280, height=720)

    cam_centers = [(-R.T @ t).flatten() for R, t in poses.values()]
    cam_centers = np.vstack(cam_centers)
    pts = np.vstack(points3d)

    def min_cam_distance(pt):
        return np.min(np.linalg.norm(cam_centers - pt, axis=1))

    filtered_pts = np.array([pt for pt in pts if min_cam_distance(pt) < max_dist])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(filtered_pts)
    z = filtered_pts[:, 2]
    z_min, z_max = z.min(), z.max()
    colors = (z - z_min) / (z_max - z_min + 1e-8)
    colors = plt.get_cmap('viridis')(colors)[:, :3]
    pcd.colors = o3d.utility.Vector3dVector(colors)
    vis.add_geometry(pcd)

    # Add camera frustums and spheres at camera centers
    for idx, (R, t) in poses.items():
        frustum = create_camera_frustum(R, t, scale=cam_size)
        vis.add_geometry(frustum)
        center = (-R.T @ t).flatten()
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=cam_size * 0.3)
        sphere.paint_uniform_color([1, 0.5, 0])
        sphere.translate(center)
        vis.add_geometry(sphere)

    # Add world coordinate frame
    world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
    vis.add_geometry(world_frame)

    render_option = vis.get_render_option()
    render_option.background_color = np.array([0.1, 0.1, 0.1])
    render_option.point_size = 3.0

    # Set viewpoint
    ctr = vis.get_view_control()
    ctr.set_front([0, 0, -1])
    ctr.set_lookat([0, 0, 0])
    ctr.set_up([0, -1, 0])
    ctr.set_zoom(0.8)

    vis.run()
    vis.destroy_window()

