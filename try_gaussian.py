import open3d as o3d
ply_path = "/shared/user59/workspace/cihan/3dgs_Vincent/my_3dgs/output/truck120w/point_cloud/iteration_740000/point_cloud.ply"

pcd = o3d.io.read_point_cloud(ply_path)
print("Fields available:")
print(pcd)
