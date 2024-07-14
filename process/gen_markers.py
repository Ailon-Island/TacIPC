import os
import numpy as np
import trimesh
from trimesh.ray.ray_triangle import RayMeshIntersector
from argparse import ArgumentParser
from icecream import ic
from vedo import show
from scipy.spatial.transform import Rotation as R 
import pickle as pkl
import sys
sys.path.append('.')

gel_folder = "resources/gel/gelslim-gel-l_0.1-3mm"
gel_surf_obj_file = 'surf.obj'

parser = ArgumentParser()
parser.add_argument('--gel_folder', type=str, default=gel_folder)
parser.add_argument('--gel_surf_obj_file', type=str, default=gel_surf_obj_file)
parser.add_argument('--vis', action='store_true', default=False)
parser.add_argument('--rot_z', type=float, default=0.)


def cartesian(arr_x, arr_y):
    xv, yv = np.meshgrid(arr_x, arr_y)
    return np.stack([xv.reshape(-1), yv.reshape(-1)], axis=1)


def trimesh_show(geoms):
    scene = trimesh.Scene()
    for g in geoms:
        scene.add_geometry(g)
    scene.show()


def bc_weight(tri_verts, point):
    A = (tri_verts[:2] - tri_verts[2]).T
    ws = np.linalg.inv(A.T @ A) @ (A.T @ (point - tri_verts[2]))
    return ws[0], ws[1], 1 - ws[0] - ws[1]


def point_edge_dist(point, edge):
    e01 = edge[1] - edge[0]
    proj_dot = np.dot(point - edge[0], e01)
    edge2 = np.dot(e01, e01)
    if proj_dot < 0:
        return np.linalg.norm(point - edge[0])
    elif proj_dot > edge2:
        return np.linalg.norm(point - edge[1])
    else:
        return np.linalg.norm(
            np.cross(point - edge[0],
                     point - edge[1])) / edge2 ** 0.5


def point_tri_edge_dist(point, tri):
    edges = np.stack([tri, tri[[1, 2, 0]]], axis=1)
    dists = [point_edge_dist(point, edge) for edge in edges]
    return min(dists)


if __name__ == '__main__':
    args = parser.parse_args()

    # 1e-3 
    marker_xy = cartesian(
        np.linspace(-9e-3, 9e-3, 19),
        np.linspace(-9e-3, 9e-3, 19)
    )
    marker_xy -= np.asarray([0.5e-3, 0.5e-3])

    gelslim_mesh = trimesh.load(os.path.join(args.gel_folder, args.gel_surf_obj_file))
    gelslim_mesh.vertices /= 1e3

    num_markers = marker_xy.shape[0]
    virtual_marker_xyz = np.concatenate([
        marker_xy, np.ones((num_markers, 1)) * -0.02
    ], axis=1)
    if args.rot_z != 0.:
        rot_mat = R.from_euler('xyz', (0., 0., args.rot_z)).as_matrix()
        virtual_marker_xyz = virtual_marker_xyz @ rot_mat.T
    marker_xy = virtual_marker_xyz[:, :2]
    np.save(os.path.join(args.gel_folder, 'marker_xy.npy'), marker_xy)
    ray_dir = np.asarray([0., 0., 1.])[None, :].repeat(num_markers, axis=0)
    marker_pcd = trimesh.PointCloud(virtual_marker_xyz)

    intersector = RayMeshIntersector(gelslim_mesh)
    intersected_face_ids = intersector.intersects_first(
        virtual_marker_xyz, ray_dir)
    filtered_faces = gelslim_mesh.faces[intersected_face_ids]

    locations, ray_inds, tri_inds = intersector.intersects_location(
        virtual_marker_xyz, ray_dir)
    filtered_locations = [None, ] * num_markers
    for idx, (location, ray_ind, tri_ind) in enumerate(
            zip(locations, ray_inds, tri_inds)):
        if tri_ind == intersected_face_ids[ray_ind] or point_tri_edge_dist(
                location, gelslim_mesh.vertices[filtered_faces[ray_ind]]) < 1e-6:
            filtered_locations[ray_ind] = location
    for loc in filtered_locations:
        if loc is None:
            print('ray-intersection got wrong results!')

    bc_ws = []
    for face, point in zip(filtered_faces, filtered_locations):
        bc_ws.append(bc_weight(gelslim_mesh.vertices[face], point))

    if args.vis:
        recovered_points = []
        for face, ws in zip(filtered_faces, bc_ws):
            face_verts = gelslim_mesh.vertices[face]
            recovered_points.append(
                ws[0] * face_verts[0] + ws[1] * face_verts[1] + ws[2] *
                face_verts[2])

        recovered_from_bc_ws_pcd = trimesh.PointCloud(recovered_points)
        filtered_mesh = trimesh.Trimesh(
            np.asarray(gelslim_mesh.vertices),
            filtered_faces)
        filtered_pcd = trimesh.PointCloud(filtered_locations)
        # recovered_from_bc_ws_pcd
        trimesh_show([filtered_mesh, marker_pcd, filtered_pcd])

    with open(os.path.join(args.gel_folder, 'marker_bc_ws.pkl'), 'wb') as f:
        pkl.dump({
            'tri_inds': np.asarray(intersected_face_ids),
            'bc_ws': np.asarray(bc_ws),
            'pos_xy': np.asarray(marker_xy)
        }, f)
