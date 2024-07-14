import os 
import json 
import trimesh 
from glob import glob 
import pickle as pkl 
import json
import numpy as np 
import sys 
sys.path.append('.')
from argparse import ArgumentParser 

parser = ArgumentParser()
parser.add_argument('--gel_folder', type=str, default="resources/gel/gelslim-gel-l_0.1-3mm")
parser.add_argument('--overwrite', action='store_true', default=False)
parser.add_argument('--rest_frame', type=int, default=0)


if __name__ == '__main__':
    args = parser.parse_args()
    for exp_path in glob('output/experiments_markers/*'):
        print(f"Processing {exp_path}")
        gel_folder = args.gel_folder
        rest_frame = args.rest_frame

        config_path = os.path.join(exp_path, 'config.json')
        if os.path.exists(config_path):
            with open(os.path.join(exp_path, 'config.json'), 'r') as f:
                config = json.load(f)
            if 'gel_pth' in config:
                gel_folder = os.path.dirname(config['gel_pth'])
            if 'move_type' in config:
                if config['move_type'] != 'press':
                    if 'press_steps' in config:
                        rest_frame = config['press_steps']
        gel_marker_bc_ws_path = os.path.join(gel_folder, 'marker_bc_ws.pkl')
        with open(gel_marker_bc_ws_path, 'rb') as f:
            bc_ws_info = pkl.load(f)
        tri_inds = bc_ws_info['tri_inds']
        bc_ws = bc_ws_info['bc_ws']

        objs_dir = os.path.join(exp_path, 'objs')
        objs_path = glob(os.path.join(objs_dir, 'gel_*.obj'))
        objs_path.sort() # sort by frame number
        n_objs = len(objs_path)

        for obj_path in objs_path:
            obj_name = os.path.splitext(os.path.basename(obj_path))[0]
            frame = obj_name.split('_')[-1]
            out_pkl_path = os.path.join(exp_path, 'markers', f'{frame}.pkl')
            out_json_path = os.path.join(exp_path, 'markers', f'{frame}.json')
            try: 
                gel_surf_mesh = trimesh.load_mesh(obj_path)
                press_gel_surf_mesh = trimesh.load_mesh(os.path.join(
                    objs_dir, f'gel_{rest_frame}.obj'))


                vert_disp = gel_surf_mesh.vertices - press_gel_surf_mesh.vertices
                tri_vert_inds = gel_surf_mesh.faces[tri_inds]
                marker_tri_disp = np.transpose(vert_disp[tri_vert_inds], (2, 0, 1))
                marker_disp = (marker_tri_disp * bc_ws).sum(axis=2).T
                rest_marker_tri_pos = np.transpose(
                    press_gel_surf_mesh.vertices[tri_vert_inds], (2, 0, 1))
                rest_marker_pos = (
                    rest_marker_tri_pos * bc_ws).sum(axis=2).T 
                
                if not os.path.exists(os.path.dirname(out_pkl_path)):
                    os.makedirs(os.path.dirname(out_pkl_path))
                with open(out_pkl_path, 'wb') as f:
                    pkl.dump({
                        'rest_marker_pos': rest_marker_pos, 
                        'marker_disp': marker_disp
                    }, f)
                with open(out_json_path, 'w') as f:
                    json.dump({
                        'rest_marker_pos': rest_marker_pos.tolist(), 
                        'marker_disp': marker_disp.tolist()
                    }, f, indent=4)
                # print(f'Generated {out_pkl_path}')
            except Exception as e:
                print(e)
            