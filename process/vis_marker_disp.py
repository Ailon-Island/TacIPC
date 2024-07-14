import os 
import json 
import trimesh 
from glob import glob 
import pickle as pkl 
import numpy as np 
import cv2
import imageio
import sys 
sys.path.append('.')
from argparse import ArgumentParser 

parser = ArgumentParser()
parser.add_argument('--gel_folder', type=str, default="resources/gel/gelslim-gel-l_0.1")

def get_frame(path):
    return int(os.path.splitext(os.path.basename(path))[0].split('_')[-1])

if __name__ == '__main__':
    args = parser.parse_args()

    for exp_path in glob('output/experiments_markers/*'):
        print(f"Processing {exp_path}")
        config_path = os.path.join(exp_path, 'config.json')
        gel_folder = args.gel_folder  
        if os.path.exists(config_path):
            with open(os.path.join(exp_path, 'config.json'), 'r') as f:
                config = json.load(f)
            if 'gel_pth' in config:
                gel_folder = os.path.dirname(config['gel_pth'])

        marker_xy_path = os.path.join(gel_folder, 'marker_xy.npy')
        marker_xy = np.load(marker_xy_path)

        markers_dir = os.path.join(exp_path, 'markers')
        markers_path = glob(os.path.join(markers_dir, '*.pkl'))
        markers_path.sort(key=get_frame) # sort by frame number

            # out_pkl_path = os.path.join(exp_path, 'markers', f'{id}_{frame}.pkl')
        vis = []
        for marker_path in markers_path:
            marker_name = os.path.splitext(os.path.basename(marker_path))[0]
            frame = marker_name.split('_')[-1]
            with open(marker_path, 'rb') as f:
                marker_info = pkl.load(f)
            marker_disp = marker_info['marker_disp']

            marker_vis = np.zeros((800, 800, 3), dtype=np.uint8)
            for i in range(marker_disp.shape[0]):
                x, y = marker_xy[i] * 100
                x, y = int(x * 300 + 400), int(y * 300 + 400)
                dx, dy, dz = marker_disp[i] * 500
                r = int(np.linalg.norm(marker_disp[i]) * 30000)
                ii, jj = i // 19, i % 19
                if 8 <= ii <= 12 and 8 <= jj <= 11:
                    cv2.circle(marker_vis, (x, y), 3, (0, 0, 255), -1)
                else: 
                    cv2.circle(marker_vis, (x, y), 3, (0, 255, 0), -1)
                cv2.arrowedLine(marker_vis, (x, y), (x + int(dx), y + int(dy)), (r, 0, 255-r), 3)
            vis.append(marker_vis)
        out_vid_path = os.path.join(exp_path, f'marker.mp4')
        imageio.mimsave(out_vid_path, vis, fps=3)
            

            