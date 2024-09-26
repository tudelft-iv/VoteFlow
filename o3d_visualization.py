
import numpy as np
import fire, time
import pickle 
import h5py
from tqdm import tqdm

import open3d as o3d
import os, sys
from src.models.model_utils.util_visualization import visualize_pcd


def visualize_pcd_list(pc_dict_list):
    assert len(pc_dict_list) < 4, len(pc_dict_list)
    color_list = []
    name_list = []
    pc_list = []
    for idx, dic in enumerate(pc_dict_list):
        pc_list.append(dic['pc'])
        color_list.append(np.zeros(len(dic['pc'])) + idx)
        name_list.append(dic['name'])
    visualize_pcd(
        np.concatenate(pc_list, axis=0),
        np.concatenate(color_list, axis=0),
    num_colors=3, 
    title=f'visualize input: {name_list[0]}-r, {name_list[1]}-g')


def vis(index,
        load_dict: bool = False,
        save_reformulated_vis: bool = False,
        save_dir = 'outputs/reformulated_vis_outputs',
        vis_name: str = 'seflow_best',
        data_dir: str ="data/argoverse2_demo/sensor/val"):
    data_index = pickle.load(open(os.path.join(data_dir, 'index_total.pkl'), 'rb'))
    scene_id, timestamp = data_index[index]
    
    key = str(timestamp)
    data_dict = {
        'scene_id': scene_id,
        'timestamp': timestamp,
    }
    
    print(vis_name)
    if load_dict:
        print(f"Load from dict")
        pkl_file_path = os.path.join(save_dir, scene_id, f'{timestamp}.pkl')
        data_dict = pickle.load(open(pkl_file_path, 'rb'))
    else:
        print(f"Load from hdf5:")
        with h5py.File(os.path.join(data_dir, f'{scene_id}.h5'), 'r') as f:
            print(f[key].keys())
            gt_flow = f[key]['flow'][:]
            print('gt flow:', gt_flow.shape)    
            data_dict['pc0'] = f[key]['lidar'][:]
            data_dict['gm0'] = f[key]['ground_mask'][:]
            data_dict['pose0'] = f[key]['pose'][:]
            for flow_key in [vis_name, 'dufo_label', 'label']:
                if flow_key in f[key]:
                    data_dict[flow_key] = f[key][flow_key][:]
            next_timestamp = str(data_index[index+1][1])
            data_dict['pose1'] = f[next_timestamp]['pose'][:]
            data_dict['pc1'] = f[next_timestamp]['lidar'][:]
            data_dict['gm1'] = f[next_timestamp]['ground_mask'][:]
        
        if save_reformulated_vis: 
            sub_dir = os.path.join(save_dir,f'{scene_id}')
            if not os.path.exists(sub_dir):
                os.makedirs(sub_dir)
            file = os.path.join(sub_dir,f'{timestamp}.pkl')
            pickle.dump(data_dict, open(file, 'wb'))
    
    pc0 = data_dict['pc0']
    gm0 = data_dict['gm0']
    pc1 = data_dict['pc1']
    gm1 = data_dict['gm1']

    pose0 = data_dict['pose0'] # ego2city
    pose1 = data_dict['pose1'] # ego2city

    ego0t1 = np.linalg.inv(pose1) @ pose0
    pose_flow = pc0[:, :3] @ ego0t1[:3, :3].T + ego0t1[:3, 3] - pc0[:, :3] ## transform pc0 to pc0 after ego motion compensation
    
    flow = data_dict[vis_name] - pose_flow # ego motion compensation here.
    transform_pc0 = pc0[:, :3] @ ego0t1[:3, :3].T + ego0t1[:3, 3]
    
    transform_pc0_no_ground = transform_pc0[~gm0]
    pc0_no_ground = pc0[~gm0]
    pc1_no_ground = pc1[~gm1]
    pc0_flow_no_ground = (pc0 + data_dict[vis_name])[~gm0] ## why?
    pc0_gt_flow_no_ground = (pc0 + gt_flow)[~gm0] ## why?
    print(f"scene_id: {scene_id}, timestamp: {timestamp}")
    print(f"pc0: {pc0.shape}, gm0: {gm0.shape}, pc1: {pc1.shape}, gm1: {gm1.shape}")
    print(f"pc0_no_ground: {transform_pc0_no_ground.shape}")
    print(f"pc1_no_ground: {pc1_no_ground.shape}")

    print(f"flow: {flow.shape}, flow_max: {flow.max()}, flow_min: {flow.min()}")    
    mask = np.linalg.norm(data_dict[vis_name][~gm0], axis = -1) > 3.33
    print('mask > 3.33:', sum(mask))
    pc_list = [
        # dict(
        #     pc = transform_pc0_no_ground,
        #     name = 'pc0'
        # ),

        dict(
            pc = pc0_flow_no_ground,
            name = 'pc0+flow'
        ) ,
        
        dict(
            pc = pc0_gt_flow_no_ground,
            name = 'pc0+gt_flow'
        ),
    ]
    visualize_pcd_list(pc_list)

if __name__ == '__main__':
    start_time = time.time()
    fire.Fire(vis)
    print(f"Time used: {time.time() - start_time:.2f} s")