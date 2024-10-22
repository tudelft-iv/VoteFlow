import os
import glob
import argparse
import numpy as np
import torch

def collate(batch, fixed_length=80000):
    # print('batch #############: ', len(batch))
    points_src = []
    points_dst = []
    flows = []
    for b in batch:
        point_src = b['point_src']
        point_dst = b['point_dst']
        scene_flow = b['scene_flow']
        if b['is_test'] is True:
            pass
        elif b['is_train'] is True:
            # downsample to fixed length
            idxs = np.random.choice(np.arange(0, len(point_src)), fixed_length, replace=True)
            point_src = point_src[idxs]
            scene_flow = scene_flow[idxs]
            idxs = np.random.choice(np.arange(0, len(point_dst)), fixed_length, replace=True)
            point_dst = point_dst[idxs]
        else: 
            NotImplementedError
        points_src.append(point_src)
        points_dst.append(point_dst)
        flows.append(scene_flow)

    data_dict = {
        'point_src': np.stack(points_src, axis=0), 
        'point_dst': np.stack(points_dst, axis=0), 
        'scene_flow': np.stack(flows, axis=0),
    }
    # for _, (k, v) in enumerate(data_dict.items()):
    #     print('collate check: ', k, v.shape)
    return data_dict

def ego_motion_compensation(points, pose):
    """
    Input (torch.Tensor):
        points:         [N, 3]
        time_indice:    [N]
        tsfm:           [n_frames, 4, 4]
    """
    # print('ego motion: ', points.shape, pose.shape)
    R, t = pose[:3,:3], pose[:3,3:4]
    rec_points = np.einsum('ij,jk -> ik', R, points.T) + t
    return rec_points.T

class Dataset_dummy(torch.utils.data.Dataset):
    def __init__(self, data_path, partition='train'):
        self.partition = partition
        self.files = self.meta_data_pca(data_path, partition)
        print(f'number of files: {len(self.files)}')
        self.background_idxes = [
            CATEGORY_NAME_TO_IDX[cat] for cat in METACATAGORIES["BACKGROUND"]
        ]
        print(f'background idx: {self.background_idxes}')

    def meta_data_pca(self, data_path, partition):
        infos = glob.glob(os.path.join(data_path, '*.npz'))
        infos.sort()
        if partition == 'train':
            # infos = infos[2:3]
            infos = infos[0:-3]
        elif partition == 'test':
            infos = infos[-3:]
        else:
            NotImplementedError

        print(f'infos, total number of test sequences: {len(infos)}')
        return infos

    def load_data_pca(self, data_path):
        data_info = dict(np.load(data_path))
        pcl_0 = data_info['pc1']
        pcl_1 = data_info['pc2']
        valid_0 = data_info['pc1_flows_valid_idx']
        valid_1 = data_info['pc2_flows_valid_idx']
        ground_0 = data_info['ground1']
        ground_1 = data_info['ground2']
        flow_0_1 = data_info['gt_flow_0_1']
        flow_1_0 = data_info['gt_flow_1_0']

        # pcl_0 = pcl_0[valid_0]
        # pcl_1 = pcl_1[valid_1]
        # flow_0_1 = flow_0_1[valid_0]
        # visualize_pcd(
        #     np.concatenate([pcl_0, pcl_1, pcl_0+flow_0_1], axis=0),
        #     np.concatenate([np.zeros(len(pcl_0))+1, np.zeros(len(pcl_1))+2, np.zeros(len(pcl_0))+0], axis=0),
        #     num_colors=3, 
        #     title=f'sanity check: src-g, dst-b, src+flow-r: {data_path}'
        #     )
        data_dict = {
            'point_src': pcl_0, 
            'point_dst': pcl_1, 
            'scene_flow': flow_0_1,
            'data_path': data_path,
            'is_test': self.partition=='test',
            'is_train': self.partition=='train',
        }

        return data_dict

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # print(f'idx: {idx} / {len(self.files)}, {self.files[idx]}')
        data = self.load_data_pca(self.files[idx])
        return data

CATEGORY_ID_TO_NAME = {
    -1: 'BACKGROUND',
    0: 'ANIMAL',
    1: 'ARTICULATED_BUS',
    2: 'BICYCLE',
    3: 'BICYCLIST',
    4: 'BOLLARD',
    5: 'BOX_TRUCK',
    6: 'BUS',
    7: 'CONSTRUCTION_BARREL',
    8: 'CONSTRUCTION_CONE',
    9: 'DOG',
    10: 'LARGE_VEHICLE',
    11: 'MESSAGE_BOARD_TRAILER',
    12: 'MOBILE_PEDESTRIAN_CROSSING_SIGN',
    13: 'MOTORCYCLE',
    14: 'MOTORCYCLIST',
    15: 'OFFICIAL_SIGNALER',
    16: 'PEDESTRIAN',
    17: 'RAILED_VEHICLE',
    18: 'REGULAR_VEHICLE',
    19: 'SCHOOL_BUS',
    20: 'SIGN',
    21: 'STOP_SIGN',
    22: 'STROLLER',
    23: 'TRAFFIC_LIGHT_TRAILER',
    24: 'TRUCK',
    25: 'TRUCK_CAB',
    26: 'VEHICULAR_TRAILER',
    27: 'WHEELCHAIR',
    28: 'WHEELED_DEVICE',
    29: 'WHEELED_RIDER'
}

CATEGORY_NAME_TO_IDX = {
    v: idx
    for idx, (_, v) in enumerate(sorted(CATEGORY_ID_TO_NAME.items()))
}

SPEED_BUCKET_SPLITS_METERS_PER_SECOND = [0, 0.5, 2.0, np.inf]
ENDPOINT_ERROR_SPLITS_METERS = [0, 0.05, 0.1, np.inf]

BACKGROUND_CATEGORIES = [
    'BOLLARD', 'CONSTRUCTION_BARREL', 'CONSTRUCTION_CONE',
    'MOBILE_PEDESTRIAN_CROSSING_SIGN', 'SIGN', 'STOP_SIGN'
]
PEDESTRIAN_CATEGORIES = [
    'PEDESTRIAN', 'STROLLER', 'WHEELCHAIR', 'OFFICIAL_SIGNALER'
]
SMALL_VEHICLE_CATEGORIES = [
    'BICYCLE', 'BICYCLIST', 'MOTORCYCLE', 'MOTORCYCLIST', 'WHEELED_DEVICE',
    'WHEELED_RIDER'
]
VEHICLE_CATEGORIES = [
    'ARTICULATED_BUS', 'BOX_TRUCK', 'BUS', 'LARGE_VEHICLE', 'RAILED_VEHICLE',
    'REGULAR_VEHICLE', 'SCHOOL_BUS', 'TRUCK', 'TRUCK_CAB', 'VEHICULAR_TRAILER',
    'TRAFFIC_LIGHT_TRAILER', 'MESSAGE_BOARD_TRAILER'
]
ANIMAL_CATEGORIES = ['ANIMAL', 'DOG']

METACATAGORIES = {
    "BACKGROUND": BACKGROUND_CATEGORIES,
    "PEDESTRIAN": PEDESTRIAN_CATEGORIES,
    "SMALL_MOVERS": SMALL_VEHICLE_CATEGORIES,
    "LARGE_MOVERS": VEHICLE_CATEGORIES
}

METACATAGORY_TO_SHORTNAME = {
    "BACKGROUND": "BG",
    "PEDESTRIAN": "PED",
    "SMALL_MOVERS": "SMALL",
    "LARGE_MOVERS": "LARGE"
}

