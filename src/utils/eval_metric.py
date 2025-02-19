"""
# 
# Created: 2024-04-14 11:57
# Copyright (C) 2024-now, RPL, KTH Royal Institute of Technology
# Author: Qingwen Zhang  (https://kin-zhang.github.io/)
# 
#
# Reference to official evaluation scripts:
# - EPE Threeway: https://github.com/argoverse/av2-api/blob/main/src/av2/evaluation/scene_flow/eval.py
# - Bucketed EPE: https://github.com/kylevedder/BucketedSceneFlowEval/blob/master/bucketed_scene_flow_eval/eval/bucketed_epe.py
"""

import torch
import os, sys
from pathlib import Path
import numpy as np
from typing import Dict, Final, List, Tuple
from tabulate import tabulate
from collections import defaultdict

from av2.structures.cuboid import CuboidList
from av2.structures.sweep import Sweep
from av2.datasets.sensor.av2_sensor_dataloader import convert_pose_dataframe_to_SE3
from av2.map.map_api import ArgoverseStaticMap
from av2.utils.io import read_feather

BASE_DIR = os.path.abspath(os.path.join( os.path.dirname( __file__ ), '../..' ))
BOUNDING_BOX_EXPANSION: Final = 0.2

SIZE_BUCKET_BOUNDARIES = torch.tensor([1, 2.5, 4.5, 6, 9, 12])
SIZE_CLASSES = ['T', 'XS', 'S', 'M', 'L', 'XL', 'U']
sys.path.append(BASE_DIR)
from src.utils.av2_eval import compute_metrics, compute_bucketed_epe, CLOSE_DISTANCE_THRESHOLD


# EPE Three-way: Foreground Dynamic, Background Dynamic, Background Static
# leaderboard link: https://eval.ai/web/challenges/challenge-page/2010/evaluation
def evaluate_leaderboard(est_flow, rigid_flow, pc0, gt_flow, is_valid, pts_ids):
    gt_is_dynamic = torch.linalg.vector_norm(gt_flow - rigid_flow, dim=-1) >= 0.05
    mask_ = ~est_flow.isnan().any(dim=1) & ~rigid_flow.isnan().any(dim=1) & ~pc0[:, :3].isnan().any(dim=1) & ~gt_flow.isnan().any(dim=1)
    mask_no_nan = mask_ & ~gt_is_dynamic.isnan() & ~is_valid.isnan() & ~pts_ids.isnan()
    est_flow = est_flow[mask_no_nan, :]
    rigid_flow = rigid_flow[mask_no_nan, :]
    pc0 = pc0[mask_no_nan, :]
    gt_flow = gt_flow[mask_no_nan, :]
    gt_is_dynamic = gt_is_dynamic[mask_no_nan]
    is_valid = is_valid[mask_no_nan]
    pts_ids = pts_ids[mask_no_nan]

    est_is_dynamic = torch.linalg.vector_norm(est_flow - rigid_flow, dim=-1) >= 0.05
    is_close = torch.all(torch.abs(pc0[:, :2]) <= CLOSE_DISTANCE_THRESHOLD, dim=1)
    res_dict = compute_metrics(
        est_flow.detach().cpu().numpy().astype(float),
        est_is_dynamic.detach().cpu().numpy().astype(bool),
        gt_flow.detach().cpu().numpy().astype(float),
        pts_ids.detach().cpu().numpy().astype(np.uint8),
        gt_is_dynamic.detach().cpu().numpy().astype(bool),
        is_close.detach().cpu().numpy().astype(bool),
        is_valid.detach().cpu().numpy().astype(bool)
    )
    return res_dict


# EPE Three-way: Foreground Dynamic, Background Dynamic, Background Static
# leaderboard link: https://eval.ai/web/challenges/challenge-page/2010/evaluation
def evaluate_size_bucketed(est_flow, rigid_flow, pc0, gt_flow, is_valid, pts_ids, sample_dict):

    scene_id = sample_dict['scene_id']
    timestamp = sample_dict['timestamp']
    eval_mask = sample_dict['eval_mask'].detach().cpu().numpy().astype(bool)
    data_root = 'data/Argoverse2/sensor/val'
    # Load 3D cuboid 
    
    annotation_feather_path = os.path.join(data_root, scene_id, 'annotations.feather')
    cuboid_list_scene = CuboidList.from_feather(annotation_feather_path)
    # res_dict = dict(
    #     Static=[],
    #     Dynamic=[]
    # )
    sweep = Sweep.from_feather(Path(data_root)/ scene_id / 'sensors'/ 'lidar'/ f'{timestamp}.feather')
    pc0_ego = sweep.xyz
    
    res_dict = defaultdict(dict)
    
    for key in SIZE_CLASSES:
        for sub_key in ['Static', 'Dynamic']:
            res_dict[key][sub_key] = []
        res_dict[key]['num'] = 0
    # print('timestamp:', type(timestamp))
    for cuboid in cuboid_list_scene:
        if cuboid.timestamp_ns == int(timestamp):

            cuboid.length_m += BOUNDING_BOX_EXPANSION
            cuboid.width_m += BOUNDING_BOX_EXPANSION
            _, obj_mask = cuboid.compute_interior_points(pc0_ego)
            # print('***')
            # print('obj mask before:', obj_mask.shape, obj_mask.sum())
            obj_mask = obj_mask[eval_mask]
            # print('eval mask:', eval_mask.shape, eval_mask.sum())
            # print('obj mask after:', obj_mask.shape, obj_mask.sum())
            if obj_mask.sum() == 0:
                continue
            est_flow_obj = est_flow[obj_mask] 
            rigid_flow_obj = rigid_flow[obj_mask]
            pc0_obj = pc0[obj_mask]
            gt_flow_obj = gt_flow[obj_mask]
            is_valid_obj = is_valid[obj_mask]
            pts_ids_obj = pts_ids[obj_mask]
            eval_dict = evaluate_leaderboard(est_flow_obj, rigid_flow_obj, pc0_obj, gt_flow_obj, is_valid_obj, pts_ids_obj)
            # print(timestamp, 'FS:', eval_dict['EPE_FS'])
            # print(timestamp, 'FD:', eval_dict['EPE_FD'])
            class_id = np.digitize(cuboid.length_m, SIZE_BUCKET_BOUNDARIES, right=True)
            class_name = SIZE_CLASSES[class_id]
            # print('***')
            # if eval_dict['EPE_FS'] != 0 and eval_dict['EPE_FD'] != 0:
            #     print(f'{timestamp}, FS: {eval_dict["EPE_FS"]}, FD: {eval_dict["EPE_FD"]}.')
            # assert not (eval_dict['EPE_FS'] == 0 and eval_dict['EPE_FD'] == 0) 
            if eval_dict['EPE_FS'] != 0:
                res_dict[class_name]['Static'].append(eval_dict['EPE_FS']) 
            if eval_dict['EPE_FD'] != 0:
                res_dict[class_name]['Dynamic'].append(eval_dict['EPE_FD'])
            if (eval_dict['EPE_FS'] == 0) and (eval_dict['EPE_FD'] == 0):
                gt_is_dynamic = torch.linalg.vector_norm(gt_flow_obj - rigid_flow_obj, dim=-1) >= 0.05
                # print('GT dynamic:', gt_is_dynamic, gt_is_dynamic.all())
                if gt_is_dynamic.all():
                    res_dict[class_name]['Dynamic'].append(eval_dict['EPE_FD'])
                else:
                    res_dict[class_name]['Static'].append(eval_dict['EPE_FS']) 
                
            #if (eval_dict['EPE_FS'] != 0) and (eval_dict['EPE_FD'] != 0):
                #print(f'{timestamp}, FS: {eval_dict["EPE_FS"]}, FD: {eval_dict["EPE_FD"]}.')
            res_dict[class_name]['num'] += 1
    return res_dict

# EPE Bucketed: BACKGROUND, CAR, PEDESTRIAN, WHEELED_VRU, OTHER_VEHICLES
def evaluate_leaderboard_v2(est_flow, rigid_flow, pc0, gt_flow, is_valid, pts_ids):
    # in x,y dis, ref to official evaluation: eval/base_per_frame_sceneflow_eval.py#L118-L119
    pc_distance = torch.linalg.vector_norm(pc0[:, :2], dim=-1)
    distance_mask = pc_distance <= CLOSE_DISTANCE_THRESHOLD

    mask_flow_non_nan = ~est_flow.isnan().any(dim=1) & ~rigid_flow.isnan().any(dim=1) & ~pc0[:, :3].isnan().any(dim=1) & ~gt_flow.isnan().any(dim=1)
    mask_eval = mask_flow_non_nan & ~is_valid.isnan() & ~pts_ids.isnan() & distance_mask
    rigid_flow = rigid_flow[mask_eval, :]
    est_flow = est_flow[mask_eval, :] - rigid_flow
    gt_flow = gt_flow[mask_eval, :] - rigid_flow # in v2 evaluation, we don't add rigid flow to evaluate
    is_valid = is_valid[mask_eval]
    pts_ids = pts_ids[mask_eval]

    res_dict = compute_bucketed_epe(
        est_flow.detach().cpu().numpy().astype(float),
        gt_flow.detach().cpu().numpy().astype(float),
        pts_ids.detach().cpu().numpy().astype(np.uint8),
        is_valid.detach().cpu().numpy().astype(bool),
    )
    return res_dict

# reference to official evaluation: bucketed_scene_flow_eval/eval/bucketed_epe.py
# python >= 3.7
from dataclasses import dataclass
import warnings
@dataclass(frozen=True, eq=True, repr=True)
class OverallError:
    static_epe: float
    dynamic_error: float

    def __repr__(self) -> str:
        static_epe_val_str = (
            f"{self.static_epe:0.6f}" if np.isfinite(self.static_epe) else f"{self.static_epe}"
        )
        dynamic_error_val_str = (
            f"{self.dynamic_error:0.6f}"
            if np.isfinite(self.dynamic_error)
            else f"{self.dynamic_error}"
        )
        return f"({static_epe_val_str}, {dynamic_error_val_str})"

    def to_tuple(self) -> Tuple[float, float]:
        return (self.static_epe, self.dynamic_error)

class BucketResultMatrix:
    def __init__(self, class_names: List[str], speed_buckets: List[Tuple[float, float]]):
        self.class_names = class_names
        self.speed_buckets = speed_buckets

        assert (
            len(self.class_names) > 0
        ), f"class_names must have at least one entry, got {len(self.class_names)}"
        assert (
            len(self.speed_buckets) > 0
        ), f"speed_buckets must have at least one entry, got {len(self.speed_buckets)}"

        # By default, NaNs are not counted in np.nanmean
        self.epe_storage_matrix = np.zeros((len(class_names), len(self.speed_buckets))) * np.NaN
        self.speed_storage_matrix = np.zeros((len(class_names), len(self.speed_buckets))) * np.NaN
        self.count_storage_matrix = np.zeros(
            (len(class_names), len(self.speed_buckets)), dtype=np.int64
        )

    def accumulate_value(
        self,
        class_name: str,
        speed_bucket: Tuple[float, float],
        average_epe: float,
        average_speed: float,
        count: int,
    ):
        assert count > 0, f"count must be greater than 0, got {count}"
        assert np.isfinite(average_epe), f"average_epe must be finite, got {average_epe}"
        assert np.isfinite(average_speed), f"average_speed must be finite, got {average_speed}"

        class_idx = self.class_names.index(class_name)
        speed_bucket_idx = self.speed_buckets.index(speed_bucket)

        prior_epe = self.epe_storage_matrix[class_idx, speed_bucket_idx]
        prior_speed = self.speed_storage_matrix[class_idx, speed_bucket_idx]
        prior_count = self.count_storage_matrix[class_idx, speed_bucket_idx]

        if np.isnan(prior_epe):
            self.epe_storage_matrix[class_idx, speed_bucket_idx] = average_epe
            self.speed_storage_matrix[class_idx, speed_bucket_idx] = average_speed
            self.count_storage_matrix[class_idx, speed_bucket_idx] = count
            return

        # Accumulate the average EPE and speed, weighted by the number of samples using np.mean
        self.epe_storage_matrix[class_idx, speed_bucket_idx] = np.average(
            [prior_epe, average_epe], weights=[prior_count, count]
        )
        self.speed_storage_matrix[class_idx, speed_bucket_idx] = np.average(
            [prior_speed, average_speed], weights=[prior_count, count]
        )
        self.count_storage_matrix[class_idx, speed_bucket_idx] += count

    def get_normalized_error_matrix(self) -> np.ndarray:
        error_matrix = self.epe_storage_matrix.copy()
        # For the 1: columns, normalize EPE entries by the speed
        error_matrix[:, 1:] = error_matrix[:, 1:] / self.speed_storage_matrix[:, 1:]
        return error_matrix

    def get_overall_class_errors(self, normalized: bool = True):
        #  -> dict[str, OverallError]
        if normalized:
            error_matrix = self.get_normalized_error_matrix()
        else:
            error_matrix = self.epe_storage_matrix.copy()
        static_epes = error_matrix[:, 0]
        # Hide the warning about mean of empty slice
        # I expect to see RuntimeWarnings in this block
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            dynamic_errors = np.nanmean(error_matrix[:, 1:], axis=1)

        return {
            class_name: OverallError(static_epe, dynamic_error)
            for class_name, static_epe, dynamic_error in zip(
                self.class_names, static_epes, dynamic_errors
            )
        }

    def get_class_entries(self, class_name: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        class_idx = self.class_names.index(class_name)

        epe = self.epe_storage_matrix[class_idx, :]
        speed = self.speed_storage_matrix[class_idx, :]
        count = self.count_storage_matrix[class_idx, :]
        return epe, speed, count

    def get_mean_average_values(self, normalized: bool = True) -> OverallError:
        overall_errors = self.get_overall_class_errors(normalized=normalized)

        average_static_epe = np.nanmean([v.static_epe for v in overall_errors.values()])
        average_dynamic_error = np.nanmean([v.dynamic_error for v in overall_errors.values()])

        return OverallError(average_static_epe, average_dynamic_error)

class OfficialMetrics:
    def __init__(self, eval_size_bucketed=False):
        # same with BUCKETED_METACATAGORIES
        self.bucketed= {
            'BACKGROUND': {'Static': [], 'Dynamic': []},
            'CAR': {'Static': [], 'Dynamic': []},
            'OTHER_VEHICLES': {'Static': [], 'Dynamic': []},
            'PEDESTRIAN': {'Static': [], 'Dynamic': []},
            'WHEELED_VRU': {'Static': [], 'Dynamic': []},
            'Mean': {'Static': [], 'Dynamic': []}
        }

        self.epe_3way = {
            'EPE_FD': [],
            'EPE_BS': [],
            'EPE_FS': [],
            'IoU': [],
            'Three-way': []
        }
        
        self.eval_size_bucketed = eval_size_bucketed
        if self.eval_size_bucketed:
            self.size_bucket_epe = {
                'T': {'Static':[], 'Dynamic':[], 'num': 0}, # 0.0 - 1.0 m Pedestrians
                'XS': {'Static':[], 'Dynamic':[], 'num': 0},  # 1.0 - 2.5 m Bicycles/e‐bikes, Standing scooters, wheelchairs, etc.
                'S': {'Static':[], 'Dynamic':[], 'num': 0},   # 2.5 - 4.5 m Motorcycles (with riders), Small cars (hatchbacks, subcompacts, city cars)
                'M': {'Static':[], 'Dynamic':[], 'num': 0},   # 4.5 - 6.0 m Sedans, SUVs, pickup trucks, Minivans
                'L': {'Static':[], 'Dynamic':[], 'num': 0},   # 6.0 - 9.0 m: Small box trucks, Large vans (delivery vans, small shuttle vans)
                'XL': {'Static':[], 'Dynamic':[], 'num': 0},  # 9.0 - 12.0 m: Standard city buses, Medium trucks (e.g., straight trucks, mid‐size box trucks)
                'U': {'Static':[], 'Dynamic':[], 'num': 0},   # > 12.0 m: Tractor‐trailers (semi‐trucks), Articulated/tandem buses (if present)
            }

        self.norm_flag = False


        # bucket_max_speed, num_buckets, distance_thresholds set is from: eval/bucketed_epe.py#L226
        bucket_edges = np.concatenate([np.linspace(0, 2.0, 51), [np.inf]])
        speed_thresholds = list(zip(bucket_edges, bucket_edges[1:]))
        self.bucketedMatrix = BucketResultMatrix(
            class_names=['BACKGROUND', 'CAR', 'OTHER_VEHICLES', 'PEDESTRIAN', 'WHEELED_VRU'],
            speed_buckets=speed_thresholds
        )
    def step(self, epe_dict, bucket_dict, size_bucket_dict):
        """
        This step function is used to store the results of **each frame**.
        """
        for key in epe_dict:
            self.epe_3way[key].append(epe_dict[key])

        if self.eval_size_bucketed:
            assert size_bucket_dict is not None
            for key in size_bucket_dict:
                self.size_bucket_epe[key]['Static'].extend(size_bucket_dict[key]['Static'])
                self.size_bucket_epe[key]['Dynamic'].extend(size_bucket_dict[key]['Dynamic'])
                self.size_bucket_epe[key]['num'] += size_bucket_dict[key]['num']
                
        for item_ in bucket_dict:
            if item_.count == 0:
                continue
            category_name = item_.name
            speed_tuple = item_.speed_thresholds
            self.bucketedMatrix.accumulate_value(
                category_name,
                speed_tuple,
                item_.avg_epe,
                item_.avg_speed,
                item_.count,
            )

    def normalize(self):
        """
        This normalize mean average results between **frame and frame**.
        """
        for key in self.epe_3way:
            self.epe_3way[key] = np.mean(self.epe_3way[key])
        self.epe_3way['Three-way'] = np.mean([self.epe_3way['EPE_FD'], self.epe_3way['EPE_BS'], self.epe_3way['EPE_FS']])
        
        if self.eval_size_bucketed:
            for key in self.size_bucket_epe:
                self.size_bucket_epe[key]['Static'] = np.mean(self.size_bucket_epe[key]['Static'])
                self.size_bucket_epe[key]['Dynamic'] = np.mean(self.size_bucket_epe[key]['Dynamic'])
            
        mean = self.bucketedMatrix.get_mean_average_values(normalized=True).to_tuple()
        class_errors = self.bucketedMatrix.get_overall_class_errors(normalized=True)
        for key in self.bucketed:
            if key == 'Mean':
                self.bucketed[key]['Static'] = mean[0]
                self.bucketed[key]['Dynamic'] = mean[1]
                continue
            for i, sub_key in enumerate(self.bucketed[key]):
                self.bucketed[key][sub_key] = class_errors[key].to_tuple()[i] # 0: static, 1: dynamic
        self.norm_flag = True
    
    def print(self):
        if not self.norm_flag:
            self.normalize()
        printed_data = []
        for key in self.epe_3way:
            printed_data.append([key,self.epe_3way[key]])
        print("Version 1 Metric on EPE Three-way:")
        print(tabulate(printed_data), "\n")

        printed_data = []
        for key in self.bucketed:
            printed_data.append([key, self.bucketed[key]['Static'], self.bucketed[key]['Dynamic']])
        print("Version 2 Metric on Category-based:")
        print(tabulate(printed_data, headers=["Class", "Static", "Dynamic", "Number"], tablefmt='orgtbl'), "\n")
        
        if self.eval_size_bucketed:
            printed_data = []
            for key in self.size_bucket_epe:
                printed_data.append([key,  np.round(self.size_bucket_epe[key]['Dynamic'], 6), np.round(self.size_bucket_epe[key]['Static'], 6), self.size_bucket_epe[key]['num']])
            print('FD and FS Metric on Size Bucketed:')
            print(tabulate(printed_data, headers=["Class", "Dynamic", "Static", "Number"], tablefmt='orgtbl'), "\n")

            metric_description =[
                ['T',  '(0.0 - 1.0] m', 'Pedestrians'],
                ['XS', '(1.0 - 2.5] m', 'Bicycles/e‐bikes, Standing scooters, wheelchairs, etc.'],
                ['S',  '(2.5 - 4.5] m', 'Motorcycles (with riders), Small cars (hatchbacks, subcompacts, city cars)'],
                ['M',  '(4.5 - 6.0] m', 'Sedans, SUVs, pickup trucks, Minivans'],
                ['L',  '(6.0 - 9.0] m', 'Small box trucks, Large vans (delivery vans, small shuttle vans)'],
                ['XL', '(9.0 - 12.0] m', 'Standard city buses, Medium trucks (e.g., straight trucks, mid‐size box trucks)'],
                ['U',  '(> 12.0 m', 'Tractor‐trailers (semi‐trucks), Articulated/tandem buses (if present)'],
            ]
            print('Size Bucketed Metric Description:')
            print(tabulate(metric_description, headers=["Class", "Range", "Description"], tablefmt='orgtbl'), "\n")