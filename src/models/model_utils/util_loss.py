import torch
from pytorch3d.ops.knn import knn_points
from pytorch3d.loss import chamfer_distance

class UnSupervisedLoss(torch.nn.Module):

    def __init__(self, dist_threshold=3.33):
        super().__init__()
        self.dist_threshold = dist_threshold

    def forward(self, points_src, points_dst, flows_pred, flows_gt, masks_src, masks_dst):

        warped_loss = 0
        points_src_warped = points_src + flows_pred
        for points_src_warped_, points_dst_, masks_src_, masks_dst_ in zip(points_src_warped, points_dst, masks_src, masks_dst):
            points_src_warped_ = points_src_warped_[masks_src_>0]
            points_dst_ = points_dst_[masks_dst_>0]
            warped_loss += warped_pc_loss(points_src_warped_, points_dst_, dist_threshold=self.dist_threshold)
        return warped_loss

# https://github.com/kylevedder/zeroflow/blob/master/pointclouds/losses/losses.py
def warped_pc_loss(warped_pc: torch.Tensor,
                   target_pc: torch.Tensor,
                   dist_threshold=2.0):
    if warped_pc.ndim == 2:
        warped_pc = warped_pc.unsqueeze(0)
    if target_pc.ndim == 2:
        target_pc = target_pc.unsqueeze(0)

    assert warped_pc.ndim == 3, f"warped_pc.ndim = {warped_pc.ndim}, not 3; shape = {warped_pc.shape}"
    assert target_pc.ndim == 3, f"target_pc.ndim = {target_pc.ndim}, not 3; shape = {target_pc.shape}"

    loss = 0

    if dist_threshold is None:
        loss += chamfer_distance(warped_pc, target_pc,
                                 point_reduction="mean")[0].sum()
        loss += chamfer_distance(target_pc, warped_pc,
                                 point_reduction="mean")[0].sum()
        return loss

    # Compute min distance between warped point cloud and point cloud at t+1.
    warped_to_target_knn = knn_points(p1=warped_pc, p2=target_pc, K=1)
    warped_to_target_distances = warped_to_target_knn.dists[0]
    target_to_warped_knn = knn_points(p1=target_pc, p2=warped_pc, K=1)
    target_to_warped_distances = target_to_warped_knn.dists[0]
    # Throw out distances that are too large (beyond the dist threshold).
    loss += warped_to_target_distances[
        warped_to_target_distances < dist_threshold].mean()

    loss += target_to_warped_distances[
        target_to_warped_distances < dist_threshold].mean()

    return loss


def pc0_to_pc1_distance(pc0: torch.Tensor,
                        pc1: torch.Tensor):

    if pc0.ndim == 2:
        pc0 = pc0.unsqueeze(0)
    if pc1.ndim == 2:
        pc1 = pc1.unsqueeze(0)

    pc0_to_pc1_knn = knn_points(p1=pc0,
                                p2=pc1,
                                lengths1=None,
                                lengths2=None,
                                K=1)
    pc0_to_pc1_distances = pc0_to_pc1_knn.dists[0]
    return pc0_to_pc1_distances


def supervised_pc_loss(warped_pc: torch.Tensor,
                   target_pc: torch.Tensor,
                   ):
    assert warped_pc.shape == target_pc.shape
    loss = torch.nn.functional.l1_loss(warped_pc, target_pc)
    return loss

