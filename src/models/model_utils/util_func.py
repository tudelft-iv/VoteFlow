import math
import numpy as np
import torch
from torch_scatter import scatter_max
import pytorch3d.ops as pytorch3d_ops


def float_division(x, y):
    print(x/y, int(x/y))
    reminder = x - int(x/y) * y
    return reminder==0.0

def tensor_mem_size(x):
    if torch.is_tensor(x):
        size_GB = x.numel() * x.element_size() /1024/1024/1024
    elif isinstance(x, list):
        size_GB = np.prod(x) * 4 /1024/1024/1024
    else:
        NotImplementedError
    return size_GB

# # # find duplicate bins for each src point, since multiple translation vectors may fall into the same bin due to the variation of point density.
def remove_duplicates(bins, offset):
    assert bins.dim()==2
    assert bins.max()<offset
    assert bins.min()>=0
    bins += torch.arange(0, len(bins), device=bins.device)[:, None] * offset # [l, n]
    bins_uni, bins_inv = torch.unique(bins, sorted=True, return_inverse=True, return_counts=False, dim=None)
    # print('bins_uni: ', bins_uni)
    # print('counts: ', counts)
    
    bins_inv -= bins_inv.min(dim=1, keepdim=True)[0]
    # print('bins_inv: ', bins_inv)

    # _, counts_per_row = torch.unique(bins_inv, return_counts=True, return_inverse=False, sorted=False)
    # max_num_cols = counts_per_row.max().item()
    # bins_no_duplicates = torch.zeros((len(bins), max_num_cols), device=bins.device) - 1.0
    bins %= offset # [l, n]
    bins_no_duplicates = torch.zeros((len(bins), bins_inv.max()+1), device=bins.device, dtype=bins.dtype) - 1
    bins_no_duplicates, _ = scatter_max(bins, bins_inv, dim=1, out=bins_no_duplicates)
    # print('bins_no_duplicates: ', bins_no_duplicates.shape, bins_no_duplicates.dtype, bins_no_duplicates)
    return bins_no_duplicates

def remove_duplicates_with_mask(bins, offset, mask):
    assert bins.dim()==2
    assert bins.max()<offset
    assert bins.min()>=0
    assert mask.shape==bins.shape
    bins += torch.arange(0, len(bins), device=bins.device)[:, None] * offset # [l, n]
    bins_uni, bins_inv = torch.unique(bins, sorted=True, return_inverse=True, return_counts=False, dim=None)
    # print('bins_uni: ', bins_uni)
    # print('counts: ', counts)
    
    bins_inv -= bins_inv.min(dim=1, keepdim=True)[0]
    bins_inv[mask]=-1
    bins_inv += 1
    # print('bins_inv: ', bins_inv)

    # _, counts_per_row = torch.unique(bins_inv, return_counts=True, return_inverse=False, sorted=False)
    # max_num_cols = counts_per_row.max().item()
    # bins_no_duplicates = torch.zeros((len(bins), max_num_cols), device=bins.device) - 1.0
    bins %= offset # [l, n]
    bins_no_duplicates = torch.zeros((len(bins), bins_inv.max()+1), device=bins.device, dtype=bins.dtype) - 1
    bins_no_duplicates, _ = scatter_max(bins, bins_inv, dim=1, out=bins_no_duplicates)
    bins_no_duplicates = bins_no_duplicates[:, 1:]
    # print('bins_no_duplicates: ', bins_no_duplicates.shape, bins_no_duplicates.dtype, bins_no_duplicates)
    return bins_no_duplicates

def calculate_unq_voxels(coords, image_dims):
    unqs, idxs = torch.unique(coords[:, 1]*image_dims[0]+coords[:, 2], return_inverse=True, sorted=True)
    unqs_voxel = torch.stack([unqs//image_dims[0], unqs%image_dims[0]], dim=1)
    return unqs_voxel, idxs 

# https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/ops/utils.html
def batched_masked_gather(x: torch.Tensor, idxs: torch.Tensor, mask: torch.Tensor,fill_value=-1.0) -> torch.Tensor:
    assert x.dim() == 3 # [b, m, c]
    assert idxs.dim() == 3 # [b, n, k]
    assert idxs.shape == mask.shape
    b, m, c = x.shape

    idxs_masked = idxs.clone()
    idxs_masked[~mask] = 0
    l, n, k = idxs.shape
    # print('batched masked gather: ', x.shape, idxs.shape, idxs_masked.shape)
    y = pytorch3d_ops.knn_gather(x, idxs_masked) # [b, n, k, c]
    y[~mask, :] = fill_value
    # print('masked gather value selected: ', y.shape)
    return y

def pad_to_batch(idxs, l):
    if idxs is None:
        return idxs
    if idxs.dim()==2:
        assert idxs.shape[0]<=l
        if idxs.shape[0]<l:
            padding = torch.zeros((l-idxs.shape[0], idxs.shape[1]), device=idxs.device)-1
            idxs = torch.cat([idxs, padding], dim=0)
        else: 
            pass
    elif idxs.dim()==1:
        assert idxs.shape[0]<=l
        if idxs.shape[0]<l:
            padding = torch.zeros((l-idxs.shape[0]), device=idxs.device)-1
            idxs = torch.cat([idxs, padding], dim=0)
        else: 
            pass
    else: 
        NotImplementedError
    # print('pad to batch: ', idxs.shape, l)
    return idxs

def calculate_idxs():
    v = torch.Tensor([[8, 4, 2, 9, 6], [5, 0, 7, 0, 3]])
    x = torch.Tensor([[2, 2, 7, 1, 0], [4, 1, 4, 1, 1]])
    y = x + torch.arange(0, len(x))[:, None] * 10 # max possible value
    # some magic pytorch functions, no loops
    uni_values, uni_idxs = torch.unique(y, return_inverse=True, sorted=True, return_counts=False)
    print(uni_values)
    print(uni_idxs)
    uni_idxs2 = uni_idxs - uni_idxs.min(1, keepdim=True)[0]
    print(uni_idxs2)

    output, output_argmax = scatter_max(v, uni_idxs2, dim=1)
    print(output.shape, output)
    print(output_argmax.shape, output_argmax) # useful for retrieve max indices

if __name__ == "__main__":

    x = torch.Tensor([[2, 2, 7, 1, 0], [4, 1, 4, 1, 1]])
    mask = x==0
    print('input x: ', x)
    print('input mask: ', mask)
    remove_duplicates_with_mask(x, offset=10, mask=mask)