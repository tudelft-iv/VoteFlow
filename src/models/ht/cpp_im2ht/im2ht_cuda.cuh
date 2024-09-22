#include <iostream>
#include <cstdio>
#include <algorithm>
#include <cstring>
#include <vector>
#include <cmath>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
// #include <THC/THC.h>
#include <THC/THCAtomics.cuh>
// #include <THC/THCDeviceUtils.cuh>

#include "cuda_utils.h"

template <typename scalar_t>
__global__ void im2ht_cuda_forward_kernel(const int64_t n_threads,
                                  const scalar_t *feats_src, 
                                  const scalar_t *feats_dst, 
                                  scalar_t *vol_ht,
                                  const scalar_t *voxels_src,
                                  const scalar_t *voxels_dst,
                                  const scalar_t *idxs_src,
                                  const scalar_t *idxs_dst,
                                  const int b, const int l, const int c, 
                                  const int m, const int n,
                                  const int h, const int w, const int d
                                )
{
  CUDA_KERNEL_LOOP(index, n_threads)
  {
    // [l, c, m, n]
    int l_tmp = index / c / m / n % l;  // l 
    int c_tmp = index / m / n % c;  // c
    int m_tmp = index / n % m; // m
    int n_tmp = index % n; // n

    for (int b_tmp = 0; b_tmp < b; ++b_tmp)
    {
      int src_tmp = idxs_src[b_tmp * (l * m) + l_tmp * m + m_tmp];  // point idx? in [b, l, m]
      int dst_tmp = idxs_dst[b_tmp * (l * n) + l_tmp * n + n_tmp];  // point idx? in [b, l, n]

      if (src_tmp>=0 && dst_tmp>=0)
      {
        // Z-Y-X after voxelization
        int y_src_tmp = voxels_src[b_tmp * (l * 2) + src_tmp * 2 + 0];  // in [b, l, 2]
        int x_src_tmp = voxels_src[b_tmp * (l * 2) + src_tmp * 2 + 1];  // in [b, l, 2]

        int y_dst_tmp = voxels_dst[b_tmp * (l * 2) + dst_tmp * 2 + 0];  // in [b, l, 2]
        int x_dst_tmp = voxels_dst[b_tmp * (l * 2) + dst_tmp * 2 + 1];  // in [b, l, 2]

        scalar_t f_src_tmp = feats_src[b_tmp * (l * c) + src_tmp * c + c_tmp];  // in [b, l, c]
        scalar_t f_dst_tmp = feats_dst[b_tmp * (l * c) + dst_tmp * c + c_tmp];  // in [b, l, c]

        int bin_x = x_dst_tmp - x_src_tmp;
        int bin_y = y_dst_tmp - y_src_tmp;
        bin_x += w/2;
        bin_y += h/2;

        if (bin_x >=0 && bin_x < w && bin_y >= 0 && bin_y < h)
        {
          // (0, 0) is at position (h//2, w//2)
          // vol_ht: [b, l, 2*c, h, w]; [b, l, c, h, w] (src feats) + [b, l, c, h, w] (dst feats)
          int64_t offset_src_tmp =  b_tmp * (l * 2 *c * h * w) + l_tmp * (2 * c * h * w) + c_tmp * (h*w) + bin_y * w + bin_x;
          int64_t offset_dst_tmp =  b_tmp * (l * 2 *c * h * w) + l_tmp * (2 * c * h * w) + (c_tmp + c) * (h*w) + bin_y * w + bin_x;
          atomicAdd(vol_ht+offset_src_tmp, f_src_tmp);
          atomicAdd(vol_ht+offset_dst_tmp, f_dst_tmp);
        }
      }
    }
  }
}

template <typename scalar_t>
__global__ void im2ht_cuda_backward_kernel(const int n_threads,
                                    scalar_t* grad_feats_src,
                                    scalar_t* grad_feats_dst,
                                    const scalar_t* grad_vol_ht, 
                                    const scalar_t *voxels_src,
                                    const scalar_t *voxels_dst,
                                    const scalar_t *idxs_src,
                                    const scalar_t *idxs_dst,
                                    const int b, const int l, const int c, 
                                    const int m, const int n,
                                    const int h, const int w, const int d
                                  )
{
  // todo: coalesce
  CUDA_KERNEL_LOOP(index, n_threads)
  {
    // [l, c, m, n]
    int l_tmp = index / c / m / n % l; // l
    int c_tmp = index / m / n % c;  // c
    int m_tmp = index / n % m; // m
    int n_tmp = index % n; // n

    for (int b_tmp = 0; b_tmp < b; ++b_tmp)
    {
      int src_tmp = idxs_src[b_tmp * (l * m) + l_tmp * m + m_tmp];  // point idx? in [b, l, m]
      int dst_tmp = idxs_dst[b_tmp * (l * n) + l_tmp * n + n_tmp];  // point idx? in [b, l, n]
      if (src_tmp>=0 && dst_tmp>=0)
      {
        // Z-Y-X after voxelization
        int y_src_tmp = voxels_src[b_tmp * (l * 2) + src_tmp * 2 + 0];  // in [b, l, 2]
        int x_src_tmp = voxels_src[b_tmp * (l * 2) + src_tmp * 2 + 1];  // in [b, l, 2]

        int y_dst_tmp = voxels_dst[b_tmp * (l * 2) + dst_tmp * 2 + 0];  // in [b, l, 2]
        int x_dst_tmp = voxels_dst[b_tmp * (l * 2) + dst_tmp * 2 + 1];  // in [b, l, 2]

        int bin_x = x_dst_tmp - x_src_tmp;
        int bin_y = y_dst_tmp - y_src_tmp;
        bin_x += w/2;
        bin_y += h/2;

        if (bin_x >=0 && bin_x < w && bin_y >= 0 && bin_y < h)
        {
          // (0, 0) is at position (h//2, w//2)
          //  vol_ht: [b, l, 2*c, h, w]; [b, l, c, h, w] (src feats) + [b, l, c, h, w] (dst feats)
          // output gradients
          int offset_vol_src_tmp =  b_tmp * (l * 2 *c * h * w) + l_tmp * (2 * c * h * w) + c_tmp * (h*w) + bin_y * w + bin_x;
          int offset_vol_dst_tmp =  b_tmp * (l * 2 *c * h * w) + l_tmp * (2 * c * h * w) + (c_tmp + c) * (h*w) + bin_y * w + bin_x;
          scalar_t grad_vol_src_tmp = grad_vol_ht[offset_vol_src_tmp];  // 
          scalar_t grad_vol_dst_tmp = grad_vol_ht[offset_vol_dst_tmp];  //
          // input grads
          int offset_feat_src_tmp = b_tmp * (l * c) + src_tmp * c + c_tmp;  // in [b, l, c]
          int offset_feat_dst_tmp = b_tmp * (l * c) + dst_tmp * c + c_tmp;  // in [b, l, c]
          atomicAdd(grad_feats_src+offset_feat_src_tmp, grad_vol_src_tmp);
          atomicAdd(grad_feats_dst+offset_feat_dst_tmp, grad_vol_dst_tmp);
        }
      }
    }
  }
}

// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% //
template <typename scalar_t>
void im2ht_cuda_forward(cudaStream_t stream,
                  const scalar_t* feats_src, 
                  const scalar_t* feats_dst, 
                  scalar_t* vol_ht,
                  const scalar_t* voxels_src,
                  const scalar_t* voxels_dst,
                  const scalar_t* idxs_src,
                  const scalar_t* idxs_dst,
                  const int b, const int l, const int c, 
                  const int m, const int n, 
                  const int h, const int w, const int d
                ) 
{
  const int num_kernels = l * c * m * n;
  // printf("dimensions num_kernels=%16d, b=%06d, l=%06d, m=%06d, n=%06d, h=%06d, w= %06d, d= %06d \n",
  //                         num_kernels, b, l, m, n, h, w, d);
  im2ht_cuda_forward_kernel<scalar_t>
  <<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS, 0, stream>>>(num_kernels, 
                                                          feats_src, 
                                                          feats_dst, 
                                                          vol_ht,
                                                          voxels_src,
                                                          voxels_dst,
                                                          idxs_src,
                                                          idxs_dst,
                                                          b, l, c,
                                                          m, n, 
                                                          h, w, d
                                                        );
                                                      
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    printf("error in im2ht_gpu_kernel: %s\n", cudaGetErrorString(err));
  }
}

template <typename scalar_t>
void im2ht_cuda_backward(cudaStream_t stream,
                  scalar_t* grad_feats_src,
                  scalar_t* grad_feats_dst,
                  const scalar_t* grad_vol, 
                  const scalar_t* voxels_src,
                  const scalar_t* voxels_dst,
                  const scalar_t* idxs_src,
                  const scalar_t* idxs_dst,
                  const int b, const int l, const int c,
                  const int m, const int n, 
                  const int h, const int w, const int d
                  )
{

  const int num_kernels = l * c * m * n;
  // ***********************************************************************//
  im2ht_cuda_backward_kernel<scalar_t>
  <<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS,0, stream>>>(num_kernels, 
                                                          grad_feats_src, 
                                                          grad_feats_dst, 
                                                          grad_vol, 
                                                          voxels_src,
                                                          voxels_dst,
                                                          idxs_src,
                                                          idxs_dst,
                                                          b, l, c,
                                                          m, n, 
                                                          h, w, d
                                                          );
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    printf("error in ht2im_gpu_kernel: %s\n", cudaGetErrorString(err));
  }

}

