# Raw GPU Timeline Head

按 Chrome trace `ts` 从早到晚排序，单位为 `us`。这里只展示前 80 行，完整数据见 `raw_gpu_timeline_us.tsv`。

| index | stage | relative_ts_us | dur_us | cat | name |
|---:|---|---:|---:|---|---|
| 0 | prefill | 0.000 | 16.591 | `privateuse1_runtime` | `musaDeviceSynchronize` |
| 1 | prefill | 374.617 | 242.729 | `privateuse1_runtime` | `musaMemcpyAsync` |
| 2 | prefill | 579.692 | 3.120 | `gpu_memcpy` | `Memcpy HtoD (Pageable -> Device)` |
| 3 | prefill | 678.780 | 125.326 | `privateuse1_runtime` | `musaMemcpyAsync` |
| 4 | prefill | 781.378 | 2.600 | `gpu_memcpy` | `Memcpy HtoD (Pageable -> Device)` |
| 5 | prefill | 857.744 | 114.428 | `privateuse1_runtime` | `musaMemcpyAsync` |
| 6 | prefill | 950.784 | 2.640 | `gpu_memcpy` | `Memcpy HtoD (Pageable -> Device)` |
| 7 | prefill | 1051.007 | 119.005 | `privateuse1_runtime` | `musaMemcpyAsync` |
| 8 | prefill | 1148.511 | 2.640 | `gpu_memcpy` | `Memcpy HtoD (Pageable -> Device)` |
| 9 | prefill | 1198.868 | 117.672 | `privateuse1_runtime` | `musaMemcpyAsync` |
| 10 | prefill | 1295.276 | 2.520 | `gpu_memcpy` | `Memcpy HtoD (Pageable -> Device)` |
| 11 | prefill | 1388.432 | 115.461 | `privateuse1_runtime` | `musaMemcpyAsync` |
| 12 | prefill | 1482.882 | 2.640 | `gpu_memcpy` | `Memcpy HtoD (Pageable -> Device)` |
| 13 | prefill | 1613.196 | 115.444 | `privateuse1_runtime` | `musaMemcpyAsync` |
| 14 | prefill | 1707.409 | 2.600 | `gpu_memcpy` | `Memcpy HtoD (Pageable -> Device)` |
| 15 | prefill | 1730.401 | 2.665 | `privateuse1_runtime` | `musaStreamSynchronize` |
| 16 | prefill | 1957.600 | 10.944 | `privateuse1_driver` | `muLaunchKernel` |
| 17 | prefill | 2013.579 | 6.801 | `kernel` | `write_req_to_token_pool_triton` |
| 18 | prefill | 2049.538 | 124.544 | `privateuse1_runtime` | `musaMemcpyAsync` |
| 19 | prefill | 2153.024 | 2.520 | `gpu_memcpy` | `Memcpy HtoD (Pageable -> Device)` |
| 20 | prefill | 2175.371 | 2.280 | `privateuse1_runtime` | `musaStreamSynchronize` |
| 21 | prefill | 2229.932 | 119.342 | `privateuse1_runtime` | `musaMemcpyAsync` |
| 22 | prefill | 2327.950 | 2.520 | `gpu_memcpy` | `Memcpy HtoD (Pageable -> Device)` |
| 23 | prefill | 2350.261 | 1.009 | `privateuse1_runtime` | `musaStreamSynchronize` |
| 24 | prefill | 2387.700 | 120.352 | `privateuse1_runtime` | `musaMemcpyAsync` |
| 25 | prefill | 2486.995 | 2.560 | `gpu_memcpy` | `Memcpy HtoD (Pageable -> Device)` |
| 26 | prefill | 2508.925 | 1.248 | `privateuse1_runtime` | `musaStreamSynchronize` |
| 27 | prefill | 2545.485 | 118.677 | `privateuse1_runtime` | `musaMemcpyAsync` |
| 28 | prefill | 2643.280 | 2.521 | `gpu_memcpy` | `Memcpy HtoD (Pageable -> Device)` |
| 29 | prefill | 2665.040 | 1.090 | `privateuse1_runtime` | `musaStreamSynchronize` |
| 30 | prefill | 2870.125 | 126.057 | `privateuse1_runtime` | `musaMemcpyAsync` |
| 31 | prefill | 2970.811 | 2.480 | `gpu_memcpy` | `Memcpy HtoD (Pageable -> Device)` |
| 32 | prefill | 3038.452 | 120.388 | `privateuse1_runtime` | `musaMemcpyAsync` |
| 33 | prefill | 3134.497 | 2.520 | `gpu_memcpy` | `Memcpy HtoD (Pageable -> Device)` |
| 34 | prefill | 3319.598 | 7.011 | `privateuse1_driver` | `muLaunchKernel` |
| 35 | prefill | 3367.545 | 4.320 | `kernel` | `compute_position_kernel` |
| 36 | prefill | 3429.900 | 8.975 | `privateuse1_runtime` | `musaLaunchKernel` |
| 37 | prefill | 3479.828 | 6.041 | `kernel` | `void musa::dnn::ScanPartSumShflKernel<int, long, long, musa::dnn::ScanOp<musa::dnn::DeviceTensor<unsigned int, musa::dnn` |
| 38 | prefill | 3483.674 | 7.740 | `privateuse1_runtime` | `musaLaunchKernel` |
| 39 | prefill | 3510.509 | 5.161 | `kernel` | `void musa::dnn::unary::kernel_unary<long, musa::dnn::unary::CastOp, long, int, long, 128>(int*, long const*, unsigned lo` |
| 40 | prefill | 3624.094 | 7.376 | `privateuse1_driver` | `muLaunchKernel` |
| 41 | prefill | 3660.031 | 5.694 | `privateuse1_runtime` | `musaLaunchKernel` |
| 42 | prefill | 3671.475 | 2.640 | `kernel` | `create_flashinfer_kv_indices_triton` |
| 43 | prefill | 3683.075 | 6.080 | `kernel` | `void musa::dnn::ScanPartSumShflKernel<int, long, long, musa::dnn::ScanOp<musa::dnn::DeviceTensor<unsigned int, musa::dnn` |
| 44 | prefill | 3694.099 | 7.882 | `privateuse1_runtime` | `musaLaunchKernel` |
| 45 | prefill | 3721.276 | 4.041 | `kernel` | `void musa::dnn::unary::kernel_unary<long, musa::dnn::unary::CastOp, long, int, long, 128>(int*, long const*, unsigned lo` |
| 46 | prefill | 3929.913 | 8.273 | `privateuse1_runtime` | `musaLaunchKernel` |
| 47 | prefill | 3978.325 | 3.880 | `kernel` | `void at::native::(anonymous namespace)::IndexSelectVectorKernel<__half, long, true, 128>(__half*, long*, __half*, long, ` |
| 48 | prefill | 4064.701 | 6.498 | `privateuse1_runtime` | `musaLaunchKernel` |
| 49 | prefill | 4113.650 | 10.880 | `kernel` | `void musa::dnn::LayerNormGlobalKernelVlen<__half, float, __half, float, 128, 4, 8, (musa::dnn::NormType)1, false>(__half` |
| 50 | prefill | 4227.531 | 9.644 | `privateuse1_runtime` | `musaLaunchKernelExC` |
| 51 | prefill | 4281.215 | 42.482 | `kernel` | `musa_asm_hhhhssgemm_nt_tce_128_32x128B128_epilogue` |
| 52 | prefill | 4348.421 | 6.734 | `privateuse1_runtime` | `musaLaunchKernel` |
| 53 | prefill | 4406.059 | 6.601 | `kernel` | `void musa::dnn::(anonymous namespace)::CopyLastContiguousKernel<8, short, true, musa::dnn::DeviceTensor<unsigned int, mu` |
| 54 | prefill | 4430.580 | 6.161 | `privateuse1_runtime` | `musaLaunchKernel` |
| 55 | prefill | 4476.976 | 4.978 | `privateuse1_runtime` | `musaLaunchKernel` |
| 56 | prefill | 4487.982 | 7.440 | `kernel` | `void musa::dnn::LayerNormGlobalKernelVlen<__half, float, __half, float, 64, 8, 8, (musa::dnn::NormType)1, false>(__half*` |
| 57 | prefill | 4497.622 | 5.281 | `kernel` | `void musa::dnn::(anonymous namespace)::CopyLastContiguousKernel<8, short, true, musa::dnn::DeviceTensor<unsigned int, mu` |
| 58 | prefill | 4542.432 | 5.511 | `privateuse1_runtime` | `musaLaunchKernel` |
| 59 | prefill | 4588.025 | 7.601 | `kernel` | `void musa::dnn::LayerNormGlobalKernelVlen<__half, float, __half, float, 64, 1, 8, (musa::dnn::NormType)1, false>(__half*` |
| 60 | prefill | 4948.611 | 7.468 | `privateuse1_driver` | `muLaunchKernel` |
| 61 | prefill | 4994.399 | 12.960 | `kernel` | `triton_poi_fused_add_cat_index_select_mul_split_sub_unsqueeze_view_0` |
| 62 | prefill | 5001.195 | 4.733 | `privateuse1_driver` | `muLaunchKernel` |
| 63 | prefill | 5023.480 | 10.800 | `kernel` | `triton_poi_fused_add_cat_index_select_mul_split_sub_unsqueeze_view_1` |
| 64 | prefill | 5250.426 | 11.076 | `privateuse1_runtime` | `musaLaunchKernel` |
| 65 | prefill | 5283.200 | 5.045 | `privateuse1_runtime` | `musaLaunchKernel` |
| 66 | prefill | 5304.169 | 7.761 | `kernel` | `void at::native::(anonymous namespace)::IndexVectorizedKernel<at::native::(anonymous namespace)::GPUIndexKernel<c10::Hal` |
| 67 | prefill | 5314.130 | 6.520 | `kernel` | `void at::native::(anonymous namespace)::IndexVectorizedKernel<at::native::(anonymous namespace)::GPUIndexKernel<c10::Hal` |
| 68 | prefill | 5361.213 | 5.468 | `privateuse1_runtime` | `musaLaunchKernel` |
| 69 | prefill | 5410.293 | 4.360 | `kernel` | `void musa::dnn::(anonymous namespace)::CopyLastContiguousKernel<8, short, true, musa::dnn::DeviceTensor<unsigned int, mu` |
| 70 | prefill | 5624.009 | 6.648 | `privateuse1_driver` | `muLaunchKernel` |
| 71 | prefill | 5669.221 | 21.081 | `kernel` | `_fwd_kernel` |
| 72 | prefill | 5771.158 | 9.280 | `privateuse1_runtime` | `musaLaunchKernelExC` |
| 73 | prefill | 5818.906 | 35.322 | `kernel` | `musa_asm_hhhhssgemm_nt_tce_128_32x128B128_epilogue` |
| 74 | prefill | 5859.676 | 5.838 | `privateuse1_runtime` | `musaLaunchKernel` |
| 75 | prefill | 5888.069 | 10.280 | `kernel` | `void LayerNormGlobalKernelVlen<__half, float, 1024, 1, 8>(__half*, __half*, __half const*, unsigned long, unsigned long,` |
| 76 | prefill | 5958.422 | 6.890 | `privateuse1_runtime` | `musaLaunchKernelExC` |
| 77 | prefill | 6005.993 | 141.644 | `kernel` | `musa_asm_hhhhssgemm_nt_tce_128_32x128B128_epilogue` |
| 78 | prefill | 6054.591 | 5.794 | `privateuse1_runtime` | `musaLaunchKernel` |
| 79 | prefill | 6143.220 | 6.999 | `privateuse1_runtime` | `musaLaunchKernelExC` |
