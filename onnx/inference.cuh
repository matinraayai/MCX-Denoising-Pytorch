#ifndef MCX_DENOISE_INFERENCE_CUH
#define MCX_DENOISE_INFERENCE_CUH
#include <onnxruntime_cxx_api.h>
#include <chrono>
#include <cmath>
#include <exception>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <string>
#include <vector>
enum class ExecutionProvider {TensorRT, CUDA, ROCM};
#define BLOCKDIM1D 32

template <typename scalar_t>
std::vector<scalar_t> denoise(std::vector<scalar_t> fluence_map, const std::vector<int64_t> shape,
                              const ExecutionProvider execution_provider);


#endif //MCX_DENOISE_INFERENCE_CUH
