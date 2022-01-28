
// https://github.com/microsoft/onnxruntime/blob/v1.8.2/csharp/test/Microsoft.ML.OnnxRuntime.EndToEndTests.Capi/CXX_Api_Sample.cpp
// https://github.com/microsoft/onnxruntime/blob/v1.8.2/include/onnxruntime/core/session/onnxruntime_cxx_api.h

#include "inference.cuh"

inline int64_t numel(std::vector<int64_t> size) {
    return size[0] * size[1] * size[2];
}

void set_execution_provider(Ort::SessionOptions& session_options, const ExecutionProvider execution_provider) {
    switch (execution_provider) {
        case ExecutionProvider::CUDA: {
            OrtCUDAProviderOptions cuda_options;
            session_options.AppendExecutionProvider_CUDA(cuda_options);
            break;
        }
        case ExecutionProvider::ROCM: {
            OrtROCMProviderOptions rocm_options;
            session_options.AppendExecutionProvider_ROCM(rocm_options);
            break;
        }
        case ExecutionProvider::TensorRT: {
            OrtTensorRTProviderOptions trt_options;
            session_options.AppendExecutionProvider_TensorRT(trt_options);
            break;
        }
        default:
            throw std::invalid_argument("Invalid ONNX execution provider");
    }
}

template <typename scalar_t>
std::vector<scalar_t*> infer(scalar_t* low, scalar_t* high,
                             const std::vector<int64_t>& size,
                             const ExecutionProvider execution_provider = ExecutionProvider::TensorRT,
                             const GraphOptimizationLevel g_optimization_level = GraphOptimizationLevel::ORT_ENABLE_EXTENDED) {
    // Session creation
    Ort::Env env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, "MCX CNN Denoising");
    Ort::SessionOptions session_opts;
    session_opts.SetIntraOpNumThreads(1);
    set_execution_provider(session_opts, execution_provider);
    session_opts.SetGraphOptimizationLevel(g_optimization_level);
    Ort::Session session(env, "cascaded.onnx", session_opts);

    // Input and output names
    std::vector<const char*> input_names{"noisy"};
    std::vector<const char*> output_names{"clean"};
    // Input and output Tensor abstraction
    Ort::MemoryInfo mem_info("mem info", OrtAllocatorType::OrtArenaAllocator, 0, OrtMemTypeDefault);
    std::vector<Ort::Value> high_tensor{
        Ort::Value::CreateTensor<scalar_t>(mem_info, high, numel(size), size.data(), size.size())
    };
    std::vector<Ort::Value> low_tensor{
        Ort::Value::CreateTensor<scalar_t>(mem_info, low, numel(size), size.data(), size.size())
    };

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    session.Run(Ort::RunOptions{nullptr}, input_names.data(),
                low_tensor.data(), 1, output_names.data(),
                low_tensor.data(), 1);
    session.Run(Ort::RunOptions{nullptr}, input_names.data(),
                high_tensor.data(), 1, output_names.data(),
                high_tensor.data(), 1);
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "Inference time: "
    << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << " ms" << std::endl;
    return {low, high};
}

template<typename scalar_t>
__global__
void k_norm(scalar_t* t, int size, float c) {
    int32_t i = gridDim.x * blockIdx.x + threadIdx.x;
    if (i < size)
        t[i] = log1p(c * t[i]);
}

template<typename scalar_t>
__global__
void k_rev_norm(scalar_t* t, int size, float c) {
    int32_t i = gridDim.x * blockIdx.x + threadIdx.x;
    if (i < size)
        t[i] = (exp(t[i]) - 1) / c;
}


template<typename scalar_t>
__global__
void k_threshold(scalar_t* high, scalar_t* low, scalar_t* original, float threshold, int size) {
    int32_t i = gridDim.x * blockIdx.x + threadIdx.x;
    if (i < size)
        original[i] = original[i] > threshold ? high[i] : low[i];
}


template <typename scalar_t>
std::vector<scalar_t> denoise(std::vector<scalar_t> fluence_map, const std::vector<int64_t> shape,
                              const ExecutionProvider execution_provider)
{
    scalar_t* d_low;
    scalar_t* d_high;
    scalar_t* d_fluence;
    int64_t num_elements = numel(shape);
    size_t volume_size = sizeof(scalar_t) * num_elements;
    cudaMalloc(&d_low, volume_size);
    cudaMalloc(&d_high, volume_size);
    cudaMalloc(&d_fluence, volume_size);
    cudaMemcpy(d_fluence, fluence_map.data(), volume_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_low, d_fluence, volume_size, cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_high, d_fluence, volume_size, cudaMemcpyDeviceToDevice);
    dim3 grid_dim(num_elements / BLOCKDIM1D + (num_elements % BLOCKDIM1D) ? 1 : 0);
    k_norm<<<BLOCKDIM1D, grid_dim>>>(d_low, num_elements, 1);
    k_norm<<<BLOCKDIM1D, grid_dim>>>(d_high, num_elements, 10000000);
    infer(d_low, d_high, shape);
    k_threshold<<<BLOCKDIM1D, grid_dim>>>(d_high, d_low, d_fluence, 0.03, num_elements);

    std::vector<scalar_t> output(numel(shape));
    cudaMemcpy(output.data(), d_fluence, sizeof(scalar_t) * numel(shape), cudaMemcpyDeviceToHost);
    cudaFree(d_low);
    cudaFree(d_high);
    cudaFree(d_fluence);

}