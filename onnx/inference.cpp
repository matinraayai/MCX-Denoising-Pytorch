
// https://github.com/microsoft/onnxruntime/blob/v1.8.2/csharp/test/Microsoft.ML.OnnxRuntime.EndToEndTests.Capi/CXX_Api_Sample.cpp
// https://github.com/microsoft/onnxruntime/blob/v1.8.2/include/onnxruntime/core/session/onnxruntime_cxx_api.h

#include "inference.h"

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
            OrtCUDAProviderOptions cuda_options;
            OrtTensorRTProviderOptions trt_options{0};
            trt_options.trt_max_partition_iterations = 1000;
            trt_options.trt_min_subgraph_size = 1;
            trt_options.trt_max_workspace_size = 1073741824;
            trt_options.trt_fp16_enable = 1;
            session_options.AppendExecutionProvider_TensorRT(trt_options);
            session_options.AppendExecutionProvider_CUDA(cuda_options);
            break;
        }
        default:
            throw std::invalid_argument("Invalid ONNX execution provider");
    }
}

template <typename scalar_t>
std::vector<scalar_t> infer(std::vector<scalar_t> noisy,
                             const std::vector<int64_t>& size,
                             const ExecutionProvider execution_provider = ExecutionProvider::CUDA,
                             const GraphOptimizationLevel g_optimization_level = GraphOptimizationLevel::ORT_ENABLE_EXTENDED) {
    // Session creation

    Ort::Env env(OrtLoggingLevel::ORT_LOGGING_LEVEL_FATAL, "MCX CNN Denoising");
    Ort::SessionOptions session_opts;
    session_opts.SetIntraOpNumThreads(1);
    set_execution_provider(session_opts, execution_provider);
    session_opts.SetGraphOptimizationLevel(g_optimization_level);
    Ort::Session session(env, "cascaded.onnx", session_opts);

    // Input and output names
    std::vector<const char*> input_names{"noisy"};
    std::vector<const char*> output_names{"clean"};

    Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::vector<Ort::Value> input_tensor;
    input_tensor.push_back(
            Ort::Value::CreateTensor<scalar_t>(mem_info, noisy.data(), numel(size), size.data(), size.size())
            );
    std::vector<scalar_t> clean(numel(size));
    std::vector<Ort::Value> output_tensor;
    output_tensor.push_back(
            Ort::Value::CreateTensor<scalar_t>(mem_info, clean.data(), numel(size), size.data(), size.size())
    );


    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    session.Run(Ort::RunOptions{nullptr}, input_names.data(),
                input_tensor.data(), 1, output_names.data(),
                output_tensor.data(), 1);
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "Inference time: "
    << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << " ms" << std::endl;
    return clean;
}


std::vector<float> denoise(std::vector<float> fluence_map, const std::vector<int64_t> shape)
{
    return infer(fluence_map, shape);
}