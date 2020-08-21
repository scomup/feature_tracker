#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Geometry>
#include <math.h>

#include "superpoint_frontend.h"
#include "cuda_runtime.h"

#define CV_32FC256 CV_MAKETYPE(CV_32F,256)

//#include "samples/common/buffers.h"

namespace feature_tracker
{

    SuperpointFrontend::SuperpointFrontend(std::string module_path,
                                           float image_h,
                                           float image_w)
        : 
          image_h_(image_h),
          image_w_(image_w)
    {
        std::stringstream gieModelStream;
        gieModelStream.seekg(0, gieModelStream.beg);
        std::ifstream cache(module_path);
        gieModelStream << cache.rdbuf();
        cache.close();
        Logger logger;
        nvinfer1::IRuntime *runtime = nvinfer1::createInferRuntime(logger);
        assert(runtime != nullptr);
        gieModelStream.seekg(0, std::ios::end);
        const int modelSize = gieModelStream.tellg();
        gieModelStream.seekg(0, std::ios::beg);
        void *modelMem = malloc(modelSize);
        gieModelStream.read((char *)modelMem, modelSize);
        engine_ = runtime->deserializeCudaEngine(modelMem, modelSize, NULL);
        context_ = engine_->createExecutionContext();
        cudaMalloc(&buffer_[0], image_h_ * image_w_ * sizeof(float));
        cudaMalloc(&buffer_[1], image_h_ * image_w_ / 64 * 65 * sizeof(float));
        cudaMalloc(&buffer_[2], image_h_ * image_w_ / 64 * 256 * sizeof(float));
    }


cv::Mat SuperpointFrontend::getDesc(const cv::Mat &img)
{
    cv::Mat image;
    img.convertTo(image, CV_32FC1); // or CV_32F works (too)
    image /= 255.;
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cudaMemcpy(buffer_[0], image.data, image_h_ * image_w_ * sizeof(float), cudaMemcpyHostToDevice);

    context_->enqueue(1, buffer_, stream, nullptr);

    float* desc_buffer = new float[image_h_ * image_w_ / 64 * 256];

    cudaMemcpy(desc_buffer, buffer_[2], image_h_ * image_w_ / 64 * 256 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    torch::Tensor coarse_desc = torch::from_blob(desc_buffer, at::IntList({1, 256, image_h_ / 8, image_w_ / 8}), at::ScalarType::Float);
    auto dn = coarse_desc.norm(2, 1);
    coarse_desc = coarse_desc.div(at::unsqueeze(dn, 1));
    //coarse_desc = coarse_desc.permute({0,2,3,1});

    cv::Mat desc(image_h_ / 8, image_w_ / 8, CV_32FC256);

    //std::memcpy(desc.data, coarse_desc.data_ptr(), sizeof(float)*coarse_desc.numel());

    auto _desc_a = coarse_desc.accessor<float, 4>();

    int width = desc.cols;
    int height = desc.rows;
    int channels = desc.channels();

    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            float* pxl = desc.ptr<float>(i, j); 

            for (int c = 0; c < channels; c++)
            {
                *(pxl+c) = _desc_a[0][c][i][j]; 
                //std::cout<<desc_local.ptr<float>(i)[256*j+c]<<std::endl;// = _desc_a[0][c][i][j];
                //std::cout<<_desc_a[0][i][j][c]<<std::endl;
            }
        }
    }
    
    return desc;
}

cv::Mat SuperpointFrontend::getDesc1D(const cv::Mat &img)
{
    cv::Mat image;
    img.convertTo(image, CV_32FC1); // or CV_32F works (too)
    image /= 255.;
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cudaMemcpy(buffer_[0], image.data, image_h_ * image_w_ * sizeof(float), cudaMemcpyHostToDevice);

    context_->enqueue(1, buffer_, stream, nullptr);

    float* desc_buffer = new float[image_h_ * image_w_ / 64 * 256];

    cudaMemcpy(desc_buffer, buffer_[2], image_h_ * image_w_ / 64 * 256 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    torch::Tensor coarse_desc = torch::from_blob(desc_buffer, at::IntList({1, 256, image_h_ / 8, image_w_ / 8}), at::ScalarType::Float);
    auto dn = coarse_desc.norm(2, 1);
    coarse_desc = coarse_desc.div(at::unsqueeze(dn, 1));
    coarse_desc = at::sum(coarse_desc, 1);

    cv::Mat desc(image_h_ / 8, image_w_ / 8, CV_32FC1);

    auto _desc_a = coarse_desc.accessor<float, 3>();

    int width = desc.cols;
    int height = desc.rows;
    int channels = desc.channels();

    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            float* pxl = desc.ptr<float>(i, j); 
            *(pxl) = _desc_a[0][i][j]; 
        }
    }
    
    return desc;
}


} // namespace visual_odometry
