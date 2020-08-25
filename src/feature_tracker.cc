
#include <chrono>
#include <thread>

#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <Eigen/Dense>
#include <opencv2/core/eigen.hpp>

#include <opencv2/opencv.hpp>
#include "feature_tracker.h"


namespace feature_tracker
{

    FeatureTracker::FeatureTracker(const Eigen::Matrix3f &Rvc,
                                   const Eigen::Matrix3f &K,
                                   const Eigen::Vector4f &rect,
                                   const int scale,
                                   FeatureExtraction *kp_frontend)
        : rect_(rect(0)/scale,rect(1)/scale,rect(2)/scale,rect(3)/scale),
          scale_(scale),
          kp_frontend_(kp_frontend)
    {
        Eigen::Matrix3f Rcv = Rvc.inverse();
        Eigen::Matrix3f Ks = K;
        Ks /= scale;
        Ks(2,2) = 1;
        Eigen::Matrix3f Ksinv = Ks.inverse();

        Eigen::Matrix<float, 3, 4> M = (Eigen::Matrix<float, 3, 4>() << 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., -1.).finished();
        Eigen::Matrix<float, 4, 3> N = (Eigen::Matrix<float, 4, 3>() << 1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., -1.).finished();
        M1_ = Ks * Rcv * M;
        M2_ = N * Rvc * Ksinv;

        Tvlvr_ = Eigen::Matrix4f::Identity();
        Twvl_ = Eigen::Matrix4f::Identity();
        precompute();
    }
    inline Eigen::Matrix3f FeatureTracker::getH() const{
        return  M1_ * Tvlvr_ * M2_;
    }

    bool FeatureTracker::track(const cv::Mat& img)
    {
        auto t0 = std::chrono::system_clock::now();
#if 1
        cv::Mat desc = kp_frontend_->getDesc1D(img);
        //double elapsed0 = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
        //std::cout<<"get desc:"<<elapsed0<<std::endl;

        if (scale_ != 8)
            cv::resize(desc, liv_data_, cv::Size(img.cols / scale_, img.rows / scale_), cv::INTER_CUBIC);
        else
            liv_data_ = desc;
#else
        cv::Mat desc;
        img.convertTo(desc, CV_32FC1, 1.0 / 255.0);
        liv_data_ = desc;

#endif
        //auto t2 = std::chrono::system_clock::now();
        //double elapsed1 = std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count();
        //std::cout<<"resize :"<<elapsed1<<std::endl;

        if(ref_data_.empty())
        {
            liv_data_.copyTo(ref_data_);
            return false;
        }

        cv::Rect roi(rect_(0), rect_(1), rect_(2), rect_(3));
        cv::Mat ref_data_roi = ref_data_(roi);

        auto ref_dxdy = dataGradient(ref_data_roi);
        //auto t3 = std::chrono::system_clock::now();

        //double elapsed2 = std::chrono::duration_cast<std::chrono::microseconds>(t3-t2).count();
        //std::cout<<"get ref_dxdy :"<<elapsed2<<std::endl;

        Eigen::Matrix4f Tvlvr = Tvlvr_;

        const int width = ref_data_roi.cols;
        const int height = ref_data_roi.rows;
        const int elements = width * height;

        float last_err = 100000;
        int cnt = 0;
        while (true)
        {
            //auto t4 = std::chrono::system_clock::now();
            auto H = getH();
            cv::Mat liv_data_roi = getLivData(H);
            //auto t5 = std::chrono::system_clock::now();
            //double elapsed4 = std::chrono::duration_cast<std::chrono::microseconds>(t5-t4).count();
            //std::cout<<"getLivData: "<<elapsed4<<std::endl;

            //auto t6 = std::chrono::system_clock::now();
            Eigen::VectorXf res;
            const float err = residuals(liv_data_roi, ref_data_roi, res);
            //auto t7 = std::chrono::system_clock::now();
            //double elapsed6 = std::chrono::duration_cast<std::chrono::microseconds>(t7-t6).count();
            //std::cout<<"residuals: "<<elapsed6<<std::endl;

            if (last_err - err < 0.0000001)
                break;
    
            last_err = err;
            //std::cout<<"error: "<<err<<std::endl;

            //auto t8 = std::chrono::system_clock::now();
            auto liv_dxdy = dataGradient(liv_data_roi);
            //auto t9 = std::chrono::system_clock::now();
            //double elapsed8 = std::chrono::duration_cast<std::chrono::microseconds>(t9-t8).count();
            //std::cout<<"get liv_dxdy: "<<elapsed8<<std::endl;

            Eigen::MatrixXf J(elements, 3);
            for(int i = 0; i < (int)JwJg_.size(); i++)
            {
                auto& JwJg = JwJg_[i];
                auto Ji = (liv_dxdy[i] + ref_dxdy[i])/2;
                auto JiJwJg = Ji * JwJg;
                J.block(i,0,1,3) = JiJwJg;
            }
            //auto t10 = std::chrono::system_clock::now();
            //double elapsed9 = std::chrono::duration_cast<std::chrono::microseconds>(t10-t9).count();
            //std::cout<<"get JiJgJw: "<<elapsed9<<std::endl;

            auto hessian = J.transpose() * J;
            auto hessian_inv = hessian.inverse();
            auto temp = -(J.transpose() * res);
            auto x0 = hessian_inv * temp;
            auto dT = exp(x0);
            Tvlvr = Tvlvr * dT;
            cnt++; 
            //auto t11 = std::chrono::system_clock::now();
            //double elapsed10 = std::chrono::duration_cast<std::chrono::microseconds>(t11-t10).count();
            //std::cout<<"get H: "<<elapsed10<<std::endl;
            //std::cout<<"--------------------->"<<cnt<<std::endl;
            //show(img, H);
        }
        static double th = 0;
        liv_data_.copyTo(ref_data_);
        Tvlvr_ = Tvlvr;
        //std::cout<<"Tvlvr track:\n"<<Tvlvr<<std::endl;
        Twvl_ = Twvl_ * Tvlvr.inverse();
        double dth = std::atan2(Tvlvr_(1, 0), Tvlvr_(0, 0));
        double th__ = std::atan2(Twvl_(1, 0), Twvl_(0, 0));
        th += dth;
        std::cout<<th<<std::endl;
        std::cout<<th__<<std::endl;
        std::cout<<"---------"<<std::endl;
        //std::cout<<Twvl_<<std::endl;
        auto t12 = std::chrono::system_clock::now();
        double elapsed120 = std::chrono::duration_cast<std::chrono::milliseconds>(t12-t0).count();
        //std::cout<<"total loop: "<<cnt<<"  time: "<<elapsed120<<std::endl;
        return true;
    }
    
    void FeatureTracker::show(const cv::Mat &img) const 
    {
        cv::Mat img_color;
        cv::cvtColor(img, img_color, cv::COLOR_GRAY2BGR);

        {

            cv::Point a(rect_[0], rect_[1]);
            cv::Point b(rect_[0], rect_[1]+rect_[2]);
            cv::Point c(rect_[0]+rect_[3], rect_[1]);
            cv::Point d(rect_[0]+rect_[3], rect_[1]+rect_[2]);

            cv::line(img_color, a*scale_, b*scale_, cv::Scalar(0, 200, 200), 2, CV_AA);
            cv::line(img_color, b*scale_, d*scale_, cv::Scalar(0, 200, 200), 2, CV_AA);
            cv::line(img_color, d*scale_, c*scale_, cv::Scalar(0, 200, 200), 2, CV_AA);
            cv::line(img_color, c*scale_, a*scale_, cv::Scalar(0, 200, 200), 2, CV_AA);
        }

        {
            Eigen::Vector3f p0(0., 0., 1.);
            Eigen::Vector3f p1(0., rect_[2], 1.);
            Eigen::Vector3f p2(rect_[3], 0., 1.);
            Eigen::Vector3f p3(rect_[3], rect_[2], 1.);
            Eigen::Matrix3f tmp = Eigen::Matrix3f::Identity();
            tmp(0,2) = rect_(0);
            tmp(1,2) = rect_(1);

            const Eigen::Matrix3f H = tmp * getH();
            p0 = H * p0;
            p0 /= p0.z();
            p1 = H * p1;
            p1 /= p1.z();
            p2 = H * p2;
            p2 /= p2.z();
            p3 = H * p3;
            p3 /= p3.z();
            cv::Point a(p0.x(), p0.y());
            cv::Point b(p1.x(), p1.y());
            cv::Point c(p2.x(), p2.y());
            cv::Point d(p3.x(), p3.y());

            cv::line(img_color, a*scale_, b*scale_, cv::Scalar(0, 200, 0), 2, CV_AA);
            cv::line(img_color, b*scale_, d*scale_, cv::Scalar(0, 200, 0), 2, CV_AA);
            cv::line(img_color, d*scale_, c*scale_, cv::Scalar(0, 200, 0), 2, CV_AA);
            cv::line(img_color, c*scale_, a*scale_, cv::Scalar(0, 200, 0), 2, CV_AA);
        }


        cv::imshow("w", img_color);
        cv::waitKey(1);
    }

    Eigen::Matrix4f FeatureTracker::exp(const Eigen::Vector3f& x) const
    {
        Eigen::Matrix4f A = Eigen::Matrix4f::Zero();
        for(int i = 0; i < (int)A_.size(); i++)
        {
            A += x[i] * A_[i];
        }
        Eigen::Matrix4f G = Eigen::Matrix4f::Zero();
        Eigen::Matrix4f A_factor = Eigen::Matrix4f::Identity();
        float i_factor = 1.;
        for(int i = 0; i < 9; i++)
        {
            G += A_factor/i_factor;
            A_factor = A_factor * A;
            i_factor*= float(i+1);
        }
        return G;
    }

    cv::Mat FeatureTracker::getLivData(Eigen::Matrix3f &H) const
    {
        Eigen::Matrix3f tmp = Eigen::Matrix3f::Identity();
        tmp(0,2) = rect_(0);
        tmp(1,2) = rect_(1);

        Eigen::Matrix3f Heigen = tmp * H;
        cv::Mat liv_data, Hcv;
        cv::eigen2cv(Heigen, Hcv);
        cv::warpPerspective(liv_data_, liv_data, Hcv, cv::Size(rect_[2], rect_[3]), cv::INTER_LINEAR + cv::WARP_INVERSE_MAP);
        //cv::warpPerspective(liv_data_, liv_data, Hcv, cv::Size(rect_[2], rect_[3]), cv::INTER_CUBIC + cv::WARP_INVERSE_MAP);
        
        return liv_data;
    }
    Eigen::Matrix4f FeatureTracker::getTwvl() const
    {
        return Twvl_;
    }

    std::vector<Eigen::Matrix<float, 1, 2>, Eigen::aligned_allocator<Eigen::Matrix<float, 1, 2>>>
    FeatureTracker::dataGradient(const cv::Mat &data) const
    {
        /*
        cv::Mat dx = shiftFrame(data, 1, 3) - data;
        cv::Mat dy = shiftFrame(data, 1, 0) - data;

        const int width = data.cols;
        const int height = data.rows;

        std::vector<Eigen::Matrix<float, 1, 2>, Eigen::aligned_allocator<Eigen::Matrix<float, 1, 2>>> gradient;
        for (int i = 0; i < height; i++)
        {
            for (int j = 0; j < width; j++)
            {
                Eigen::Matrix<float, 1, 2> dxdy;
                dxdy(0,0) = *dx.ptr<float>(i, j);
                dxdy(0,1) = *dy.ptr<float>(i, j);
                gradient.push_back(dxdy);
            }
        }
        */
        const int width = data.cols;
        const int height = data.rows;

        std::vector<Eigen::Matrix<float, 1, 2>, Eigen::aligned_allocator<Eigen::Matrix<float, 1, 2>>> gradient;
        for (int i = 0; i < height; i++)
        {
            
            for (int j = 0; j < width; j++)
            {
                float x00, x01, x10;
                x00 = *data.ptr<float>(i, j);
                if(j+1==width)
                    x01 = 0;
                else
                    x01 = *data.ptr<float>(i, j+1);
                if(i+1==height)
                    x10 = 0;
                else
                    x10 = *data.ptr<float>(i+1, j);

                Eigen::Matrix<float, 1, 2> dxdy;
                dxdy(0,0) = x01 - x00;
                dxdy(0,1) = x10 - x00;
                gradient.push_back(dxdy);
            }
        }


        return gradient;
    }

    float FeatureTracker::residuals(const cv::Mat &data1, const cv::Mat &data2, Eigen::VectorXf &res) const
    {
        cv::Mat d = data1 - data2;

        const int width = data1.cols;
        const int height = data1.rows;

        res.resize(width * height);
        float m = 0;
        for (int i = 0; i < height; i++)
        {
            for (int j = 0; j < width; j++)
            {
                const float *pxl1 = data1.ptr<float>(i, j);
                const float *pxl2 = data2.ptr<float>(i, j);

                const float r = *(pxl1) - *(pxl2);
                m += r*r;
                res(i * width  + j) = r;
            }
        }
        float err = std::sqrt(m / width / height);
        return err;
    }

    void FeatureTracker::precompute()
    {
        const Eigen::Matrix<float, 4, 4> A1 = (Eigen::Matrix<float, 4, 4>() << 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0).finished();
        const Eigen::Matrix<float, 4, 4> A2 = (Eigen::Matrix<float, 4, 4>() << 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0).finished();
        const Eigen::Matrix<float, 4, 4> A3 = (Eigen::Matrix<float, 4, 4>() << 0, -1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0).finished();

         Eigen::Matrix<float,3, 3, Eigen::RowMajor> H1 = M1_ * A1 * M2_;
         Eigen::Matrix<float,3, 3, Eigen::RowMajor> H2 = M1_ * A2 * M2_;
         Eigen::Matrix<float,3, 3, Eigen::RowMajor> H3 = M1_ * A3 * M2_;
        
        Eigen::MatrixXf Jg(9,3);

        Eigen::Map<Eigen::MatrixXf> H191(H1.data(), 9,1);
        Eigen::Map<Eigen::MatrixXf> H291(H2.data(), 9,1);
        Eigen::Map<Eigen::MatrixXf> H391(H3.data(), 9,1);

        Jg.block(0,0,9,1) = H191;
        Jg.block(0,1,9,1) = H291;
        Jg.block(0,2,9,1) = H391;

        A_.push_back(A1);
        A_.push_back(A2);
        A_.push_back(A3);

        for (float v = 0.5; v < rect_(2); v+=1)//row
        {
            for (float u = 0.5; u < rect_(3); u+=1)//col
            {
                Eigen::Matrix<float, 2, 9> Jw =
                    (Eigen::Matrix<float, 2, 9>()
                         << u, v, 1, 0, 0, 0, -u * u, -u * v, -u, 
                            0, 0, 0, u, v, 1, -v * u, -v * v, -v)
                        .finished();
                JwJg_.push_back(Jw * Jg);
            }
        }
    }
} // namespace feature_tracker
