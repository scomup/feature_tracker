
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
                                   SuperpointFrontend *kp_frontend)
        : rect_(rect),
          kp_frontend_(kp_frontend)
    {
        Eigen::Matrix3f Rcv = Rvc.inverse();
        Eigen::Matrix3f Kinv = K.inverse();

        Eigen::Matrix<float, 3, 4> M = (Eigen::Matrix<float, 3, 4>() << 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., -1.).finished();
        Eigen::Matrix<float, 4, 3> N = (Eigen::Matrix<float, 4, 3>() << 1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., -1.).finished();
        M1_ = K * Rcv * M;
        M2_ = N * Rvc * Kinv;

        Hroi_ = Eigen::Matrix3f::Identity();
        Hroi_(0,2) = rect_(0);
        Hroi_(1,2) = rect_(1);
        precompute();
    }

    void FeatureTracker::track(const cv::Mat& img)
    {
        auto t0 = std::chrono::system_clock::now();
        cv::Mat desc = kp_frontend_->getDesc(img);
        auto t1 = std::chrono::system_clock::now();
        double elapsed0 = std::chrono::duration_cast<std::chrono::milliseconds>(t1-t0).count();
        std::cout<<"get desc:"<<elapsed0<<std::endl;
        cv::resize(desc, liv_data_, cv::Size(640,360),cv::INTER_CUBIC);
        auto t2 = std::chrono::system_clock::now();
        double elapsed1 = std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count();
        std::cout<<"resize :"<<elapsed1<<std::endl;

        if(ref_data_.empty())
        {
            liv_data_.copyTo(ref_data_);
            return;
        }

        cv::Rect roi(rect_(0), rect_(1), rect_(2), rect_(3));
        cv::Mat ref_data_roi = ref_data_(roi);

        auto ref_dxdy = dataGradient(ref_data_roi);
        auto t3 = std::chrono::system_clock::now();

        double elapsed2 = std::chrono::duration_cast<std::chrono::milliseconds>(t3-t2).count();
        std::cout<<"get ref_dxdy :"<<elapsed2<<std::endl;

        Eigen::Matrix4f Tvlvr = Eigen::Matrix4f::Identity();
        Eigen::Matrix3f H = Eigen::Matrix3f::Identity();
        Eigen::Vector3f x0 = Eigen::Vector3f::Zero();

        const int width = ref_data_roi.cols;
        const int height = ref_data_roi.rows;
        const int channels = ref_data_roi.channels();
        const int elements = width * height * channels;

        float last_err = 100000;

        while (true)
        {
            auto t4 = std::chrono::system_clock::now();
            cv::Mat liv_data_roi = getLivData(H);
            auto t5 = std::chrono::system_clock::now();
            double elapsed4 = std::chrono::duration_cast<std::chrono::milliseconds>(t5-t4).count();
            std::cout<<"getLivData :"<<elapsed4<<std::endl;
            auto t6 = std::chrono::system_clock::now();
            Eigen::VectorXf res;
            const float err = residuals(liv_data_roi, ref_data_roi, res);
            auto t7 = std::chrono::system_clock::now();
            double elapsed6 = std::chrono::duration_cast<std::chrono::milliseconds>(t7-t6).count();
            std::cout<<"residuals :"<<elapsed6<<std::endl;

            if (last_err - err < 0.0000001)
                break;
            last_err = err;
            std::cout<<"error: "<<err<<std::endl;
            auto t8 = std::chrono::system_clock::now();
            auto liv_dxdy = dataGradient(liv_data_roi);
            auto t9 = std::chrono::system_clock::now();
            double elapsed8 = std::chrono::duration_cast<std::chrono::milliseconds>(t9-t8).count();
            std::cout<<"get liv_dxdy :"<<elapsed8<<std::endl;

            Eigen::MatrixXf J(elements, 3);
            for(int i = 0; i < JwJg_.size(); i++)
            {
                auto& JwJg = JwJg_[i];
                auto Ji = (liv_dxdy[i] + ref_dxdy[i])/2;
                auto JiJgJw = Ji * JwJg;
                J.block(256*i,0,256,3) = JiJgJw;
            }
            auto t10 = std::chrono::system_clock::now();
            double elapsed9 = std::chrono::duration_cast<std::chrono::milliseconds>(t10-t9).count();
            std::cout<<"get JiJgJw :"<<elapsed9<<std::endl;

            auto hessian = J.transpose() * J;
            auto hessian_inv = hessian.inverse();
            auto temp = -(J.transpose() * res);
            auto x0 = hessian_inv * temp;
            auto dT = exp(x0);
            Tvlvr = Tvlvr * dT;
            H = M1_ * Tvlvr * M2_;
            H_ = H;
            auto t11 = std::chrono::system_clock::now();
            double elapsed10 = std::chrono::duration_cast<std::chrono::milliseconds>(t11-t10).count();
            std::cout<<"get H :"<<elapsed10<<std::endl;
            exit(0);

            //std::cout<<H<<std::endl;
            //show(img, H);
        }
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

            cv::line(img_color, a, b, cv::Scalar(0, 200, 200), 2, CV_AA);
            cv::line(img_color, b, d, cv::Scalar(0, 200, 200), 2, CV_AA);
            cv::line(img_color, d, c, cv::Scalar(0, 200, 200), 2, CV_AA);
            cv::line(img_color, c, a, cv::Scalar(0, 200, 200), 2, CV_AA);
        }

        {
            Eigen::Vector3f p0(0., 0., 1.);
            Eigen::Vector3f p1(0., rect_[2], 1.);
            Eigen::Vector3f p2(rect_[3], 0., 1.);
            Eigen::Vector3f p3(rect_[3], rect_[2], 1.);

            const Eigen::Matrix3f tempH = Hroi_ * H_;
            p0 = tempH * p0;
            p0 /= p0.z();
            p1 = tempH * p1;
            p1 /= p1.z();
            p2 = tempH * p2;
            p2 /= p2.z();
            p3 = tempH * p3;
            p3 /= p3.z();
            cv::Point a(p0.x(), p0.y());
            cv::Point b(p1.x(), p1.y());
            cv::Point c(p2.x(), p2.y());
            cv::Point d(p3.x(), p3.y());

            cv::line(img_color, a, b, cv::Scalar(0, 200, 0), 2, CV_AA);
            cv::line(img_color, b, d, cv::Scalar(0, 200, 0), 2, CV_AA);
            cv::line(img_color, d, c, cv::Scalar(0, 200, 0), 2, CV_AA);
            cv::line(img_color, c, a, cv::Scalar(0, 200, 0), 2, CV_AA);
        }


        cv::imshow("w", img_color);
        cv::waitKey();
    }

    Eigen::Matrix4f FeatureTracker::exp(const Eigen::Vector3f& x) const
    {
        Eigen::Matrix4f A = Eigen::Matrix4f::Zero();
        for(int i = 0; i < A_.size(); i++)
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
        Eigen::Matrix3f Heigen = Hroi_ * H;
        cv::Mat liv_data, Hcv;
        cv::eigen2cv(Heigen, Hcv);
        cv::warpPerspective(liv_data_, liv_data, Hcv, cv::Size(rect_[2], rect_[3]), cv::INTER_LINEAR + cv::WARP_INVERSE_MAP);
        return liv_data;
    }

    std::vector<Eigen::Matrix<float, 256, 2>, Eigen::aligned_allocator<Eigen::Matrix<float, 256, 2>>>
    FeatureTracker::dataGradient(const cv::Mat &data) const
    {
        
        cv::Mat dx = shiftFrame(data, 1, 3) - data;
        cv::Mat dy = shiftFrame(data, 1, 0) - data;

        int width = data.cols;
        int height = data.rows;
        int channels = data.channels();

        std::vector<Eigen::Matrix<float, 256, 2>, Eigen::aligned_allocator<Eigen::Matrix<float, 256, 2>>> gradient;
        for (int i = 0; i < height; i++)
        {
            for (int j = 0; j < width; j++)
            {
                Eigen::Map<Eigen::VectorXf> dxij(dx.ptr<float>(i, j), 256);
                Eigen::Map<Eigen::VectorXf> dyij(dy.ptr<float>(i, j), 256);

                Eigen::Matrix<float, 256, 2> dxdy;
                dxdy.block(0, 0, 256, 1) = dxij;
                dxdy.block(0, 1, 256, 1) = dyij;
                //std::cout<<dxij<<std::endl;
                gradient.push_back(dxdy);
            }
        }
        
        /*
        int width = data.cols;
        int height = data.rows;
        int channels = data.channels();

        std::vector<Eigen::Matrix<float, 256, 2>, Eigen::aligned_allocator<Eigen::Matrix<float, 256, 2>>> gradient;
        for (int i = 0; i < height; i++)
        {
            for (int j = 0; j < width; j++)
            {
                Eigen::VectorXf dx;
                Eigen::VectorXf dy;
                if ((j == width - 1) || (i == height - 1))
                {
                    dx = Eigen::VectorXf::Zero(256);
                    dy = Eigen::VectorXf::Zero(256);
                }
                else
                {
                    const Eigen::Map<const Eigen::VectorXf> data_i0j0(data.ptr<float>(i, j), 256);
                    const Eigen::Map<const Eigen::VectorXf> data_i1j0(data.ptr<float>(i + 1, j), 256);
                    const Eigen::Map<const Eigen::VectorXf> data_i0j1(data.ptr<float>(i, j + 1), 256);
                    dx = data_i1j0 - data_i0j0;
                    dy = data_i0j1 - data_i0j0;
                }
                Eigen::Matrix<float, 256, 2> dxdy;
                dxdy.block(0, 0, 256, 1) = dx;
                dxdy.block(0, 1, 256, 1) = dy;
                gradient.push_back(dxdy);
                //std::cout << dxdy << std::endl;
            }
        }
        */

        return gradient;
    }

    cv::Mat FeatureTracker::shiftFrame(const cv::Mat frame, const int pixels, const int direction) const
    {
        //create a same sized temporary Mat with all the pixels flagged as invalid (-1)
        cv::Mat temp = cv::Mat::zeros(frame.size(), frame.type());

        switch (direction)
        {
        case (0): //ShiftUp
            frame(cv::Rect(0, pixels, frame.cols, frame.rows - pixels)).copyTo(temp(cv::Rect(0, 0, temp.cols, temp.rows - pixels)));
            break;
        case (1): //ShiftRight
            frame(cv::Rect(0, 0, frame.cols - pixels, frame.rows)).copyTo(temp(cv::Rect(pixels, 0, frame.cols - pixels, frame.rows)));
            break;
        case (2): //ShiftDown
            frame(cv::Rect(0, 0, frame.cols, frame.rows - pixels)).copyTo(temp(cv::Rect(0, pixels, frame.cols, frame.rows - pixels)));
            break;
        case (3): //ShiftLeft
            frame(cv::Rect(pixels, 0, frame.cols - pixels, frame.rows)).copyTo(temp(cv::Rect(0, 0, frame.cols - pixels, frame.rows)));
            break;
        default:
            std::cout << "Shift direction is not set properly" << std::endl;
        }
        return temp;
    }

    float FeatureTracker::residuals(const cv::Mat &data1, const cv::Mat &data2, Eigen::VectorXf &res)
    {

        const int width = data1.cols;
        const int height = data1.rows;
        const int channels = data1.channels();

        res.resize(width * height * channels);
        int cnt = 0;
        float m = 0;
        for (int i = 0; i < height; i++)
        {
            for (int j = 0; j < width; j++)
            {
                const float *pxl1 = data1.ptr<float>(i, j);
                const float *pxl2 = data2.ptr<float>(i, j);

                for (int c = 0; c < channels; c++)
                {
                    const float r = *(pxl1 + c) - *(pxl2 + c);
                    m += r*r;
                    res(i * width * channels + j * channels + c) = r;
                }
            }
        }
        float err = std::sqrt(m / width / height);

        //cv::cv2eigen(r,)
        //Eigen::Map<Eigen::VectorXf> eigenT(r.data(), r.rows * r.cols * r.dims);

        //Eigen::Map<Eigen::VectorXf> dxij(dx.ptr<float>(i, j), 256);
        //m = np.sum(residuals * residuals) return np.sqrt(m / (self.rect[2] * self.rect[3])),
        //residuals.reshape(-1)
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

        for (int v = 0; v < rect_(2); v++)//row
        {
            for (int u = 0; u < rect_(3); u++)//col
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
