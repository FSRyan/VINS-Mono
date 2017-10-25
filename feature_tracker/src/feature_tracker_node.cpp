#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/Imu.h>
#include <cv_bridge/cv_bridge.h>
#include <message_filters/subscriber.h>

#include "feature_tracker.h"

#define SHOW_UNDISTORTION 0

vector<uchar> r_status;
vector<float> r_err;
queue<sensor_msgs::ImageConstPtr> img_buf;

ros::Publisher pub_img,pub_match;

FeatureTracker trackerData[NUM_OF_CAM];
double first_image_time;
int pub_count = 1;
bool first_image_flag = true;

void img_callback(const sensor_msgs::ImageConstPtr &img_msg)
{
    if(first_image_flag)
    {
        first_image_flag = false;
        first_image_time = img_msg->header.stamp.toSec();
    }

    // frequency control  频率控制，保证每秒钟处理的image不多于FREQ；
    if (round(1.0 * pub_count / (img_msg->header.stamp.toSec() - first_image_time)) <= FREQ)
    {
        PUB_THIS_FRAME = true;
        // reset the frequency control
        if (abs(1.0 * pub_count / (img_msg->header.stamp.toSec() - first_image_time) - FREQ) < 0.01 * FREQ)
        {
            first_image_time = img_msg->header.stamp.toSec();
            pub_count = 0;
        }
    }
    else
        PUB_THIS_FRAME = false;
    //转换为mat图像
    cv_bridge::CvImageConstPtr ptr = cv_bridge::toCvCopy(img_msg, sensor_msgs::image_encodings::MONO8);
    cv::Mat show_img = ptr->image;
    TicToc t_r;
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        ROS_DEBUG("processing camera %d", i);
        /*readImage内部函数：
        1.图像太亮或太暗，使用限制对比度的直方图均衡
        2.calcOpticalFlowPyrLK光流跟踪
        3.利用基础矩阵去除外点
        4.setMask()??是做什么的---鱼眼相机专用，remove edge noisy points
        5.检测角点，凑够MAX_CNT个点进行追踪
        6.根据是否发布该图片对prev_img\cur_img\forw_img的信息进行更新
        */
        if (i != 1 || !STEREO_TRACK)
            trackerData[i].readImage(ptr->image.rowRange(ROW * i, ROW * (i + 1)));//rowRange函数取的实际行数，只取到范围的右边界，而不取左边界,在此处指的是取第i张image

        else
        {
            if (EQUALIZE)
            {
                cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(); //限制对比度的自适应直方图均衡，提高图像对比度
                clahe->apply(ptr->image.rowRange(ROW * i, ROW * (i + 1)), trackerData[i].cur_img);
            }
            else
                trackerData[i].cur_img = ptr->image.rowRange(ROW * i, ROW * (i + 1));
        }

#if SHOW_UNDISTORTION
        trackerData[i].showUndistortion("undistrotion_" + std::to_string(i));
#endif
    }
//这部分没看，STEREO_TRACK=false
    if ( PUB_THIS_FRAME && STEREO_TRACK && trackerData[0].cur_pts.size() > 0)
    {
        pub_count++;
        r_status.clear();
        r_err.clear();
        TicToc t_o;
        cv::calcOpticalFlowPyrLK(trackerData[0].cur_img, trackerData[1].cur_img, trackerData[0].cur_pts, trackerData[1].cur_pts, r_status, r_err, cv::Size(21, 21), 3);
        ROS_DEBUG("spatial optical flow costs: %fms", t_o.toc());
        vector<cv::Point2f> ll, rr;
        vector<int> idx;
        for (unsigned int i = 0; i < r_status.size(); i++)
        {
            if (!inBorder(trackerData[1].cur_pts[i]))
                r_status[i] = 0;

            if (r_status[i])
            {
                idx.push_back(i);

                Eigen::Vector3d tmp_p;
                trackerData[0].m_camera->liftProjective(Eigen::Vector2d(trackerData[0].cur_pts[i].x, trackerData[0].cur_pts[i].y), tmp_p);
                tmp_p.x() = FOCAL_LENGTH * tmp_p.x() / tmp_p.z() + COL / 2.0;
                tmp_p.y() = FOCAL_LENGTH * tmp_p.y() / tmp_p.z() + ROW / 2.0;
                ll.push_back(cv::Point2f(tmp_p.x(), tmp_p.y()));

                trackerData[1].m_camera->liftProjective(Eigen::Vector2d(trackerData[1].cur_pts[i].x, trackerData[1].cur_pts[i].y), tmp_p);
                tmp_p.x() = FOCAL_LENGTH * tmp_p.x() / tmp_p.z() + COL / 2.0;
                tmp_p.y() = FOCAL_LENGTH * tmp_p.y() / tmp_p.z() + ROW / 2.0;
                rr.push_back(cv::Point2f(tmp_p.x(), tmp_p.y()));
            }
        }
        if (ll.size() >= 8)
        {
            vector<uchar> status;
            TicToc t_f;
            cv::findFundamentalMat(ll, rr, cv::FM_RANSAC, 1.0, 0.5, status);
            ROS_DEBUG("find f cost: %f", t_f.toc());
            int r_cnt = 0;
            for (unsigned int i = 0; i < status.size(); i++)
            {
                if (status[i] == 0)
                    r_status[idx[i]] = 0;
                r_cnt += r_status[idx[i]];
            }
        }
    }
//更新相机跟踪点的ID
    for (unsigned int i = 0;; i++)
    {
        bool completed = false;
        for (int j = 0; j < NUM_OF_CAM; j++)
            if (j != 1 || !STEREO_TRACK)
                completed |= trackerData[j].updateID(i);//更新ID
        if (!completed)
            break;
    }

   if (PUB_THIS_FRAME)
   {
        pub_count++;
        sensor_msgs::PointCloudPtr feature_points(new sensor_msgs::PointCloud);
        sensor_msgs::ChannelFloat32 id_of_point;
        sensor_msgs::ChannelFloat32 u_of_point;
        sensor_msgs::ChannelFloat32 v_of_point;

        feature_points->header = img_msg->header;
        feature_points->header.frame_id = "world";
        //归一化坐标？
       /*
        * Pc=[X/Z,Y/Z,1]T,Pc经过内参之后就得到像素坐标
        * Puv=K*Pc
        */
        vector<set<int>> hash_ids(NUM_OF_CAM);
        for (int i = 0; i < NUM_OF_CAM; i++)
        {
            if (i != 1 || !STEREO_TRACK)
            {
                auto un_pts = trackerData[i].undistortedPoints();
                auto &cur_pts = trackerData[i].cur_pts;
                auto &ids = trackerData[i].ids;
                for (unsigned int j = 0; j < ids.size(); j++)
                {
                    int p_id = ids[j];
                    hash_ids[i].insert(p_id);
                    geometry_msgs::Point32 p;
                    p.x = un_pts[j].x;
                    p.y = un_pts[j].y;
                    p.z = 1;

                    feature_points->points.push_back(p);
                    id_of_point.values.push_back(p_id * NUM_OF_CAM + i);//假设0-50存放左摄像头，那么右摄像头就是
                    u_of_point.values.push_back(cur_pts[j].x);
                    v_of_point.values.push_back(cur_pts[j].y);
                    ROS_ASSERT(inBorder(cur_pts[j]));
                }
            }
                //STEREO_TRACK=FALSE
            else if (STEREO_TRACK)
            {
                auto r_un_pts = trackerData[1].undistortedPoints();
                auto &ids = trackerData[0].ids;
                for (unsigned int j = 0; j < ids.size(); j++)
                {
                    if (r_status[j])
                    {
                        int p_id = ids[j];
                        hash_ids[i].insert(p_id);
                        geometry_msgs::Point32 p;
                        p.x = r_un_pts[j].x;
                        p.y = r_un_pts[j].y;
                        p.z = 1;

                        feature_points->points.push_back(p);
                        id_of_point.values.push_back(p_id * NUM_OF_CAM + i);
                    }
                }
            }
        }
        feature_points->channels.push_back(id_of_point);
        feature_points->channels.push_back(u_of_point);
        feature_points->channels.push_back(v_of_point);
        ROS_DEBUG("publish %f, at %f", feature_points->header.stamp.toSec(), ros::Time::now().toSec());
       //发布
        pub_img.publish(feature_points);//feature_points中points中的点是归一化坐标平面中的点，channels中的信息是其图像中的像素点
       //标记跟踪
        if (SHOW_TRACK)
        {
            ptr = cv_bridge::cvtColor(ptr, sensor_msgs::image_encodings::BGR8);

            //cv::Mat stereo_img(ROW * NUM_OF_CAM, COL, CV_8UC3);
            cv::Mat stereo_img = ptr->image;

            for (int i = 0; i < NUM_OF_CAM; i++)
            {
                cv::Mat tmp_img = stereo_img.rowRange(i * ROW, (i + 1) * ROW);
                cv::cvtColor(show_img, tmp_img, CV_GRAY2RGB);
                if (i != 1 || !STEREO_TRACK)
                {
                    for (unsigned int j = 0; j < trackerData[i].cur_pts.size(); j++)
                    {
                        double len = std::min(1.0, 1.0 * trackerData[i].track_cnt[j] / WINDOW_SIZE);
                        cv::circle(tmp_img, trackerData[i].cur_pts[j], 2, cv::Scalar(255 * (1 - len), 0, 255 * len), 2);
                        //char name[10];
                        //sprintf(name, "%d", trackerData[i].ids[j]);
                        //cv::putText(tmp_img, name, trackerData[i].cur_pts[j], cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
                    }
                }
                else
                {
                    for (unsigned int j = 0; j < trackerData[i].cur_pts.size(); j++)
                    {
                        if (r_status[j])
                        {
                            cv::circle(tmp_img, trackerData[i].cur_pts[j], 2, cv::Scalar(0, 255, 0), 2);
                            cv::line(stereo_img, trackerData[i - 1].cur_pts[j], trackerData[i].cur_pts[j] + cv::Point2f(0, ROW), cv::Scalar(0, 255, 0));
                        }
                    }
                }
            }
            /*
            cv::imshow("vis", stereo_img);
            cv::waitKey(5);
            */
            pub_match.publish(ptr->toImageMsg());
        }
    }
    ROS_INFO("whole feature tracker processing costs: %f", t_r.toc());
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "feature_tracker");
    ros::NodeHandle n("~");
    ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);
    readParameters(n);
    //读入相机的参数
    for (int i = 0; i < NUM_OF_CAM; i++)
        trackerData[i].readIntrinsicParameter(CAM_NAMES[i]);

    if(FISHEYE)
    {
        for (int i = 0; i < NUM_OF_CAM; i++)
        {
            trackerData[i].fisheye_mask = cv::imread(FISHEYE_MASK, 0);
            if(!trackerData[i].fisheye_mask.data)
            {
                ROS_INFO("load mask fail");
                ROS_BREAK();
            }
            else
                ROS_INFO("load mask success");
        }
    }

    ros::Subscriber sub_img = n.subscribe(IMAGE_TOPIC, 100, img_callback); //监听IMAGE_TOPIC, 有图像信息发布到IMAGE_TOPIC上时，执行回调函数

    pub_img = n.advertise<sensor_msgs::PointCloud>("feature", 1000);
    pub_match = n.advertise<sensor_msgs::Image>("feature_img",1000);
    /*
    if (SHOW_TRACK)
        cv::namedWindow("vis", cv::WINDOW_NORMAL);
    */
    ros::spin();
    return 0;
}
