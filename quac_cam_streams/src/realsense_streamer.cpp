#include "realsense_streamer/realsense_streamer.hpp"
#include <opencv4/opencv2/opencv.hpp>

std::atomic<bool> keep_running{true};
void handle_signal(int signum) { keep_running.store(false); }

RealsenseStreamer::RealsenseStreamer() : CamStreamer("realsense_streamer") {}

bool camera_is_connected(const std::string& serial_number)
{
  rs2::context ctx;
  rs2::device_list devices = ctx.query_devices();

  for (auto&& dev : devices)
    if (dev.get_info(RS2_CAMERA_INFO_SERIAL_NUMBER) == serial_number)
      return true;
    
  return false;
}

void RealsenseStreamer::pointcloud_loop()
{
  rs2::align align_to_depth(RS2_STREAM_DEPTH);

  while (keep_running.load())
  {
    rs.pointcloud.mutex.lock();
    if (rs.pointcloud.available) rs.pointcloud.working = true;
    rs.pointcloud.mutex.unlock();

    if (rs.pointcloud.working)
    {
      rs2::frameset aligned_frames = align_to_depth.process(rs.pointcloud.frameset);

      rs2::depth_frame depth_frame = aligned_frames.get_depth_frame();
      rs2::video_frame color_frame = aligned_frames.get_color_frame();

      rs.pointcloud.points = rs.pointcloud.pc.calculate(depth_frame);
      rs.pointcloud.pc.map_to(color_frame);

      auto vertices = rs.pointcloud.points.get_vertices();
      auto tex_coords = rs.pointcloud.points.get_texture_coordinates();

      const uint8_t* color_data = static_cast<const uint8_t*>(color_frame.get_data());

      size_t num_points = rs.pointcloud.points.size();

      pointcloud.msg.header.stamp = now();
      pointcloud.msg.header.frame_id = pointcloud.frame;

      pointcloud.msg.height = 1;
      pointcloud.msg.width = num_points;
      pointcloud.msg.is_dense = false;
      pointcloud.msg.is_bigendian = false;

      sensor_msgs::PointCloud2Modifier modifier(pointcloud.msg);
      modifier.setPointCloud2FieldsByString(2, "xyz", "rgb");
      modifier.resize(num_points);

      sensor_msgs::PointCloud2Iterator<float> iter_x(pointcloud.msg, "x");
      sensor_msgs::PointCloud2Iterator<float> iter_y(pointcloud.msg, "y");
      sensor_msgs::PointCloud2Iterator<float> iter_z(pointcloud.msg, "z");

      sensor_msgs::PointCloud2Iterator<uint8_t> iter_r(pointcloud.msg, "r");
      sensor_msgs::PointCloud2Iterator<uint8_t> iter_g(pointcloud.msg, "g");
      sensor_msgs::PointCloud2Iterator<uint8_t> iter_b(pointcloud.msg, "b");

      for (size_t i = 0; i < num_points; ++i)
      {
        const rs2::vertex& v = vertices[i];
        const rs2::texture_coordinate& t = tex_coords[i];

        if (v.z <= 0.f || !std::isfinite(v.z))
        {
          *iter_x = *iter_y = *iter_z = std::numeric_limits<float>::quiet_NaN();
          *iter_r = *iter_g = *iter_b = 0;
        }
        else
        {
          *iter_x = v.x;
          *iter_y = v.y;
          *iter_z = v.z;

          int u = std::min(std::max(int(t.u * (float)capture.depth.width), 0), capture.depth.width - 1);
          int v_px = std::min(std::max(int(t.v * (float)capture.depth.height), 0), capture.depth.height - 1);

          int idx = (v_px * capture.depth.width + u) * 3;

          *iter_b = color_data[idx + 0];
          *iter_g = color_data[idx + 1];
          *iter_r = color_data[idx + 2];
        }

        ++iter_x; ++iter_y; ++iter_z;
        ++iter_r; ++iter_g; ++iter_b;
      }

      pointcloud.publisher->publish(pointcloud.msg);
      rs.pointcloud.mutex.lock();
      rs.pointcloud.working = false;
      rs.pointcloud.available = false;
      rs.pointcloud.mutex.unlock();
    }
    else std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }
}

void RealsenseStreamer::run()
{
  // make sure camera is connected
  if (camera_is_connected(capture.device_id) == false)
  {
    RCLCPP_ERROR(get_logger(), "camera with serial %s not connected", capture.device_id.c_str());
    return;
  }

  rs.capture.cfg.enable_device(capture.device_id);
  rs.capture.cfg.enable_stream(RS2_STREAM_COLOR, capture.color.width, capture.color.height, RS2_FORMAT_BGR8, capture.fps);
  rs.capture.cfg.enable_stream(RS2_STREAM_DEPTH, capture.depth.width, capture.depth.height, RS2_FORMAT_Z16, capture.fps);

  rs2::align align_to_color(RS2_STREAM_COLOR);

  rs.capture.pipeline.start(rs.capture.cfg);

  signal(SIGTERM, handle_signal);
  signal(SIGINT, handle_signal);
  if (pointcloud.enable)
  {
    rs.pointcloud.working = false;
    rs.pointcloud.available = false;
    rs.pointcloud.thread = std::thread([this](){ pointcloud_loop();});
  }

  RCLCPP_INFO(get_logger(), "Camera opened");
  while (keep_running.load())
  {    
    rs2::frameset frames = rs.capture.pipeline.wait_for_frames();
    rs2::video_frame color_frame = frames.get_color_frame();
    rs2::depth_frame depth_frame = frames.get_depth_frame();

    push_gst_frame((void*)color_frame.get_data());

    rs2::video_stream_profile color_profile = 
    color_frame.get_profile().as<rs2::video_stream_profile>();
    rs2_intrinsics color_intr = color_profile.get_intrinsics();

    sensor_msgs::msg::CameraInfo cam_info;
    
    cam_info.header.stamp = now();
    cam_info.header.frame_id = image.frame;
    cam_info.width = color_intr.width;
    cam_info.height = color_intr.height;

    cam_info.k.at(0) = color_intr.fx;
    cam_info.k.at(2) = color_intr.ppx;
    cam_info.k.at(4) = color_intr.fy;
    cam_info.k.at(5) = color_intr.ppy;
    cam_info.k.at(8) = 1;

    cam_info.p.at(0) = cam_info.k.at(0);
    cam_info.p.at(1) = 0;
    cam_info.p.at(2) = cam_info.k.at(2);
    cam_info.p.at(3) = 0;
    cam_info.p.at(4) = 0;
    cam_info.p.at(5) = cam_info.k.at(4);
    cam_info.p.at(6) = cam_info.k.at(5);
    cam_info.p.at(7) = 0;
    cam_info.p.at(8) = 0;
    cam_info.p.at(9) = 0;
    cam_info.p.at(10) = 1;
    cam_info.p.at(11) = 0;

    cam_info.r.at(0) = 1.0;
    cam_info.r.at(1) = 0.0;
    cam_info.r.at(2) = 0.0;
    cam_info.r.at(3) = 0.0;
    cam_info.r.at(4) = 1.0;
    cam_info.r.at(5) = 0.0;
    cam_info.r.at(6) = 0.0;
    cam_info.r.at(7) = 0.0;
    cam_info.r.at(8) = 1.0;

    int coeff_size(5);
    if (color_intr.model == RS2_DISTORTION_KANNALA_BRANDT4)
    {
      cam_info.distortion_model = "equidistant";
      coeff_size = 4;
    }
    else cam_info.distortion_model = "plumb_bob";

    cam_info.d.resize(coeff_size);
    for (int i = 0; i < coeff_size; i++) cam_info.d.at(i) = color_intr.coeffs[i];

    cam_info_publisher->publish(cam_info);
    
    // image
    if (image.interval_i == 0 && image.enable)
    {
      rs2::frameset aligned_frames = align_to_color.process(frames);
      rs2::depth_frame aligned_depth_frame = aligned_frames.get_depth_frame();

      if (!aligned_depth_frame) continue;

      if (image.publisher->get_subscription_count())
      {
        quac_interfaces::msg::ImageBGRD msg;

        msg.header = cam_info.header;
        msg.height = capture.color.height;
        msg.width = capture.color.width;
        msg.depth_scale = aligned_depth_frame.get_units();

        msg.fx = color_intr.fx;
        msg.fy = color_intr.fy;
        msg.ppx = color_intr.ppx;
        msg.ppy = color_intr.ppy;

        msg.bgr_data.resize(capture.color.width * capture.color.height * 3);
        memcpy(msg.bgr_data.data(), color_frame.get_data(), capture.color.width * capture.color.height * 3);

        msg.depth_data.resize(capture.color.width * capture.color.height);
        memcpy(msg.depth_data.data(), aligned_depth_frame.get_data(), capture.color.width * capture.color.height * 2);

        image.publisher->publish(msg);
      }

      if(image.rtabmap_publisher->get_subscription_count())
      {
        rtabmap_msgs::msg::RGBDImage msg;
        msg.header = cam_info.header;
        msg.rgb_camera_info = cam_info;
        msg.depth_camera_info = cam_info;

        msg.depth_camera_info.p.at(3) = 0;
        msg.depth_camera_info.p.at(7) = 0;

        cv::Mat bgr(capture.color.height, capture.color.width, CV_8UC3, (void*)color_frame.get_data());

        cv::Mat rgb;
        cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);

        msg.rgb.header = cam_info.header;
        msg.rgb.height = capture.color.height;
        msg.rgb.width  = capture.color.width;
        msg.rgb.encoding = "rgb8";
        msg.rgb.is_bigendian = false;
        msg.rgb.step = static_cast<sensor_msgs::msg::Image::_step_type>(rgb.step);
        msg.rgb.data.resize(capture.color.width * capture.color.height * 3);
        memcpy(msg.rgb.data.data(), rgb.data, capture.color.width * capture.color.height * 3);

        cv::Mat depth(capture.color.height, capture.color.width, CV_8UC2, (void*)aligned_depth_frame.get_data());

        msg.depth.header = cam_info.header;
        msg.depth.height = capture.color.height;
        msg.depth.width  = capture.color.width;

        msg.depth.encoding = "16UC1";
        msg.depth.is_bigendian = false;
        msg.depth.step = static_cast<sensor_msgs::msg::Image::_step_type>(depth.step);
        msg.depth.data.resize(capture.color.width * capture.color.height * 2);
        memcpy(msg.depth.data.data(), depth.data, capture.color.width * capture.color.height * 2);


        image.rtabmap_publisher->publish(msg);
      }
    }

    if (image.enable) image.interval_i = (image.interval_i + 1) % image.interval;

    // pointcloud
    if (pointcloud.enable) pointcloud.interval_i++;
    if (pointcloud.interval_i >= pointcloud.interval && pointcloud.enable)
    {
      rs.pointcloud.mutex.lock();
      if (rs.pointcloud.working == false)
      {
        rs.pointcloud.available = true;
        rs.pointcloud.frameset = frames;
        pointcloud.interval_i = 0;
      }
      rs.pointcloud.mutex.unlock();
    }
  }

  if (pointcloud.enable) rs.pointcloud.thread.join();
  rs.capture.pipeline.stop();
  RCLCPP_INFO(get_logger(), "Closed camera");

  deinit();
}

int main (int argc, char *argv[])
{
  rclcpp::init(argc, argv);

  auto node = std::make_shared<RealsenseStreamer>();
  node->run();
  
  rclcpp::shutdown();
}