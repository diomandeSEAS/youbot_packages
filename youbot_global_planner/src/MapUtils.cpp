#include <ros/ros.h>
#include <filters/filter_chain.hpp>
#include <std_msgs/String.h>
#include <geometry_msgs/Twist.h>
#include <tf2/convert.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <sensor_msgs/LaserScan.h>
#include <nav_msgs/OccupancyGrid.h>
#include <geometry_msgs/PoseStamped.h>
#include <Eigen/Core>
#include <array>
#include <grid_map_msgs/GridMap.h>
#include <grid_map_ros/grid_map_ros.hpp>
#include "grid_map_core/GridMap.hpp"
#include <cmath>
#include <yaml.h>
#include <visualization_msgs/Marker.h>

// This class should be used to handle the map information received from the map topic.
// TODO:
// 1. The constructor should starts a node that subscribes to the map topic.
//    The map metadata should be stored in relevant values. Height, width, resolution, etc.
//    The map itself should be stored in a Eigen::MatrixXd object.
// 2. The node should also suscribe to the tf topic to get the robot's pose wrt the map.
// 3. The class should have a function to generate a list of segments that represent the rays from the robot's pose to the map.
//    To do this, the method should create a normal segment located at a distance d from the robot pose and normal to its x axis.
//    n points should be selected from this segment and the rays should be generated from the robot pose to each of these points.
// 4. The class should have a method to check if a point is inside the map. To do this, the method should fetch all points that
//    belong to a ray, then pick the closest occupied one to the robot pose. Do this for each ray and return the set
// 5. The class should possess a variable that stores a map that scores each grid cell based on its level of exploration. During
//    each node spin, the value map should be updated with the new information.

#define DEG_TO_RAD(degree) ((degree) * (M_PI / 180.0))


class MapUtils {
    public:
        MapUtils(int argc, char **argv){
            init_exploration_cost = 1;
            gridMap.add("exploration_costMap");
            gridMap["exploration_costMap"].setConstant(init_exploration_cost);
            tf_sub_ = nh_.subscribe("tf", 1, &MapUtils::tfCallback, this);
            map_sub_ = nh_.subscribe("map", 1, &MapUtils::mapCallback, this);
            exploration_map_pub = nh_.advertise<grid_map_msgs::GridMap>("exploration_map", 1);
            viz_pub = nh_.advertise<visualization_msgs::Marker>("rays", 1);
            alpha_min = DEG_TO_RAD(-29);
            alpha_max = DEG_TO_RAD(29);
            rate = 1;
            n_rays = 21;
            n_raypoints = 20;
            d_lookahead = 2.0;
        }

        void mapCallback(const nav_msgs::OccupancyGrid::ConstPtr& map_) {
            ROS_INFO("Map received.");
            map_height_ = map_->info.height;
            map_width_ = map_->info.width;
            map_resolution_ = map_->info.resolution;
            grid_map::GridMapRosConverter::fromOccupancyGrid(*map_, "occupancy_map", gridMap);
            filterMap();
            updateMap();
            grid_map_msgs::GridMap message;
            grid_map::GridMapRosConverter::toMessage(gridMap, message);
            exploration_map_pub.publish(message);
        }

        void tfCallback(const tf2_msgs::TFMessage::ConstPtr& msg) {
            static tf2_ros::Buffer tfBuffer;
            static tf2_ros::TransformListener tfListener(tfBuffer);
            try {
            geometry_msgs::TransformStamped transformStamped = tfBuffer.lookupTransform("map", "base_footprint", ros::Time(0));
            Eigen::Matrix4d eigenTransform;
            eigenTransform.setIdentity();
            eigenTransform(0, 3) = transformStamped.transform.translation.x;
            eigenTransform(1, 3) = transformStamped.transform.translation.y;
            eigenTransform(2, 3) = transformStamped.transform.translation.z;
            Eigen::Quaterniond q(
                transformStamped.transform.rotation.w,
                transformStamped.transform.rotation.x,
                transformStamped.transform.rotation.y,
                transformStamped.transform.rotation.z
            );
            Eigen::Matrix3d rotationMatrix = q.toRotationMatrix();
            eigenTransform.block<3, 3>(0, 0) = rotationMatrix;
            robot_pose_ = eigenTransform;
            } catch (tf2::TransformException &ex) {
            ROS_WARN("TF exception: %s", ex.what());
            return;
            }
        }

        void publishViz(Eigen::MatrixXd ray_hitpoints){
            visualization_msgs::Marker marker;
            marker.header.frame_id = "map";
            marker.header.stamp = ros::Time();
            marker.ns = "rays";
            marker.id = 0;
            marker.type = visualization_msgs::Marker::LINE_LIST;
            marker.action = visualization_msgs::Marker::ADD;
            marker.pose.orientation.w = 1.0;
            marker.scale.x = 0.05;
            marker.color.r = 1.0;
            marker.color.a = 1.0;
            marker.points.clear();
            for (int i=0;i<n_rays;i++){
                geometry_msgs::Point start;
                start.x = robot_pose_(0,3);
                start.y = robot_pose_(1,3);
                start.z = 0.0;
                geometry_msgs::Point end;
                end.x = ray_hitpoints(i,0);
                end.y = ray_hitpoints(i,1);
                end.z = 0.0;
                marker.points.push_back(start);
                marker.points.push_back(end);
            }
            viz_pub.publish(marker);
        }

        void filterMap(){
            filters::FilterChain<grid_map::GridMap> filterChain_("grid_map::GridMap");
            std::string filterChainParametersName_;
            nh_.param("filter_chain_parameter_name", filterChainParametersName_, std::string("grid_map_filters"));
            if (!filterChain_.configure(filterChainParametersName_, nh_)) {
                ROS_ERROR("Could not configure the filter chain!");
            }
            gridMap.add("output_nan");
            gridMap.add("output_math_expr2");
            gridMap.add("output_final");

            // ROS_INFO("CONFIGURED");
            if (!filterChain_.update(gridMap,gridMap)) {
                ROS_ERROR("Could not update the filter chain!");
            }
            // ROS_INFO("Layers in the grid map:");
            // for (const auto& layer : gridMap.getLayers()) {
            //     ROS_INFO_STREAM("Layer: " << layer);
            // }

            ROS_INFO("Map filtering done.");
         
        }

        void updateMap(){
            Eigen::MatrixXd ray_endpoints = generateRays();
            Eigen::MatrixXd ray_hitpoints = Eigen::MatrixXd::Zero(n_rays, 2);
            for (int i=0;i<n_rays;i++){
                ray_hitpoints.row(i) = CheckRayHit(ray_endpoints.row(i));
            }
            publishViz(ray_hitpoints);
        }

        Eigen::Vector2d CheckRayHit(Eigen::Vector2d ray_endPoint){
            ROS_INFO("Ray endpoint: %f, %f", ray_endPoint(0), ray_endPoint(1));
            Eigen::Vector2d ray_point = ray_endPoint;
            Eigen::MatrixXd ray_hitpoint(n_raypoints, 2);
            grid_map::Index rayIndex;
            const auto& data_from = gridMap["output_final"];
            auto& data_to = gridMap["exploration_costMap"];
            for (int i = 5; i<n_raypoints; i++){
                ray_point = robot_pose_.block<2,1>(0,3) + i*(ray_endPoint - robot_pose_.block<2,1>(0,3))/n_raypoints;
                ROS_INFO("Robot pose: %f, %f", robot_pose_(0,3), robot_pose_(1,3));
                ROS_INFO("Ray point: %f, %f", ray_point(0), ray_point(1));
                gridMap.getIndex(ray_point, rayIndex);
                auto& val = data_from(rayIndex(0), rayIndex(1));
                ROS_INFO("Value at ray point: %f", val);
                if (val > 75){
                    data_to(rayIndex(0), rayIndex(1)) -= data_to(rayIndex(0), rayIndex(1)) > 0 ? 1.0 / rate : 0;
                    if (!isnan(val)){
                        break;
                    }
                }
                ROS_INFO("Ray point: %f, %f", ray_point(0), ray_point(1));
            }
            return ray_point;
        }

        Eigen::MatrixXd generateRays(){
            Eigen::Vector4d fwd_vec(d_lookahead, 0.0, 0.0, 1.0);
            Eigen::MatrixXd ray_endpoints(n_rays, 2);
            for (int i = 0; i<n_rays; i++){
                Eigen::Vector4d ray(d_lookahead, d_lookahead * tan(alpha_min + i * (alpha_max - alpha_min) / float(n_rays - 1)), 0.0, 1.0);
                ROS_INFO("Ray: %f, %f", ray(0), ray(1));
                Eigen::Vector4d ray_end = robot_pose_*ray;
                ray_endpoints.row(i) = ray_end.head(2);
            }
            return ray_endpoints;
        }
        
    
    private:
        ros::NodeHandle nh_;                                //The ROS node handle.
        ros::Subscriber map_sub_;                           //The subscriber to the map topic.
        ros::Subscriber tf_sub_;                            //The subscriber to the tf topic.
        ros::Publisher  viz_pub;                           //The publisher to the rays topic.
        ros::Publisher exploration_map_pub;                //The publisher to the exploration map topic.
        float rate;                                         //The rate at which the node should spin.
        Eigen::Matrix4d robot_pose_;                        //The robot's pose wrt the map.
        int map_height_;                                    //The height of the map.
        int map_width_;                                     //The width of the map.
        float map_resolution_;                              //The resolution of the map.
        grid_map::GridMap gridMap;                          //The grid map object.
        int n_rays;                                         //The number of rays to generate.                                     
        int n_raypoints;                                    //The number of points to generate for each ray.
        float d_lookahead;                                  //The distance to look ahead.
        float init_exploration_cost;                        //The initial exploration cost.
        float alpha_min;
        float alpha_max;

};

int main(int argc, char **argv) {
    ros::init(argc, argv, "map_utils");
    MapUtils mapUtils(argc, argv);
    ros::spin();
    return 0;
}