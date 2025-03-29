#ifndef HORIZONTAL_CAVITY_DETECTOR_H
#define HORIZONTAL_CAVITY_DETECTOR_H

#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/LaserScan.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/region_growing.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/features/normal_3d.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl_conversions/pcl_conversions.h>
#include <nlohmann/json.hpp>
#include <fstream>
#include <iostream>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <Eigen/Dense>
#include <tf2_ros/transform_listener.h>
#include <nav_msgs/OccupancyGrid.h>
#include <grid_map_ros/GridMapRosConverter.hpp>
#include <grid_map_msgs/GridMap.h>
#include <grid_map_core/GridMap.hpp>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/conditional_removal.h>
#include <pcl/filters/crop_box.h>
#include <tuple>





struct FilteringParameters {

    // Euclidean Cluster Extraction parameters
    double ec_leaf_size_x;
    double ec_leaf_size_y;
    double ec_leaf_size_z;
    int ec_min_cluster_size;
    int ec_max_cluster_size;
    double ec_cluster_tolerance;
    int ec_max_iterations;
    double ec_distance_threshold;

    // Region Growing parameters
    int reg_min_cluster_size;
    int reg_max_cluster_size;
    int reg_number_of_neighbours;
    double reg_smoothness_threshold;
    double reg_curvature_threshold;

    //gridmap filter parameters
    float boxfilter_xmin;
    float boxfilter_xmax;
    float boxfilter_ymin;
    float boxfilter_ymax;
    float boxfilter_zmin;
    float boxfilter_zmax;
    float gridfilter_xmin;
    float gridfilter_xmax;
    float gridfilter_ymin;
    float gridfilter_ymax;
    float gridfilter_zmin;
    float gridfilter_zmax;

    //cluster aggregation parameters
    double planefitting_eps_angle;
    int planefitting_max_iterations;
    double planefitting_distance_threshold;
    double aggregation_distance_threshold;


    // Default constructor with default values
    FilteringParameters()
        : ec_leaf_size_x(0.02), ec_leaf_size_y(0.02), ec_leaf_size_z(0.02),
          ec_min_cluster_size(10), ec_max_cluster_size(250000),
          ec_cluster_tolerance(0.5), ec_max_iterations(1000), ec_distance_threshold(0.02),
          reg_min_cluster_size(50), reg_max_cluster_size(1000000), reg_number_of_neighbours(30),
          reg_smoothness_threshold(0.05235988), reg_curvature_threshold(1.0), boxfilter_xmin(-0.2),
          boxfilter_xmax(3.0), boxfilter_ymin(-0.2), boxfilter_ymax(3.0), 
          boxfilter_zmin(0.02), boxfilter_zmax(0.3), gridfilter_xmin(-0.2), gridfilter_xmax(3.0),
          gridfilter_ymin(-0.2), gridfilter_ymax(3.0), gridfilter_zmin(-0.1), gridfilter_zmax(0.3),
          planefitting_eps_angle(0.07),planefitting_max_iterations(50),planefitting_distance_threshold(0.01),
          aggregation_distance_threshold(0.05) {}

    // Function to load parameters from JSON configuration file
    void loadParams(const std::string& config_file) {
        // Open the JSON configuration file
        std::ifstream config_stream(config_file);
        if (!config_stream.is_open()) {
            ROS_ERROR("Could not open config file: %s", config_file.c_str());
            // Set default values if the config file is not available
            setDefaults();
            return;
        }

        // Parse the JSON content
        nlohmann::json config_json;
        config_stream >> config_json;

        // Load the Euclidean Cluster Extraction parameters
        ec_leaf_size_x = config_json["ec"].value("leaf_size_x", 0.02);
        ec_leaf_size_y = config_json["ec"].value("leaf_size_y", 0.02);
        ec_leaf_size_z = config_json["ec"].value("leaf_size_z", 0.02);
        ec_max_iterations = config_json["ec"].value("max_iterations", 1000);
        ec_distance_threshold = config_json["ec"].value("distance_threshold", 0.02);
        ec_cluster_tolerance = config_json["ec"].value("cluster_tolerance", 0.5);
        ec_min_cluster_size = config_json["ec"].value("min_cluster_size", 10);
        ec_max_cluster_size = config_json["ec"].value("max_cluster_size", 250000);

        // Load the Region Growing parameters
        reg_min_cluster_size = config_json["reg"].value("min_cluster_size", 50);
        reg_max_cluster_size = config_json["reg"].value("max_cluster_size", 1000000);
        reg_number_of_neighbours = config_json["reg"].value("number_of_neighbours", 30);
        reg_smoothness_threshold = config_json["reg"].value("smoothness_threshold", 0.05235988);  // in radians
        reg_curvature_threshold = config_json["reg"].value("curvature_threshold", 1.0);

        //Load the gridmap filter parameters
        boxfilter_xmin = config_json["mf"].value("boxfilter_xmin", -0.2);
        boxfilter_xmax = config_json["mf"].value("boxfilter_xmax", 3.0);
        boxfilter_ymin = config_json["mf"].value("boxfilter_ymin", -0.2);
        boxfilter_ymax = config_json["mf"].value("boxfilter_ymax", 3.0);
        boxfilter_zmin = config_json["mf"].value("boxfilter_zmin", 0.02);
        boxfilter_zmax = config_json["mf"].value("boxfilter_zmax", 0.3);
        gridfilter_xmin = config_json["mf"].value("gridfilter_xmin", -0.2);
        gridfilter_xmax = config_json["mf"].value("gridfilter_xmax", 3.0);
        gridfilter_ymin = config_json["mf"].value("gridfilter_ymin", -0.2);
        gridfilter_ymax = config_json["mf"].value("gridfilter_ymax", 3.0);
        gridfilter_zmin = config_json["mf"].value("gridfilter_zmin", -0.1);
        gridfilter_zmax = config_json["mf"].value("gridfilter_zmax", 0.3);

        planefitting_eps_angle = config_json["ca"].value("planefitting_eps_angle", 0.07);
        planefitting_max_iterations = config_json["ca"].value("planefitting_max_iterations", 50);
        planefitting_distance_threshold = config_json["ca"].value("planefitting_max_iterations", 0.01);
        aggregation_distance_threshold = config_json["ca"].value("aggregation_distance_threshold", 0.05);

        // ROS_INFO("Loaded parameters from config file:");
        // ROS_INFO("Euclidean Cluster Extraction parameters:");
        // ROS_INFO("  Leaf size: [%f, %f, %f]", ec_leaf_size_x, ec_leaf_size_y, ec_leaf_size_z);
        // ROS_INFO("  Max iterations: %d", ec_max_iterations);
        // ROS_INFO("  Distance threshold: %f", ec_distance_threshold);
        // ROS_INFO("  Cluster tolerance: %f", ec_cluster_tolerance);
        // ROS_INFO("  Min cluster size: %d", ec_min_cluster_size);
        // ROS_INFO("  Max cluster size: %d", ec_max_cluster_size);

        // ROS_INFO("Region Growing parameters:");
        // ROS_INFO("  Min cluster size: %d", reg_min_cluster_size);
        // ROS_INFO("  Max cluster size: %d", reg_max_cluster_size);
        // ROS_INFO("  Number of neighbours: %d", reg_number_of_neighbours);
        // ROS_INFO("  Smoothness threshold: %f", reg_smoothness_threshold);
        // ROS_INFO("  Curvature threshold: %f", reg_curvature_threshold);

        // ROS_INFO("Gridmap filter parameters:");
        // ROS_INFO("  Boxfilter xmin: %f", boxfilter_xmin);
        // ROS_INFO("  Boxfilter xmax: %f", boxfilter_xmax);
        // ROS_INFO("  Boxfilter ymin: %f", boxfilter_ymin);
        // ROS_INFO("  Boxfilter ymax: %f", boxfilter_ymax);
        // ROS_INFO("  Boxfilter zmin: %f", boxfilter_zmin);
        // ROS_INFO("  Boxfilter zmax: %f", boxfilter_zmax);
        // ROS_INFO("  Gridfilter xmin: %f", gridfilter_xmin);
        // ROS_INFO("  Gridfilter xmax: %f", gridfilter_xmax);
        // ROS_INFO("  Gridfilter ymin: %f", gridfilter_ymin);
        // ROS_INFO("  Gridfilter ymax: %f", gridfilter_ymax);
        // ROS_INFO("  Gridfilter zmin: %f", gridfilter_zmin);
        // ROS_INFO("  Gridfilter zmax: %f", gridfilter_zmax);
    }

    void updateParamsFromJSON(const std::string& config_file) {
        // Open the JSON configuration file
        std::ifstream config_stream(config_file);
        if (!config_stream.is_open()) {
            ROS_ERROR("Could not open config file: %s", config_file.c_str());
            return;  // No update is done if the file cannot be opened
        }

        // Parse the JSON content
        nlohmann::json config_json;
        config_stream >> config_json;

        // Check and update Euclidean Cluster Extraction parameters
        if (config_json["ec"].contains("leaf_size_x")) {
            ec_leaf_size_x = config_json["ec"]["leaf_size_x"];
        }
        if (config_json["ec"].contains("leaf_size_y")) {
            ec_leaf_size_y = config_json["ec"]["leaf_size_y"];
        }
        if (config_json["ec"].contains("leaf_size_z")) {
            ec_leaf_size_z = config_json["ec"]["leaf_size_z"];
        }
        if (config_json["ec"].contains("max_iterations")) {
            ec_max_iterations = config_json["ec"]["max_iterations"];
        }
        if (config_json["ec"].contains("distance_threshold")) {
            ec_distance_threshold = config_json["ec"]["distance_threshold"];
        }
        if (config_json["ec"].contains("cluster_tolerance")) {
            ec_cluster_tolerance = config_json["ec"]["cluster_tolerance"];
        }
        if (config_json["ec"].contains("min_cluster_size")) {
            ec_min_cluster_size = config_json["ec"]["min_cluster_size"];
        }
        if (config_json["ec"].contains("max_cluster_size")) {
            ec_max_cluster_size = config_json["ec"]["max_cluster_size"];
        }

        // Check and update Region Growing parameters
        if (config_json["reg"].contains("min_cluster_size")) {
            reg_min_cluster_size = config_json["reg"]["min_cluster_size"];
        }
        if (config_json["reg"].contains("max_cluster_size")) {
            reg_max_cluster_size = config_json["reg"]["max_cluster_size"];
        }
        if (config_json["reg"].contains("number_of_neighbours")) {
            reg_number_of_neighbours = config_json["reg"]["number_of_neighbours"];
        }
        if (config_json["reg"].contains("smoothness_threshold")) {
            reg_smoothness_threshold = config_json["reg"]["smoothness_threshold"];
        }
        if (config_json["reg"].contains("curvature_threshold")) {
            reg_curvature_threshold = config_json["reg"]["curvature_threshold"];
        }

        // Check and update gridmap filter parameters
        if (config_json["mf"].contains("boxfilter_xmin")) {
            boxfilter_xmin = config_json["mf"]["boxfilter_xmin"];
        }
        if (config_json["mf"].contains("boxfilter_xmax")) {
            boxfilter_xmax = config_json["mf"]["boxfilter_xmax"];
        }
        if (config_json["mf"].contains("boxfilter_ymin")) {
            boxfilter_ymin = config_json["mf"]["boxfilter_ymin"];
        }
        if (config_json["mf"].contains("boxfilter_ymax")) {
            boxfilter_ymax = config_json["mf"]["boxfilter_ymax"];
        }
        if (config_json["mf"].contains("boxfilter_zmin")) {
            boxfilter_zmin = config_json["mf"]["boxfilter_zmin"];
        }
        if (config_json["mf"].contains("boxfilter_zmax")) {
            boxfilter_zmax = config_json["mf"]["boxfilter_zmax"];
        }
        if (config_json["mf"].contains("gridfilter_xmin")) {
            gridfilter_xmin = config_json["mf"]["gridfilter_xmin"];
        }
        if (config_json["mf"].contains("gridfilter_xmax")) {
            gridfilter_xmax = config_json["mf"]["gridfilter_xmax"];
        }
        if (config_json["mf"].contains("gridfilter_ymin")) {
            gridfilter_ymin = config_json["mf"]["gridfilter_ymin"];
        }
        if (config_json["mf"].contains("gridfilter_ymax")) {
            gridfilter_ymax = config_json["mf"]["gridfilter_ymax"];
        }
        if (config_json["mf"].contains("gridfilter_zmin")) {
            gridfilter_zmin = config_json["mf"]["gridfilter_zmin"];
        }
        if (config_json["mf"].contains("gridfilter_zmax")) {
            gridfilter_zmax = config_json["mf"]["gridfilter_zmax"];
        }

        //Check update cluster aggregation
        if (config_json["ca"].contains("planefitting_eps_angle")) {
            planefitting_eps_angle = config_json["ca"]["planefitting_eps_angle"];
        }
        if (config_json["ca"].contains("planefitting_max_iterations")) {
            planefitting_eps_angle = config_json["ca"]["planefitting_max_iterations"];
        }
        if (config_json["ca"].contains("planefitting_distance_threshold")) {
            planefitting_eps_angle = config_json["ca"]["planefitting_distance_threshold"];
        }
        if (config_json["ca"].contains("aggregation_distance_threshold")) {
            planefitting_eps_angle = config_json["ca"]["aggregation_distance_threshold"];
        }






        // ROS_INFO("Updated parameters from config file:");
        // ROS_INFO("Euclidean Cluster Extraction parameters:");
        // ROS_INFO("  Leaf size: [%f, %f, %f]", ec_leaf_size_x, ec_leaf_size_y, ec_leaf_size_z);
        // ROS_INFO("  Max iterations: %d", ec_max_iterations);
        // ROS_INFO("  Distance threshold: %f", ec_distance_threshold);
        // ROS_INFO("  Cluster tolerance: %f", ec_cluster_tolerance);
        // ROS_INFO("  Min cluster size: %d", ec_min_cluster_size);
        // ROS_INFO("  Max cluster size: %d", ec_max_cluster_size);

        // ROS_INFO("Region Growing parameters:");
        // ROS_INFO("  Min cluster size: %d", reg_min_cluster_size);
        // ROS_INFO("  Max cluster size: %d", reg_max_cluster_size);
        // ROS_INFO("  Number of neighbours: %d", reg_number_of_neighbours);
        // ROS_INFO("  Smoothness threshold: %f", reg_smoothness_threshold);
        // ROS_INFO("  Curvature threshold: %f", reg_curvature_threshold);

        // ROS_INFO("Gridmap filter parameters:");
        // ROS_INFO("  Boxfilter xmin: %f", boxfilter_xmin);
        // ROS_INFO("  Boxfilter xmax: %f", boxfilter_xmax);
        // ROS_INFO("  Boxfilter ymin: %f", boxfilter_ymin);
        // ROS_INFO("  Boxfilter ymax: %f", boxfilter_ymax);
        // ROS_INFO("  Boxfilter zmin: %f", boxfilter_zmin);
        // ROS_INFO("  Boxfilter zmax: %f", boxfilter_zmax);
        // ROS_INFO("  Gridfilter xmin: %f", gridfilter_xmin);
        // ROS_INFO("  Gridfilter xmax: %f", gridfilter_xmax);
        // ROS_INFO("  Gridfilter ymin: %f", gridfilter_ymin);
        // ROS_INFO("  Gridfilter ymax: %f", gridfilter_ymax);
        // ROS_INFO("  Gridfilter zmin: %f", gridfilter_zmin);
        // ROS_INFO("  Gridfilter zmax: %f", gridfilter_zmax);
    }

    // Function to set default values for all parameters
    void setDefaults() {
        ec_leaf_size_x = 0.02;
        ec_leaf_size_y = 0.02;
        ec_leaf_size_z = 0.02;
        ec_min_cluster_size = 10;
        ec_max_cluster_size = 250000;
        ec_cluster_tolerance = 0.5;
        ec_max_iterations = 1000;
        ec_distance_threshold = 0.02;

        reg_min_cluster_size = 50;
        reg_max_cluster_size = 1000000;
        reg_number_of_neighbours = 30;
        reg_smoothness_threshold = 0.05235988;
        reg_curvature_threshold = 1.0;

        planefitting_eps_angle=  0.07;
        planefitting_max_iterations=  50;
        planefitting_distance_threshold=  0.01;
        aggregation_distance_threshold=  0.10;
    }
};


class Stud {
    public:
        Stud(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cluster);
        Stud(const Eigen::Vector3d& centroid,
            const Eigen::Vector3d& n,
            const pcl::PointCloud<pcl::PointXYZ>::Ptr& c, int cnt,
            float w, float h, float l,bool a);
        ~Stud();
        void update(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cluster,bool hasModel);
        void downsampleCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr& cluster);
        void removeStatisticalOutliers(pcl::PointCloud<pcl::PointXYZ>::Ptr& cluster);
        void print() const;

        Eigen::Vector3d centroidPos;
        Eigen::Vector3d normal;
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud;
        int clusterCount;
        float width;
        float height;
        float length;
        bool isUpdated;
};

class Channel {
    public:
        Channel(const std::vector<Stud>& StudPair,const Eigen::Affine3d& map_H_cavity);
        ~Channel();
        void print();

        //member variables
        float length;
        float width;
        float height;
        Eigen::Affine3d filling_pos;
        Eigen::Vector4f bounding_box; //minx maxx miny maxy
        std::string parentID;


};


class HorizontalCavityDetector {
    public:
        HorizontalCavityDetector(const std::string& nodeName);
        ~HorizontalCavityDetector();
        void pointCloudCallback(const sensor_msgs::PointCloud2::ConstPtr& msg, const sensor_msgs::LaserScan::ConstPtr& msg2);
        void filterOutterWalls(const pcl::PointCloud<pcl::PointXYZ>::Ptr& input_cloud);
        Eigen::Affine3d transformToEigen(const geometry_msgs::TransformStamped& transform);
        void publishCluster(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud_cluster,std::string frame);
        void publishCluster(const pcl::PointCloud<pcl::PointXYZ>::Ptr& combined_cloud,std::string frame);
        void publishLineSegmentsToRViz(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_filtered, const std::vector<pcl::ModelCoefficients::Ptr>& coefficients_list, ros::Publisher& marker_pub);
        std::vector <pcl::PointIndices> regionGrowingSegmentation(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud);
        pcl::PointCloud<pcl::PointXYZ>::Ptr removeMapInliers(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, grid_map::GridMap& grid);
        void occupancyGridCallback(const nav_msgs::OccupancyGrid::ConstPtr& map_msg);
        void changePointCloudTransform(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloudin,pcl::PointCloud<pcl::PointXYZ>::Ptr& cloudout,std::string from,std::string to);
        void aggregateClouds(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloudin,std::vector <pcl::PointIndices>& clusters);
        void equalizeStudVector();
        void publishStuds();
        void publishStudClouds();
        bool isClusterThresholdReached();
        void getChannelProperties();

    private:


    
        ros::NodeHandle nh_;
        message_filters::Subscriber<sensor_msgs::PointCloud2> pointCloudSub_; 
        message_filters::Subscriber<sensor_msgs::LaserScan> laserScanSub_;
        ros::Subscriber mapSub_;
        std::shared_ptr<message_filters::TimeSynchronizer<sensor_msgs::PointCloud2, sensor_msgs::LaserScan>> sync_;
        tf2_ros::TransformListener tf_listener_;
        tf2_ros::Buffer tf2_buffer_; 
        Eigen::Affine3d map_H_baselink_;
        Eigen::Affine3d map_H_cavity_;
        Eigen::Affine3d map_H_velodyne_;
        Eigen::Affine3d velodyne_H_map_;
        ros::Publisher cluster_pub_;
        ros::Publisher marker_pub_;
        FilteringParameters fp_;
        std::string config_file_path_;
        grid_map::GridMap grid_map;
        std::vector<Stud> studVector_;
        std::vector<Channel> channelVector_;
};

#endif