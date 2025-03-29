#include "youbot_horizontal_cavity_detection/HorizontalCavityDetector.h"
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/PointIndices.h>
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
#include <pcl/ModelCoefficients.h>
#include <pcl/search/kdtree.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/extract_clusters.h>
#include "laser_geometry/laser_geometry.h"
#include <tf2_ros/transform_listener.h>
#include <geometry_msgs/TransformStamped.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Vector3.h>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <message_filters/sync_policies/approximate_time.h>
#include <ros/package.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <geometry_msgs/Point.h>
#include <nav_msgs/OccupancyGrid.h>
#include <grid_map_ros/GridMapRosConverter.hpp>
#include <grid_map_msgs/GridMap.h>
#include <grid_map_core/GridMap.hpp>
#include <grid_map_core/Polygon.hpp>
#include <grid_map_core/grid_map_core.hpp>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/conditional_removal.h>
#include <pcl/filters/crop_box.h>
#include <pcl/common/centroid.h>
#include <pcl/common/common.h>
// #include <cavity_detection_api/api.h>
#include <cavity_detection_msgs/Roi.h>

// -----------------HELPER FUNCTIONS-------------------------
namespace helpers {
    Eigen::Vector2d castRay(const grid_map::GridMap& map, const Eigen::Vector2d& origin, const Eigen::Vector2d& direction){
        Eigen::Vector2d current = origin;
        Eigen::Vector2d step = direction.normalized() * 0.05;
        const std::string layer = "layer";  // Your map layer
        const float threshold = 0.1f;

        while (map.isInside(current)) {
            grid_map::Index index;
            if (map.getIndex(current, index)) {
                if (map.isValid(index, layer)) {
                    float value = map.at(layer, index);
                    if (value > threshold || std::isnan(value)) {
                        return current-2*step;
                    }
                }
            }
            current += step;
        }
        return current;
    }
    void removeOutliersIQR(std::vector<double>& data) {
        if (data.size() < 4) return;
    
        std::sort(data.begin(), data.end());
        size_t q1_idx = data.size() / 4;
        size_t q3_idx = (3 * data.size()) / 4;
    
        double Q1 = data[q1_idx];
        double Q3 = data[q3_idx];
        double IQR = Q3 - Q1;
    
        double lower_bound = Q1 - 1.5 * IQR;
        double upper_bound = Q3 + 1.5 * IQR;
    
        std::vector<double> filtered;
        for (double val : data) {
            if (val >= lower_bound && val <= upper_bound)
                filtered.push_back(val);
        }
        data = filtered;
    }
    void removeOutliersIQRNormal(std::vector<Eigen::Vector3d>& normals) {
        if (normals.size() < 4) return;

        // Extract the x, y, and z components of the vectors
        std::vector<double> x_vals, y_vals, z_vals;
        for (const auto& normal : normals) {
            x_vals.push_back(normal.x());
            y_vals.push_back(normal.y());
            z_vals.push_back(normal.z());
        }

        // Remove outliers based on IQR for each component (x, y, z)
        removeOutliersIQR(x_vals);
        removeOutliersIQR(y_vals);
        removeOutliersIQR(z_vals);

        // Rebuild the filtered normals list
        std::vector<Eigen::Vector3d> filteredNormals;
        size_t min_size = std::min({x_vals.size(), y_vals.size(), z_vals.size()});
        for (size_t i = 0; i < min_size; ++i) {
            filteredNormals.emplace_back(x_vals[i], y_vals[i], z_vals[i]);
        }

        // Replace the original normals list with the filtered normals
        normals = filteredNormals;
    }
}
//-------------------------------------------------------------------------------
//---------------------------------Stud storage class definition-----------------------
//-------------------------------------------------------------------------------
Stud::Stud(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cluster):
    cloud(new pcl::PointCloud<pcl::PointXYZ>),
    clusterCount(1),
    isUpdated(true)
{
    update(cluster,false);
};

Stud::Stud(const Eigen::Vector3d& centroid, const Eigen::Vector3d& n, const pcl::PointCloud<pcl::PointXYZ>::Ptr& c,int cnt, float w, float h, float l,bool a):
    centroidPos(centroid),
    normal(n),
    cloud(c),
    clusterCount(cnt),
    width(w),
    height(h),
    length(l),
    isUpdated(false)
{};

Stud::~Stud(){}

void Stud::update(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cluster,bool hasModel){

    //Create plane segmentation model
    pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers (new pcl::PointIndices);
    pcl::SACSegmentation<pcl::PointXYZ> seg;
    seg.setOptimizeCoefficients(true);
    seg.setMethodType(pcl::SAC_RANSAC);
    //if there already exists a model, find the closest plane that matches the plane within its vicinity
    if (hasModel) {
        seg.setModelType (pcl::SACMODEL_PERPENDICULAR_PLANE);
        seg.setAxis(Eigen::Vector3f(normal.cast<float>()));
        seg.setEpsAngle(0.07);
        // ROS_INFO("Using existing normal model to fit plane");
    // else fit a new plane through the model
    } else {
        seg.setModelType (pcl::SACMODEL_PLANE);
        // ROS_INFO("Using new model normal model to fit plane");
    }
    seg.setMaxIterations(50);
    seg.setDistanceThreshold(0.02);
    seg.setInputCloud(cluster);
    seg.segment(*inliers,*coefficients);

    //Initialize properties of studs

    if (inliers->indices.size() > 0) {
        pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::ExtractIndices<pcl::PointXYZ> extract;
        extract.setInputCloud(cluster);
        extract.setIndices(inliers);
        extract.setNegative(false);
        extract.filter(*filtered_cloud);
        downsampleCloud(filtered_cloud);
        if (clusterCount == 0){
            removeStatisticalOutliers(cloud);
        }
        if (hasModel){
            *cloud+=*filtered_cloud;
        } else {
            *cloud=*filtered_cloud;
        }
    } else {
        ROS_INFO("Could not estimate a planar model for the given dataset.");
        return;
    }

    //Recalculate centroid average
    pcl::PointXYZ avg_centroid;
    pcl::computeCentroid(*cloud,avg_centroid);
    Eigen::Vector3d new_centroid(centroidPos[0],avg_centroid.y,avg_centroid.z);
    centroidPos = new_centroid;

    //Recalculate normal average
    Eigen::Vector3d avg_normal(coefficients->values[0],coefficients->values[1],0.0);
    avg_normal*=(avg_normal[1]<0?1:-1);
    avg_normal += normal * clusterCount;
    avg_normal /= (clusterCount+1);
    normal = avg_normal;
    normal.normalize();

    //Calculate width, height, length
    pcl::PointXYZ minPt, maxPt;
    pcl::getMinMax3D(*cloud,minPt,maxPt);
    // length = maxPt.x - minPt.x;
    width =  maxPt.y - minPt.y > 0.035 ? 0.035:maxPt.y - minPt.y;
    height = maxPt.z;
    length = maxPt.x - minPt.x;
    clusterCount++;
    isUpdated = true;
};

void Stud::downsampleCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr& cluster){
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered (new pcl::PointCloud<pcl::PointXYZ>);
    pcl::VoxelGrid<pcl::PointXYZ> vg;
    vg.setInputCloud(cluster);
    vg.setLeafSize(0.02f, 0.05f, 0.02f); 
    vg.filter (*cloud_filtered);
    *cluster = *cloud_filtered;
}

void Stud::removeStatisticalOutliers(pcl::PointCloud<pcl::PointXYZ>::Ptr& cluster){
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered (new pcl::PointCloud<pcl::PointXYZ>);
    pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
    sor.setInputCloud (cloud);
    sor.setMeanK (5);
    sor.setStddevMulThresh (1.0);
    sor.filter (*cloud_filtered);
    sor.setNegative (false);
    sor.filter (*cloud_filtered);
    *cluster = *cloud_filtered;
}

void Stud::print() const {
    ROS_INFO("Stud Info:");
    ROS_INFO("  Centroid Position: (%.3f, %.3f, %.3f)", 
             centroidPos.x(), centroidPos.y(), centroidPos.z());
    ROS_INFO("  Normal Vector:     (%.3f, %.3f, %.3f)", 
             normal.x(), normal.y(), normal.z());
    ROS_INFO("  Cluster Point Count: %zu", 
             cloud ? cloud->points.size() : 0);
    ROS_INFO("  Cluster Count: %d", clusterCount);
    ROS_INFO("  Dimensions (W x H x L): %.3f x %.3f x %.3f", 
             width, height, length);
};


//-------------------------------------------------------------------------------
//---------------------------------Channel storage class definition-----------------------
//-------------------------------------------------------------------------------
Channel::Channel(const std::vector<Stud>& StudPair, const Eigen::Affine3d& map_H_cavity) {
    length = StudPair[0].length;
    width = StudPair[1].centroidPos[1] - StudPair[0].centroidPos[1];
    height = StudPair[0].height;
    bounding_box << 0.0, length, StudPair[0].centroidPos[1], StudPair[1].centroidPos[1];
    Eigen::Affine3d pose = Eigen::Affine3d::Identity();
    pose.translation() = Eigen::Vector3d(1.0, 0.5 * (StudPair[1].centroidPos[1] + StudPair[0].centroidPos[1]), 0.0);
    filling_pos = map_H_cavity * pose;
    parentID = std::to_string(std::rand() % 100);
};

Channel::~Channel(){};

void Channel::print() {
    ROS_INFO_STREAM("Stud Info:");
    ROS_INFO_STREAM("  Channel ID: " << parentID);
    ROS_INFO_STREAM("Affine3d matrix: \n" << filling_pos.matrix());
    ROS_INFO_STREAM("Bounding box: " << bounding_box.transpose());
    ROS_INFO_STREAM("  Dimensions (W x H x L): " 
                    << width << " x " << height << " x " << length);
};


//-------------------------------------------------------------------------------
//---------------------------------HorizontalCavityDetector-----------------------
//-------------------------------------------------------------------------------

HorizontalCavityDetector::HorizontalCavityDetector(const std::string& nodeName):
    nh_(nodeName),
    tf2_buffer_(),
    tf_listener_(tf2_buffer_)
{
    pointCloudSub_.subscribe(nh_,"/velodyne_points",1);
    laserScanSub_.subscribe(nh_,"/scan",1);
    mapSub_ = nh_.subscribe("/map",10,&HorizontalCavityDetector::occupancyGridCallback,this);
    sync_ = std::make_shared<message_filters::TimeSynchronizer<sensor_msgs::PointCloud2, sensor_msgs::LaserScan>>(pointCloudSub_, laserScanSub_, 20);
    sync_->registerCallback(boost::bind(&HorizontalCavityDetector::pointCloudCallback, this, _1,_2));
    cluster_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("/cluster_points", 10);
    marker_pub_ = nh_.advertise<visualization_msgs::MarkerArray>("studs", 10);

    std::string package_path = ros::package::getPath("youbot_horizontal_cavity_detection");
    config_file_path_ = package_path + "/config/" + "horizontal_cavity_filtering_pipeline.json";
    fp_.loadParams(config_file_path_); // Pass the full path to the constructor
};

HorizontalCavityDetector::~HorizontalCavityDetector(){};

void HorizontalCavityDetector::occupancyGridCallback(const nav_msgs::OccupancyGrid::ConstPtr& occupancy_grid_msg){
    grid_map::GridMapRosConverter::fromOccupancyGrid(*occupancy_grid_msg, "layer", grid_map);
}

void HorizontalCavityDetector::pointCloudCallback(const sensor_msgs::PointCloud2::ConstPtr& msg, const sensor_msgs::LaserScan::ConstPtr& msg2){
    // ROS_INFO("CALLBACK!");
    fp_.updateParamsFromJSON(config_file_path_);
    //Convert the pointcloud message to a pcl pointcloud
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::fromROSMsg(*msg, *cloud);
    //COnvert the laserscan message to a pcl pointcloud
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud2(new pcl::PointCloud<pcl::PointXYZ>);
    laser_geometry::LaserProjection projector;
    sensor_msgs::PointCloud2 pntCloud;
    projector.projectLaser(*msg2, pntCloud, 6);
    pcl::fromROSMsg(pntCloud, *cloud2);
    //Get transform from map to base link and map to cavity frame
    try {
        geometry_msgs::TransformStamped transformStamped = tf2_buffer_.lookupTransform("base_link", "map", ros::Time(0));
        map_H_baselink_ = transformToEigen(transformStamped);
        transformStamped = tf2_buffer_.lookupTransform("map", "velodyne", ros::Time(0));
        map_H_velodyne_ = transformToEigen(transformStamped);
        transformStamped = tf2_buffer_.lookupTransform("velodyne", "map", ros::Time(0));
        velodyne_H_map_ = transformToEigen(transformStamped);
        transformStamped = tf2_buffer_.lookupTransform("map", "cavity", ros::Time(0));
        map_H_cavity_ = transformToEigen(transformStamped);
    } catch (tf2::TransformException &ex) {
        ROS_WARN("Could not get transform: %s", ex.what());
        return;
    }
    //Convert nav_msg/OccupancyGrid into a gridmap object
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_out = removeMapInliers(cloud,grid_map);
    // ROS_INFO("removed outliers");
    pcl::PointCloud<pcl::PointXYZ>::Ptr cavity_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    changePointCloudTransform(cloud_out,cavity_cloud,"velodyne","cavity");
    // ROS_INFO("converted frames");
    std::vector <pcl::PointIndices> clusters = regionGrowingSegmentation(cavity_cloud);
    // ROS_INFO("applied region growing");
    // ROS_INFO("Cluster vector has %d elements",clusters.size());
    aggregateClouds(cavity_cloud,clusters);
    // ROS_INFO("Cloud aggregated");
    equalizeStudVector();
    // if (isClusterThresholdReached()){
    for (const auto& stud : studVector_){
        stud.print();
    }
    // }
    publishStuds();
    publishStudClouds();

};

Eigen::Affine3d HorizontalCavityDetector::transformToEigen(const geometry_msgs::TransformStamped& transform) {
    Eigen::Translation3d translation(
        transform.transform.translation.x,
        transform.transform.translation.y,
        transform.transform.translation.z
    );
    Eigen::Quaterniond rotation(
        transform.transform.rotation.w,
        transform.transform.rotation.x,
        transform.transform.rotation.y,
        transform.transform.rotation.z
    );
    Eigen::Affine3d affine_matrix = translation * rotation;
    return affine_matrix;
}

//Line segmentation and clustering
//https://pointclouds.org/documentation/tutorials/cluster_extraction.html
void HorizontalCavityDetector::filterOutterWalls(const pcl::PointCloud<pcl::PointXYZ>::Ptr& input_cloud){
    // ROS_INFO("IN FILTER");
    
    //Create a storage point cloud object pointer to store remaining points after each filtering iteration
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_f (new pcl::PointCloud<pcl::PointXYZ>);
    //Euclidean Clustering
    pcl::VoxelGrid<pcl::PointXYZ> vg;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered (new pcl::PointCloud<pcl::PointXYZ>);
    vg.setInputCloud (input_cloud);
    vg.setLeafSize (fp_.ec_leaf_size_x, fp_.ec_leaf_size_y, fp_.ec_leaf_size_z);
    vg.filter (*cloud_filtered);

    //Create the Line segmentation model
    pcl::SACSegmentation<pcl::PointXYZ> seg;
    pcl::PointIndices::Ptr inliers (new pcl::PointIndices);
    pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_line (new pcl::PointCloud<pcl::PointXYZ> ());
    pcl::PCDWriter writer;
    seg.setOptimizeCoefficients (true);
    seg.setModelType (pcl::SACMODEL_LINE);
    seg.setMethodType (pcl::SAC_RANSAC);
    seg.setMaxIterations (fp_.ec_max_iterations);
    seg.setDistanceThreshold (fp_.ec_distance_threshold);

    // Loop that is in charge of the iterative line fitting and cluster removal
    int nr_points = (int) cloud_filtered->size ();
    while (cloud_filtered->size () > 0.3 * nr_points)
    {
        // Segment the largest planar component from the remaining cloud
        seg.setInputCloud (cloud_filtered);
        seg.segment (*inliers, *coefficients);
        if (inliers->indices.size () == 0)
        {
            std::cout << "Could not estimate a planar model for the given dataset." << std::endl;
            break;
        }

        // Extract the planar inliers from the input cloud
        pcl::ExtractIndices<pcl::PointXYZ> extract;
        extract.setInputCloud (cloud_filtered);
        extract.setIndices (inliers);
        extract.setNegative (false);

        // Get the points associated with the planar surface
        extract.filter (*cloud_line);

        // Remove the linear inliers, extract the rest
        extract.setNegative (true);
        extract.filter (*cloud_f);
        *cloud_filtered = *cloud_f;
    }

    // Creating the KdTree object for the search method of the extraction
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);
    tree->setInputCloud (cloud_filtered);

    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
    ec.setClusterTolerance (fp_.ec_cluster_tolerance);
    ec.setMinClusterSize (fp_.ec_min_cluster_size);
    ec.setMaxClusterSize (fp_.ec_max_cluster_size);
    ec.setSearchMethod (tree);
    ec.setInputCloud (cloud_filtered);
    ec.extract (cluster_indices);
    ROS_INFO("Number of clusters detected: %zu", cluster_indices.size());

    int j = 0;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr combined_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    for (const auto& cluster : cluster_indices) {
        uint8_t r = (j * 50) % 256;
        uint8_t g = (j * 100) % 256;
        uint8_t b = (j * 150) % 256;

        for (const auto& idx : cluster.indices) {
            pcl::PointXYZRGB point;
            point.x = (*cloud_filtered)[idx].x;
            point.y = (*cloud_filtered)[idx].y;
            point.z = (*cloud_filtered)[idx].z;
        
            // Assign color based on the cluster index (or any other color logic you want)
            point.r = r;
            point.g = g;
            point.b = b;
        
            combined_cloud->push_back(point);
        }
        j++;
    }

    // Now that all clusters are combined, publish them together
    ROS_INFO("Publishing combined cloud with %zu points", combined_cloud->size());
    // publishCluster(combined_cloud,"velodyne");

}

void HorizontalCavityDetector::publishCluster(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& combined_cloud,std::string frame) {
    sensor_msgs::PointCloud2 cluster_msg;
    pcl::toROSMsg(*combined_cloud, cluster_msg);
    cluster_msg.header.stamp = ros::Time::now();
    cluster_msg.header.frame_id = frame; // Adjust the frame ID as per your setup
    cluster_pub_.publish(cluster_msg);
}

void HorizontalCavityDetector::publishCluster(const pcl::PointCloud<pcl::PointXYZ>::Ptr& combined_cloud,std::string frame) {
    sensor_msgs::PointCloud2 cluster_msg;
    pcl::toROSMsg(*combined_cloud, cluster_msg);
    cluster_msg.header.stamp = ros::Time::now();
    cluster_msg.header.frame_id = frame; // Adjust the frame ID as per your setup
    cluster_pub_.publish(cluster_msg);
}

std::vector <pcl::PointIndices> HorizontalCavityDetector::regionGrowingSegmentation(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud){
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered (new pcl::PointCloud<pcl::PointXYZ>);
    *cloud_filtered = *cloud;
    
    pcl::search::Search<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);
    pcl::PointCloud <pcl::Normal>::Ptr normals (new pcl::PointCloud <pcl::Normal>);
    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> normal_estimator;
    normal_estimator.setSearchMethod (tree);
    normal_estimator.setInputCloud (cloud_filtered);
    normal_estimator.setKSearch (50);
    normal_estimator.compute (*normals);

    pcl::IndicesPtr indices (new std::vector <int>);
    pcl::removeNaNFromPointCloud(*cloud_filtered, *indices);

    pcl::RegionGrowing<pcl::PointXYZ, pcl::Normal> reg;
    reg.setMinClusterSize (fp_.reg_min_cluster_size);
    reg.setMaxClusterSize (fp_.reg_max_cluster_size);
    reg.setSearchMethod (tree);
    reg.setNumberOfNeighbours (fp_.reg_number_of_neighbours);
    reg.setInputCloud (cloud_filtered);
    reg.setIndices (indices);
    reg.setInputNormals (normals);
    reg.setSmoothnessThreshold (fp_.reg_smoothness_threshold);
    reg.setCurvatureThreshold (fp_.reg_curvature_threshold);

    std::vector <pcl::PointIndices> clusters;
    reg.extract (clusters);
    // publishCluster(reg.getColoredCloud(),"cavity");
    return clusters;

}

void HorizontalCavityDetector::publishLineSegmentsToRViz(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_filtered, const std::vector<pcl::ModelCoefficients::Ptr>& coefficients_list, ros::Publisher& marker_pub) {
    // Create a Marker message for multiple line segments
    visualization_msgs::Marker line_marker;
    line_marker.header.frame_id = "base_link";  // Set the frame (e.g., base_link)
    line_marker.header.stamp = ros::Time::now();
    line_marker.ns = "line_segments";
    line_marker.id = 0;  // Static ID for the marker (since we're adding multiple lines in one marker)
    line_marker.type = visualization_msgs::Marker::LINE_LIST;
    line_marker.action = visualization_msgs::Marker::ADD;

    // Set the line color (Red)
    line_marker.color.r = 1.0f;
    line_marker.color.g = 0.0f;
    line_marker.color.b = 0.0f;
    line_marker.color.a = 1.0;  // Opaque

    line_marker.scale.x = 0.02;  // Line width

    // Loop through all coefficients and create line segments
    for (size_t i = 0; i < coefficients_list.size(); ++i) {
    const pcl::ModelCoefficients::Ptr& coefficients = coefficients_list[i];

    // Extract line coefficients
    Eigen::Vector3f point_on_line(coefficients->values[0], coefficients->values[1], coefficients->values[2]);
    Eigen::Vector3f line_direction(coefficients->values[3], coefficients->values[4], coefficients->values[5]);

    // Create two points to define the line segment
    geometry_msgs::Point p1, p2;
    p1.x = point_on_line[0];
    p1.y = point_on_line[1];
    p1.z = point_on_line[2];

    // Scale the direction vector for visualization
    p2.x = p1.x + 0.5 * line_direction[0];  // Scale by 0.5 for visualization
    p2.y = p1.y + 0.5 * line_direction[1];
    p2.z = p1.z + 0.5 * line_direction[2];

    // Add the points to the marker
    line_marker.points.push_back(p1);
    line_marker.points.push_back(p2);
    }

    // Publish the marker with all the line segments
    marker_pub.publish(line_marker);
}

pcl::PointCloud<pcl::PointXYZ>::Ptr HorizontalCavityDetector::removeMapInliers(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, grid_map::GridMap& grid){
    
    //Transform the pointcloud from Velodyne Frame to Map frame
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered (new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_cloud (new pcl::PointCloud<pcl::PointXYZ> ());
    HorizontalCavityDetector::changePointCloudTransform(cloud,transformed_cloud,"velodyne","map");

    //Narrow region of interest based on information on cavity
    pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_filtered_cloud (new pcl::PointCloud<pcl::PointXYZ> ());
    Eigen::Vector3d cavity_frame_pos = map_H_cavity_.translation();
    // ROS_INFO("Cavity frame position: [%f, %f, %f]", cavity_frame_pos[0], cavity_frame_pos[1], cavity_frame_pos[2]);
    float boxfilter_xmin = cavity_frame_pos[0] + fp_.boxfilter_xmin;
    float boxfilter_xmax = cavity_frame_pos[0] + fp_.boxfilter_xmax;
    float boxfilter_ymin = cavity_frame_pos[1] + fp_.boxfilter_ymin;
    float boxfilter_ymax = cavity_frame_pos[1] + fp_.boxfilter_ymax;
    float boxfilter_zmin = cavity_frame_pos[2] + fp_.boxfilter_zmin;
    float boxfilter_zmax = cavity_frame_pos[2] + fp_.boxfilter_zmax;
    Eigen::Vector4f cavity_pos_max(boxfilter_xmax,boxfilter_ymax,boxfilter_zmax,1.0f);
    Eigen::Vector4f cavity_pos_min(boxfilter_xmin,boxfilter_ymin,boxfilter_zmin,1.0f);
    pcl::CropBox<pcl::PointXYZ > boxFilter1;
    boxFilter1.setMin(cavity_pos_min);
    boxFilter1.setMax(cavity_pos_max);
    boxFilter1.setInputCloud(transformed_cloud);
    boxFilter1.setNegative(false);
    boxFilter1.filter(*transformed_filtered_cloud);


    //Filter points that correspond with occupied gridmap points
    const auto& data = grid["layer"];
    Eigen::Vector2d position;
    for (grid_map::GridMapIterator iterator(grid); !iterator.isPastEnd(); ++iterator) {
        if (!std::isnan(grid.at("layer", *iterator)) && grid.at("layer", *iterator) > 10){
            grid.getPosition(*iterator,position);
            pcl::CropBox<pcl::PointXYZ > boxFilter2;
            float gridfilter_xmin = position[0] + fp_.gridfilter_xmin;
            float gridfilter_xmax = position[0] + fp_.gridfilter_xmax;
            float gridfilter_ymin = position[1] + fp_.gridfilter_ymin;
            float gridfilter_ymax = position[1] + fp_.gridfilter_ymax;
            float gridfilter_zmin =  fp_.gridfilter_zmin;
            float gridfilter_zmax =  fp_.gridfilter_zmax;
            Eigen::Vector4f pos_max(gridfilter_xmax,gridfilter_ymax,gridfilter_zmax,1.0f);
            Eigen::Vector4f pos_min(gridfilter_xmin,gridfilter_ymin,gridfilter_zmin,1.0f);
            boxFilter2.setMin(pos_min);
            boxFilter2.setMax(pos_max);
            boxFilter2.setInputCloud(transformed_filtered_cloud);
            boxFilter2.setNegative(true);
            boxFilter2.filter(*cloud_filtered);
            *transformed_filtered_cloud = *cloud_filtered; 
        }
        
    }

    //Transform the pointcloud from  Map frame to Velodyne Frame 
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_out (new pcl::PointCloud<pcl::PointXYZ>);
    // HorizontalCavityDetector::publishCluster(transformed_filtered_cloud,"map");
    HorizontalCavityDetector::changePointCloudTransform(transformed_filtered_cloud,cloud_out,"map","velodyne");
    return cloud_out;

}

void HorizontalCavityDetector::changePointCloudTransform(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloudin,pcl::PointCloud<pcl::PointXYZ>::Ptr& cloudout,std::string from,std::string to){
    try {
        geometry_msgs::TransformStamped transformStamped = tf2_buffer_.lookupTransform(to, from, ros::Time(0));
        Eigen::Affine3d to_H_from = transformToEigen(transformStamped);
        pcl::transformPointCloud (*cloudin, *cloudout, to_H_from);
    } catch (tf2::TransformException &ex) {
        ROS_WARN("Could not get transform: %s", ex.what());
        return;
    }
};

void HorizontalCavityDetector::aggregateClouds(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloudin,std::vector <pcl::PointIndices>& clusters) {
    //Reset Stud Vector update stauts from the previous iteration
    for (Stud& studObject : studVector_) {studObject.isUpdated = false;}
    
    //Set up the extractor to use while iterating through clusters
    pcl::ExtractIndices<pcl::PointXYZ> extract;
    extract.setInputCloud (cloudin);
    extract.setNegative (false);

    //cluster iterator loop
    for (const auto& cluster : clusters) {
        // extract current cluster
        pcl::PointCloud<pcl::PointXYZ>::Ptr cluster_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::PointIndices::ConstPtr clusterPtr = boost::make_shared<pcl::PointIndices>(cluster);
        extract.setIndices(clusterPtr);
        extract.filter(*cluster_cloud);
        
        //If the stud bay is empty, the current cluster belongs to a stud yet to be instantiated
        if (studVector_.empty()) {
            studVector_.push_back(Stud(cluster_cloud));
            continue;
            //Else find the stud the cluster is the closest to and update it with the new observation
        } else {
            //Find current cluster centroid
            pcl::PointXYZ currCentroid;
            pcl::computeCentroid(*cluster_cloud,currCentroid);
            //Iteratively compare the cluster y position to that of each stud
            bool used = false; // Check if a cluster has been used
            for (Stud& studObject : studVector_) {
                //No point in updating the stud twice
                if (studObject.isUpdated){continue;}
                //If a match is found then proceed to update the stud
                else if (abs(currCentroid.y - studObject.centroidPos[1]) < fp_.aggregation_distance_threshold) {
                    studObject.update(cluster_cloud,true);
                    studObject.isUpdated = true; //Mark the stud as updated
                    used = true;
                    break; //Exit the stud loop, a stud has been found
                }
            }
            //if no matching stud is found then a new stud has been found
            if (!used) {
                ROS_INFO("No match, Addigm a new Stud");
                studVector_.push_back(Stud(cluster_cloud));
            }
                

        }
    }
    ROS_INFO("Stud vector now has %d objects",studVector_.size());
};

void HorizontalCavityDetector::equalizeStudVector(){
    std::vector<double> lengthList;
    std::vector<double> centroidXList;
    std::vector<double> normalListX;
    std::vector<double> normalListY;
    std::vector<double> normalListZ;
    for (Stud& stud : studVector_){
        //Calculate ray vector
        Eigen::Vector3d alongVec3d = stud.normal.cross(Eigen::Vector3d(0,0,1));
        alongVec3d.normalize();
        alongVec3d*=alongVec3d[0]<0?-1:1;
        //modify centroid and ray vector so it is in map frame
        Eigen::Vector2d centroid = (map_H_cavity_ * Eigen::Vector4d(stud.centroidPos[0], stud.centroidPos[1], stud.centroidPos[2], 1.0)).head(2);
        Eigen::Vector3d alongVec3dTransformed = map_H_cavity_.rotation() * alongVec3d; 
        Eigen::Vector2d alongVec(alongVec3dTransformed[0], alongVec3dTransformed[1]);

        //get point where ray hits
        Eigen::Vector2d collision = helpers::castRay(grid_map,centroid,alongVec);
        collision = (map_H_cavity_.inverse() * Eigen::Vector4d(collision[0],collision[1],0.0,1.0)).head(2);
        Eigen::Vector2d y_inter = centroid + (-centroid[0] / alongVec[0]) * alongVec;
        lengthList.push_back((y_inter - collision).norm());
        centroidXList.push_back((0.5*(y_inter + collision))[0]);
        normalListX.push_back(stud.normal[0]);
        normalListY.push_back(stud.normal[1]);
        normalListZ.push_back(stud.normal[2]);
    }

    helpers::removeOutliersIQR(lengthList);
    helpers::removeOutliersIQR(centroidXList);
    helpers::removeOutliersIQR(centroidXList);
    helpers::removeOutliersIQR(normalListX);
    helpers::removeOutliersIQR(normalListY);
    helpers::removeOutliersIQR(normalListZ);

    double avg_length = std::accumulate(lengthList.begin(),lengthList.end(),0.0)/lengthList.size();
    double avg_centroid_x = std::accumulate(centroidXList.begin(),centroidXList.end(),0.0)/centroidXList.size();
    double avg_normal_x = std::accumulate(normalListX.begin(),normalListX.end(),0.0)/normalListX.size();
    double avg_normal_y = std::accumulate(normalListY.begin(),normalListY.end(),0.0)/normalListY.size();
    double avg_normal_z = std::accumulate(normalListZ.begin(),normalListZ.end(),0.0)/normalListZ.size();

    for (Stud& stud : studVector_){
        stud.length = avg_length;
        stud.centroidPos[0] = avg_centroid_x;
        stud.normal = Eigen::Vector3d(avg_normal_x,avg_normal_y,avg_normal_z);
    }

}

void HorizontalCavityDetector::publishStuds() {
    visualization_msgs::MarkerArray arr;
    int i = 1;
    for (const Stud& stud : studVector_){
        visualization_msgs::Marker marker;
        marker.header.frame_id = "cavity";  // Replace with your frame
        marker.header.stamp = ros::Time::now();
        marker.ns = "studs";
        marker.id = i;  // Each marker must have a unique ID per namespace
        marker.type = visualization_msgs::Marker::CUBE;
        marker.action = visualization_msgs::Marker::ADD;
        marker.pose.position.x = stud.centroidPos[0];
        marker.pose.position.y = stud.centroidPos[1];
        marker.pose.position.z = stud.centroidPos[2];
        //find quaternion from vectors
        Eigen::Vector3d unit(0,1,0);
        Eigen::Quaterniond q;
        q.setFromTwoVectors(unit,-stud.normal);
        marker.pose.orientation.x = q.x();
        marker.pose.orientation.y = q.y();
        marker.pose.orientation.z = q.z();
        marker.pose.orientation.w = q.w();
        marker.scale.x = stud.length;  // width (along local X-axis)
        marker.scale.y = stud.width;  // depth (along local Y-axis)
        marker.scale.z = stud.height;  // height (along local Z-axis)
        marker.color.r = 0.0f;
        marker.color.g = 1.0f;
        marker.color.b = 0.0f;
        marker.color.a = 1.0f;
        marker.lifetime = ros::Duration(0.0);  // 0 = permanent
        arr.markers.push_back(marker);
        i++;
    }
    marker_pub_.publish(arr);

};

void HorizontalCavityDetector::publishStudClouds(){
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloudout(new pcl::PointCloud<pcl::PointXYZ>);
    for (const Stud& stud : studVector_){
        *cloudout+=*(stud.cloud);
    }
    publishCluster(cloudout,"cavity");
};

bool HorizontalCavityDetector::isClusterThresholdReached(){
    std::vector<double> cnt;
    bool reached = true;
    double mean;
    double sizeVec = studVector_.size();
    for (Stud& stud : studVector_) {
        mean+=stud.clusterCount/sizeVec;
    }
    int finalMean = std::floor(sizeVec);
    return (finalMean%5 == 0);
};

void HorizontalCavityDetector::getChannelProperties(){
    std::sort(studVector_.begin(),studVector_.end(),[](const Stud& a,const Stud& b){
        return a.centroidPos[1] < b.centroidPos[1];
    });
    for(int i = 1; i < studVector_.size();i++){
        std::vector<Stud> studPair = {studVector_[i - 1], studVector_[i]};
        channelVector_.push_back(Channel(studPair,map_H_cavity_));
    }
}

int main(int argc, char** argv){
    ros::init(argc, argv, "horizontal_cavity_filtering_node");
    HorizontalCavityDetector detector("HorizontalDetection");
    ros::AsyncSpinner spinner(3);  // Two threads for spinning
    spinner.start();  // Start the async spinner
    ros::waitForShutdown();  // Wait for shutdown to stop spinning
}
