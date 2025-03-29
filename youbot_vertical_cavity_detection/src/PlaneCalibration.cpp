#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/features/normal_3d.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/common/common.h>
#include <Eigen/Dense>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>

class PlaneCalibration {
public:
    // Define a comparator to sort points by Z-value (height)
    static bool compareByZ(const pcl::PointXYZ& p1, const pcl::PointXYZ& p2)
    {
        return p1.z < p2.z;  // Sorting in ascending order (lowest z-value first)
    }

    double getZAverageOfLowestPoints(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, size_t n)
    {
        // Sort the points based on Z value (ascending)
        std::vector<pcl::PointXYZ, Eigen::aligned_allocator<pcl::PointXYZ>> sorted_points = cloud->points;
        std::sort(sorted_points.begin(), sorted_points.end(), compareByZ);

        // Calculate the Z average of the n lowest points
        double z_sum = 0.0;
        size_t count = std::min(n, sorted_points.size());  // Ensure we don't exceed the number of points
        for (size_t i = 0; i < count; ++i)
        {
            z_sum += sorted_points[i].z;
        }

        double z_average = (count > 0) ? (z_sum / count) : 0.0;
        return z_average;
    }

    Eigen::Vector3f fitPlaneAndGetAngle(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, size_t n)
    {
        // Select the n lowest points
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_lowest(new pcl::PointCloud<pcl::PointXYZ>);
        std::vector<pcl::PointXYZ, Eigen::aligned_allocator<pcl::PointXYZ>> sorted_points = cloud->points;
        std::sort(sorted_points.begin(), sorted_points.end(), compareByZ);
        
        size_t count = std::min(n, sorted_points.size());
        cloud_lowest->points = std::vector<pcl::PointXYZ, Eigen::aligned_allocator<pcl::PointXYZ>>(sorted_points.begin(), sorted_points.begin() + count);

        // Fit a plane to these n lowest points using SACSegmentation
        pcl::SACSegmentation<pcl::PointXYZ> seg;
        pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
        pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_plane(new pcl::PointCloud<pcl::PointXYZ>);

        // Set up the plane segmentation object
        seg.setOptimizeCoefficients(true);
        seg.setModelType(pcl::SACMODEL_PLANE);
        seg.setMethodType(pcl::SAC_RANSAC);
        seg.setDistanceThreshold(0.01);  // Maximum distance from a point to the plane for it to be considered inlier
        seg.setInputCloud(cloud_lowest);
        seg.segment(*inliers, *coefficients);

        if (inliers->indices.size() == 0)
        {
            std::cerr << "Could not estimate a planar model for the given dataset." << std::endl;
            return Eigen::Vector3f(0, 0, 0);  // No plane fitted
        }

        // The normal vector of the fitted plane
        Eigen::Vector3f normal(coefficients->values[0], coefficients->values[1], coefficients->values[2]);

        // Calculate the angle between the plane normal and the Z-axis (vertical plane)
        // The Z-axis is represented by the vector (0, 0, 1)
        Eigen::Vector3f z_axis(0, 0, 1);

        // Calculate the angle (in radians) using the dot product formula:
        // cos(theta) = dot(normal, z_axis) / (||normal|| * ||z_axis||)
        float dot_product = normal.dot(z_axis);
        float normal_magnitude = normal.norm();
        float z_axis_magnitude = z_axis.norm();
        float cos_angle = dot_product / (normal_magnitude * z_axis_magnitude);
        
        // Clamp cos_angle to avoid numerical issues that may cause NaN
        cos_angle = std::clamp(cos_angle, -1.0f, 1.0f);

        // Calculate the angle in radians
        float angle_rad = std::acos(cos_angle);

        // Convert radians to degrees
        float angle_deg = angle_rad * 180.0 / M_PI;

        std::cout << "Angle between the plane and the Z-axis: " << angle_deg << " degrees" << std::endl;

        return normal;  // Return the normal vector of the plane
    }
};
