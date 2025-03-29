
#include <ros/ros.h>
#include <nav_msgs/OccupancyGrid.h>
#include <grid_map_ros/GridMapRosConverter.hpp>
#include <grid_map_msgs/GridMap.h>
#include <grid_map_core/GridMap.hpp>

class OccupancyGridToGridMapNode
{
public:
    OccupancyGridToGridMapNode()
    {
        ros::NodeHandle nh;
        grid_map_pub_ = nh.advertise<grid_map_msgs::GridMap>("/grid_map_unfiltered", 10);
        occupancy_grid_sub_ = nh.subscribe("/map", 10, &OccupancyGridToGridMapNode::occupancyGridCallback, this);
        ROS_INFO("OccupancyGridToGridMapNode initialized.");
    }

    void occupancyGridCallback(const nav_msgs::OccupancyGrid::ConstPtr& occupancy_grid_msg)
    {
        grid_map::GridMap grid_map;
        grid_map::GridMapRosConverter::fromOccupancyGrid(*occupancy_grid_msg, "layer", grid_map);

        grid_map_msgs::GridMap grid_map_msg;
        grid_map::GridMapRosConverter::toMessage(grid_map, grid_map_msg);

        grid_map_pub_.publish(grid_map_msg);
        ROS_INFO("Published grid map.");
    }

private:
    ros::Publisher grid_map_pub_;
    ros::Subscriber occupancy_grid_sub_;
};

int main(int argc, char** argv)
{
    ros::init(argc, argv, "occupancy_grid_to_grid_map_node");
    OccupancyGridToGridMapNode node;
    ros::spin();
    return 0;
}