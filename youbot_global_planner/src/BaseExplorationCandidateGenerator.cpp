#include <ros/ros.h>
#include <std_msgs/String.h>
#include <geometry_msgs/Twist.h>
#include <sensor_msgs/LaserScan.h>
#include <nav_msgs/OccupancyGrid.h>
#include <geometry_msgs/PoseStamped.h>
#include <Eigen/Dense>

class MapInfo {
    public:
        MapInfo(nav_msgs::OccupancyGrid map):
            map_(map)
        {
            map_post_ = Eigen::MatrixXd::Zero(map_.info.width, map_.info.height);
            for (unsigned int i = 0; i < map_.info.width; ++i) {
                for (unsigned int j = 0; j < map_.info.height; ++j) {
                    int index = i + j * map_.info.width;
                    if (map_.data[index] == 0) {
                        map_post_(i, j) = 1.0;  // Free space
                    } else if (map_.data[index] == 100) {
                        map_post_(i, j) = -1.0; // Occupied space
                    } else {
                        map_post_(i, j) = 0.0;  // Unknown space
                    }
                }
            }



        }
    private:
        nav_msgs::OccupancyGrid map_;
        Eigen::MatrixXd map_post_;
};

class BaseExplorationCandidateGenerator {
    public:
        /*
        Constructor for the ExplorationCandidateGenerator class.
        @param nh The ROS node handle.
        */
        BaseExplorationCandidateGenerator(int argc, char** argv):
            {
                ros::init(argc, argv, "base_exploration_candidate_generator");
                ros::NodeHandle nh_;
                ros::ServiceServer generator = nh_.advertiseService("generate_candidate", generateCandidate);
                ROS_INFO("Exploration candidate generator Service initialized.");
                
            }
        
        void GetConsecutiveMaps() {
            
        }

        
    private:
        ros::NodeHandle nh_;                            //The ROS node handle.
        ros::Subscriber map_sub_;                        //The subscriber to the map topic.
        ros::Publisher candidate_pub_;                   //The publisher to the candidate topic.
        nav_msgs::OccupancyGrid map_;                    //The map object.
        geometry_msgs::PoseStamped candidate_;           //The candidate object.
        
        void mapCallback(const nav_msgs::OccupancyGrid::ConstPtr& map) {
            map_ = *map;