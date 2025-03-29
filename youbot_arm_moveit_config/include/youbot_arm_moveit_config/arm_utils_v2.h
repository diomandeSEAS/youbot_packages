#ifndef ARM_UTILS_V2_H
#define ARM_UTILS_V2_H

#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/planning_scene_interface/planning_scene_interface.h>
#include <moveit_msgs/DisplayRobotState.h>
#include <moveit_msgs/DisplayTrajectory.h>
#include <moveit_msgs/CollisionObject.h>
#include <moveit_visual_tools/moveit_visual_tools.h>
#include <moveit/robot_model_loader/robot_model_loader.h>
#include <moveit/robot_model/robot_model.h>
#include <moveit/robot_state/robot_state.h>
#include <moveit_msgs/PlanningScene.h>
#include <moveit_msgs/GetStateValidity.h>
#include <moveit_msgs/ApplyPlanningScene.h>
#include <moveit/robot_state/conversions.h>
#include <ros/ros.h>
#include <moveit/planning_scene_monitor/planning_scene_monitor.h>
#include <moveit/planning_pipeline/planning_pipeline.h>
#include <moveit/kinematic_constraints/utils.h>
#include <moveit/kinematic_constraints/kinematic_constraint.h>
#include <moveit/planning_interface/planning_response.h>
#include <tf2_ros/transform_listener.h>
#include <moveit/moveit_cpp/moveit_cpp.h>
#include <moveit/moveit_cpp/planning_component.h>


// This class stores an API to communicating with the robot at the high level. It is responsible for processing requests from outside entities and executing accordingly
// For now it just contains a few basic functions to move the robot to a position or a pose. More functions will be added as the project progresses
// Eventually, This class needs to initialize and start an action server that allows the command of high level tasks from outside entities (Global Planner)
// Non exhaustive list of possible tasks:
// --> Move to a position with progress and completion feedback
// --> Move to a pose with progress and completion feedback
// --> Move to a saved state with progress and completion feedback
// --> Move to insert blower hose into a cavity object and orient it upwards or downward on demand
// --> Move to aim at a point with progress and completion feedback
// --> (TENTATIVE) Servoing to aim the stream of insulation to a specific point of a cavity cluster object 
class ArmUtils {
    public:
        //constructor
        ArmUtils(ros::NodeHandle &nh_);

        //destructor
        ~ArmUtils();

        //member functions
        moveit::core::RobotStatePtr getCurrentState();
        void deleteAllMarkers();
        void PublishCavityObjectToScene(const int& cavity_id,const Eigen::VectorXd& dimensions, const Eigen::Vector3d& position, const Eigen::Quaterniond& orientation,const Eigen::Vector3d hole_position);
        void removeCavityObjectFromScene(const int& cavity_id);
        void goToPosition(const Eigen::Vector3d& position);
        void goToPose(const Eigen::Vector3d& position, const Eigen::Quaterniond& orientation);
        void aimAtPoint(const Eigen::Vector3d& point);

        //public member variables
        moveit_visual_tools::MoveItVisualTools visual_tools;
        std::shared_ptr<moveit_cpp::PlanningComponent> planning_components_ptr;
        const moveit::core::JointModelGroup* joint_model_group_ptr;
        std::shared_ptr<moveit_cpp::MoveItCpp> moveit_cpp_ptr;

    private:
        const std::string PLANNING_GROUP;
        moveit::core::RobotModelConstPtr robot_model_ptr;
        moveit::planning_interface::PlanningSceneInterface planning_scene_interface;
        moveit_msgs::CollisionObject cavity;
        ros::AsyncSpinner spinner;
};

#endif