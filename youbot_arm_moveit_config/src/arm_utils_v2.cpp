#include "youbot_arm_moveit_config/arm_utils_v2.h"
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



ArmUtils::ArmUtils(ros::NodeHandle &nh_):                                
        spinner(1),                                          // Initialize AsyncSpinner
        PLANNING_GROUP("arm_1"),                               // Initialize move_group_name
        planning_scene_interface(),                      // Initialize PlanningSceneInterface
        visual_tools("arm_link_0")                      // Initialize MoveItVisualTools
    {
        spinner.start(); // Start the AsyncSpinner

        moveit_cpp::MoveItCpp::Options options(nh_);
        nh_.setParam("move_group/moveit_controller_manager", "moveit_simple_controller_manager/MoveItSimpleControllerManager");
        nh_.setParam("ArmInterfaceClient/moveit_controller_manager", "moveit_simple_controller_manager/MoveItSimpleControllerManager");


        options.planning_pipeline_options.pipeline_names.push_back("/move_group/planning_pipelines/ompl");
        options.planning_pipeline_options.parent_namespace = "/move_group";
        auto moveit_cpp_ptr = std::make_shared<moveit_cpp::MoveItCpp>(options,nh_);
        moveit_cpp_ptr->getPlanningSceneMonitorNonConst()->providePlanningSceneService();
        planning_components_ptr = std::make_shared<moveit_cpp::PlanningComponent>(PLANNING_GROUP, moveit_cpp_ptr);
        robot_model_ptr = moveit_cpp_ptr->getRobotModel();
        joint_model_group_ptr = robot_model_ptr->getJointModelGroup(PLANNING_GROUP);

        // Visual tools
        visual_tools.deleteAllMarkers();

        //Scene storage variables
        cavity = moveit_msgs::CollisionObject();
        cavity.header.frame_id = "arm_link_0";


        ROS_INFO("ArmUtils initialized");
    }

// Destructor
ArmUtils::~ArmUtils() {
    spinner.stop();
    //No need to use the delete keyword as the shared pointers (or custom object ending w Ptr) will automatically delete the objects when they go out of scope
}

void ArmUtils::deleteAllMarkers() {
    visual_tools.deleteAllMarkers();
}

//This function should publish a cavity object to the scene as a registered collision_geometry. It should also contain 4 subframes that condition the motion of the EE within the cavity.
// Subframe 1: Frame that aligns the EE with the cavity hole
// Subframe 2: Frame to insert the blower into the cavity
// Subframe 3: Frame to orient the blower downwards
// Subframe 4: Frame to orient the blower upwards
// Inputs:
// --> cavity_id: The id of the cavity object. Should be a retrievable index from the csvity object db
// --> dimensions: The dimensions of the cavity object. An arrat of form [wall_thickness,depth,witdh,height]
// --> position: The position of the cavity object
// --> orientation: The orientation of the cavity object
// --> hole_position: The position of the hole in the cavity object through which filling should occur
void ArmUtils::PublishCavityObjectToScene(const int& cavity_id,const Eigen::VectorXd& dimensions, const Eigen::Vector3d& position, const Eigen::Quaterniond& orientation,const Eigen::Vector3d hole_position){
    
    cavity.id = std::to_string(cavity_id);

    shape_msgs::SolidPrimitive cavity_wall_left;
    cavity_wall_left.type = shape_msgs::SolidPrimitive::BOX;
    cavity_wall_left.dimensions.resize(3);
    cavity_wall_left.dimensions[0] = dimensions[1];
    cavity_wall_left.dimensions[1] = dimensions[0];
    cavity_wall_left.dimensions[2] = dimensions[3];
    geometry_msgs::Pose cavity_wall_left_pose;
    cavity_wall_left_pose.position.x = position.x();
    cavity_wall_left_pose.position.y = position.y() + (dimensions[1]/2 + dimensions[0]/2);
    cavity_wall_left_pose.position.z = position.z();
    cavity_wall_left_pose.orientation.x = orientation.x();
    cavity_wall_left_pose.orientation.y = orientation.y();
    cavity_wall_left_pose.orientation.z = orientation.z();
    cavity_wall_left_pose.orientation.w = orientation.w();

    shape_msgs::SolidPrimitive cavity_wall_right;
    cavity_wall_right.type = shape_msgs::SolidPrimitive::BOX;
    cavity_wall_right.dimensions.resize(3);
    cavity_wall_right.dimensions[0] = dimensions[1];
    cavity_wall_right.dimensions[1] = dimensions[0];
    cavity_wall_right.dimensions[2] = dimensions[3];
    geometry_msgs::Pose cavity_wall_right_pose;
    cavity_wall_right_pose.position.x = position.x();
    cavity_wall_right_pose.position.y = position.y() - (dimensions[1]/2 + dimensions[0]/2);
    cavity_wall_right_pose.position.z = position.z();
    cavity_wall_right_pose.orientation.x = orientation.x();
    cavity_wall_right_pose.orientation.y = orientation.y();
    cavity_wall_right_pose.orientation.z = orientation.z();
    cavity_wall_right_pose.orientation.w = orientation.w();

    shape_msgs::SolidPrimitive cavity_wall_back;
    cavity_wall_back.type = shape_msgs::SolidPrimitive::BOX;
    cavity_wall_back.dimensions.resize(3);
    cavity_wall_back.dimensions[0] = dimensions[0];
    cavity_wall_back.dimensions[1] = dimensions[2];
    cavity_wall_back.dimensions[2] = dimensions[3];
    geometry_msgs::Pose cavity_wall_back_pose;
    cavity_wall_back_pose.position.x = position.x() + dimensions[1]/2;
    cavity_wall_back_pose.position.y = position.y();
    cavity_wall_back_pose.position.z = position.z();
    cavity_wall_back_pose.orientation.x = orientation.x();
    cavity_wall_back_pose.orientation.y = orientation.y();
    cavity_wall_back_pose.orientation.z = orientation.z();
    cavity_wall_back_pose.orientation.w = orientation.w();

    geometry_msgs::Pose cavity_hole_pre;
    cavity_hole_pre.position.x = hole_position.x()-0.05;
    cavity_hole_pre.position.y = hole_position.y();
    cavity_hole_pre.position.z = hole_position.z();
    cavity_hole_pre.orientation.x = orientation.x();
    cavity_hole_pre.orientation.y = orientation.y();
    cavity_hole_pre.orientation.z = orientation.z();
    cavity_hole_pre.orientation.w = orientation.w();
    cavity.subframe_names.push_back("hole");
    cavity.subframe_poses.push_back(cavity_hole_pre);

    geometry_msgs::Pose cavity_hole_post;
    cavity_hole_post.position.x = hole_position.x()+0.05;
    cavity_hole_post.position.y = hole_position.y();
    cavity_hole_post.position.z = hole_position.z();
    cavity_hole_post.orientation.x = orientation.x();
    cavity_hole_post.orientation.y = orientation.y();
    cavity_hole_post.orientation.z = orientation.z();
    cavity_hole_post.orientation.w = orientation.w();
    cavity.subframe_names.push_back("hole");
    cavity.subframe_poses.push_back(cavity_hole_post);


    geometry_msgs::Pose cavity_hole_orient_down;
    cavity_hole_orient_down.position.x = hole_position.x();
    cavity_hole_orient_down.position.y = hole_position.y();
    cavity_hole_orient_down.position.z = hole_position.z();
    Eigen::Quaterniond down_orientation = orientation * Eigen::AngleAxisd(-M_PI / 4, Eigen::Vector3d::UnitY());
    cavity_hole_orient_down.orientation.x = down_orientation.x();
    cavity_hole_orient_down.orientation.y = down_orientation.y();
    cavity_hole_orient_down.orientation.z = down_orientation.z();
    cavity_hole_orient_down.orientation.w = down_orientation.w();
    cavity.subframe_names.push_back("orient_down");
    cavity.subframe_poses.push_back(cavity_hole_orient_down);

    geometry_msgs::Pose cavity_hole_orient_up;
    cavity_hole_orient_up.position.x = hole_position.x();
    cavity_hole_orient_up.position.y = hole_position.y();
    cavity_hole_orient_up.position.z = hole_position.z();
    Eigen::Quaterniond up_orientation = orientation * Eigen::AngleAxisd(M_PI / 4, Eigen::Vector3d::UnitY());
    cavity_hole_orient_up.orientation.x = up_orientation.x();
    cavity_hole_orient_up.orientation.y = up_orientation.y();
    cavity_hole_orient_up.orientation.z = up_orientation.z();
    cavity_hole_orient_up.orientation.w = up_orientation.w();
    cavity.subframe_names.push_back("orient_up");
    cavity.subframe_poses.push_back(cavity_hole_orient_up);


    cavity.subframe_names.push_back("hole_pre");
    cavity.subframe_poses.push_back(cavity_hole_pre);

    cavity.subframe_names.push_back("hole_post");
    cavity.subframe_poses.push_back(cavity_hole_post);

    cavity.subframe_names.push_back("orient_down");
    cavity.subframe_poses.push_back(cavity_hole_orient_down);

    cavity.subframe_names.push_back("orient_up");
    cavity.subframe_poses.push_back(cavity_hole_orient_up);


    cavity.primitives.push_back(cavity_wall_left);
    cavity.primitives.push_back(cavity_wall_right);
    cavity.primitives.push_back(cavity_wall_back);
    cavity.primitive_poses.push_back(cavity_wall_right_pose);
    cavity.primitive_poses.push_back(cavity_wall_left_pose);
    cavity.primitive_poses.push_back(cavity_wall_back_pose);
    cavity.operation = moveit_msgs::CollisionObject::ADD;

    planning_scene_interface.applyCollisionObject(cavity);

}

void ArmUtils::removeCavityObjectFromScene(const int& cavity_id){
    moveit_msgs::CollisionObject cavity;
    cavity.id = std::to_string(cavity_id);
    cavity.operation = moveit_msgs::CollisionObject::REMOVE;
    planning_scene_interface.applyCollisionObject(cavity);

}
    
void ArmUtils::goToPose(const Eigen::Vector3d& position, const Eigen::Quaterniond& orientation){
    geometry_msgs::PoseStamped pose;
    pose.header.frame_id = "arm_link_0";
    pose.pose.position.x = position.x();
    pose.pose.position.y = position.y();
    pose.pose.position.z = position.z();
    pose.pose.orientation.x = orientation.x();
    pose.pose.orientation.y = orientation.y();
    pose.pose.orientation.z = orientation.z();
    pose.pose.orientation.w = orientation.w();
    std::vector<double> tolerance_pose(3, 0.05);
    std::vector<double> tolerance_angle(3, 0.02);
    moveit_msgs::Constraints pose_goal = kinematic_constraints::constructGoalConstraints("hose_holder_hose_holder_tip_link", pose, tolerance_pose, tolerance_angle);
    std::vector<moveit_msgs::Constraints> pose_goals;
    pose_goals.push_back(pose_goal);
    planning_components_ptr->setGoal(pose_goals);
    planning_components_ptr->setStartStateToCurrentState();
    planning_interface::MotionPlanResponse sol = planning_components_ptr->plan();
    if (sol.error_code_.val == moveit_msgs::MoveItErrorCodes::SUCCESS) {
        ROS_INFO("Successfully found a path to pose.");
        // Visualize the goal pose in Rviz
        visual_tools.publishAxisLabeled(pose.pose, "target_pose");
        visual_tools.publishText(Eigen::Isometry3d::Identity(), "Goal Pose", rviz_visual_tools::WHITE, rviz_visual_tools::XLARGE);
        // Visualize the trajectory in Rviz
        visual_tools.publishTrajectoryLine(sol.trajectory_, joint_model_group_ptr);
        visual_tools.trigger();
    } else {
        ROS_INFO_STREAM("Planning failed with error code: " << sol.error_code_);
    }

    // Execute the trajectory
    if (planning_components_ptr->execute()) {
        ROS_INFO("Successfully executed the trajectory.");
    } else {
        ROS_ERROR("Failed to execute the trajectory.");
    }
}

void ArmUtils::goToPosition(const Eigen::Vector3d& position){
    geometry_msgs::PoseStamped pose;
    pose.header.frame_id = "arm_link_0";
    pose.pose.position.x = position.x();
    pose.pose.position.y = position.y();
    pose.pose.position.z = position.z();
    pose.pose.orientation.x = 1e-6;
    pose.pose.orientation.y = 1e-6;
    pose.pose.orientation.z = 1e-6;
    pose.pose.orientation.w = 1.000000;
    visual_tools.publishAxis(pose.pose);
    visual_tools.trigger();
    std::vector<double> tolerance_pose(3, 1e-2);
    std::vector<double> tolerance_angle(3, 0.5);
    moveit_msgs::Constraints pose_goal = kinematic_constraints::constructGoalConstraints("hose_holder_hose_holder_tip_link", pose, tolerance_pose, tolerance_angle);
    planning_components_ptr->setStartStateToCurrentState();
    std::vector<moveit_msgs::Constraints> pose_goals;
    pose_goals.push_back(pose_goal);
    planning_components_ptr->setGoal(pose_goals);
    planning_interface::MotionPlanResponse sol = planning_components_ptr->plan();
    if (sol.error_code_.val == moveit_msgs::MoveItErrorCodes::SUCCESS) {
        ROS_INFO("Successfully found a path to pose.");
        // Visualize the goal pose in Rviz
        visual_tools.publishAxisLabeled(pose.pose, "target_pose");
        visual_tools.publishText(Eigen::Isometry3d::Identity(), "Goal Pose", rviz_visual_tools::WHITE, rviz_visual_tools::XLARGE);
        // Visualize the trajectory in Rviz
        visual_tools.publishTrajectoryLine(sol.trajectory_, joint_model_group_ptr);
        visual_tools.trigger();
    } else {
        ROS_INFO_STREAM("Planning failed with error code: " << sol.error_code_);
    }

    // Execute the trajectory
    if (planning_components_ptr->execute()) {
        ROS_INFO("Successfully executed the trajectory.");
    } else {
        ROS_ERROR("Failed to execute the trajectory.");
    }
}

void ArmUtils::aimAtPoint(const Eigen::Vector3d& point){
    geometry_msgs::PoseStamped pose;
    pose.header.frame_id = "arm_link_0";
    pose.pose.position.x = point.x();
    pose.pose.position.y = point.y();
    pose.pose.position.z = point.z();
    pose.pose.orientation.x = 1e-6;
    pose.pose.orientation.y = 1e-6;
    pose.pose.orientation.z = 1e-6;
    pose.pose.orientation.w = 1.000000;
    visual_tools.publishAxis(pose.pose);
    visual_tools.trigger();
    std::vector<double> tolerance_pose(3, 1e-2);
    std::vector<double> tolerance_angle(3, 0.5);
    moveit_msgs::Constraints pose_goal = kinematic_constraints::constructGoalConstraints("hose_holder_hose_holder_tip_link", pose, tolerance_pose, tolerance_angle);
    planning_components_ptr->setStartStateToCurrentState();
    std::vector<moveit_msgs::Constraints> pose_goals;
    pose_goals.push_back(pose_goal);
    planning_components_ptr->setGoal(pose_goals);
    planning_interface::MotionPlanResponse sol = planning_components_ptr->plan();
    if (sol.error_code_.val == moveit_msgs::MoveItErrorCodes::SUCCESS) {
        ROS_INFO("Successfully found a path to pose.");
        // Visualize the goal pose in Rviz
        visual_tools.publishAxisLabeled(pose.pose, "target_pose");
        visual_tools.publishText(Eigen::Isometry3d::Identity(), "Goal Pose", rviz_visual_tools::WHITE, rviz_visual_tools::XLARGE);
        // Visualize the trajectory in Rviz
        visual_tools.publishTrajectoryLine(sol.trajectory_, joint_model_group_ptr);
        visual_tools.trigger();
    } else {
        ROS_INFO_STREAM("Planning failed with error code: " << sol.error_code_);
    }

    // Execute the trajectory
    if (planning_components_ptr->execute()) {
        ROS_INFO("Successfully executed the trajectory.");
    } else {
        ROS_ERROR("Failed to execute the trajectory.");
    }
}


int main(int argc, char** argv) {
    // ROS initialization
    ros::init(argc, argv, "ArmInterfaceClient");
    ros::NodeHandle nh_("move_group");
    // Create the client with a named NodeHandle
    ArmUtils client(nh_);

    // Move to a position

    Eigen::Affine3d end_effector_state = client.moveit_cpp_ptr->getCurrentState()->getGlobalLinkTransform("hose_holder_hose_holder_tip_link");
    Eigen::Vector3d position = end_effector_state.translation();
    ROS_INFO("Current position: %f, %f, %f", position.x(), position.y(), position.z());
    Eigen::Vector3d new_position = position;
    client.goToPosition(new_position);
    client.visual_tools.prompt("Press 'next' in the RvizVisualToolsGui window to continue the demo");
    end_effector_state.translation() = new_position;
    Eigen::Quaterniond orientation(end_effector_state.rotation());
    client.goToPose(new_position, orientation);
    client.visual_tools.prompt("Press 'next' in the RvizVisualToolsGui window to continue the demo");
    Eigen::Vector3d aim_position = new_position + Eigen::Vector3d(2, 2, 0);
    client.aimAtPoint(aim_position);
    client.visual_tools.prompt("Press 'next' in the RvizVisualToolsGui window to continue the demo");

    // Publish a cavity object to the scene
    Eigen::VectorXd dimensions(4);
    dimensions << 0.1, 0.25, 0.75, 3.0;
    Eigen::Vector3d cavity_position(0.5, 0, 1.5);
    Eigen::Quaterniond cavity_orientation(1, 0, 0, 0);
    Eigen::Vector3d hole_position(-0.25, 0, 1.5);
    client.PublishCavityObjectToScene(1, dimensions, cavity_position, cavity_orientation, hole_position);

    client.visual_tools.prompt("Press 'next' in the RvizVisualToolsGui window to continue the demo");

    // Remove the cavity object from the scene
    client.removeCavityObjectFromScene(1);



    ros::shutdown();
    return 0;
}