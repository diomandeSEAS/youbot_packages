#include <ros/package.h>
#include <string>
#include <ros/ros.h>
#include <kdl/jntarray.hpp>
#include <Eigen/Dense>
#include <vector>
#include <cmath>
#include <limits>
#include <typeinfo>
#include <ros/ros.h>
#include <actionlib/client/simple_action_client.h>
#include <actionlib/client/terminal_state.h>
#include <control_msgs/FollowJointTrajectoryAction.h>
#include <sensor_msgs/JointState.h>
#include <tf2_msgs/TFMessage.h>
#include <tf_conversions/tf_kdl.h>
#include <signal.h>
#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/planning_scene_interface/planning_scene_interface.h>
#include <moveit_msgs/DisplayRobotState.h>
#include <moveit_msgs/DisplayTrajectory.h>
#include <moveit_msgs/AttachedCollisionObject.h>
#include <moveit_msgs/CollisionObject.h>
#include <moveit_visual_tools/moveit_visual_tools.h>
#include <tf2_eigen/tf2_eigen.h>
#include <youbot_arm_moveit_config/TargetArm.h>


//tau = 1 rotation in radiants
const double tau = 2 * M_PI;

class ArmControlClient {
public:
    // Constructor
    ArmControlClient(const std::string& node_name)
        : nh_(node_name), // NodeHandle with explicit name
          visual_tools("base_footprint"),  // Initialize MoveItVisualTools
          targetSub(nh_.subscribe("/target_arm", 10,&ArmControlClient::TargetCallback,this)),
          start_arm(false)
    {

        // Initialize Moveit components
        move_group_interface = std::make_unique<moveit::planning_interface::MoveGroupInterface>("arm_1");
        planning_scene_interface = std::make_unique<moveit::planning_interface::PlanningSceneInterface>();
        joint_model_group = move_group_interface->getCurrentState()->getJointModelGroup("arm_1");

        move_group_interface->setMaxAccelerationScalingFactor(1.0);
        move_group_interface->setMaxVelocityScalingFactor(1.0);
        move_group_interface->setPlanningTime(1.0);
        ROS_INFO("%s initialized.", node_name.c_str());


        //Output important information about robot model:
        std::string planning_reference_frame = move_group_interface->getPlanningFrame();
        std::string pose_reference_frame = move_group_interface->getPoseReferenceFrame();
        std::string end_effector_link = move_group_interface->getEndEffectorLink();
        std::string group_name = move_group_interface->getName();
        std::vector<std::string> joint_names = move_group_interface->getVariableNames();

        ROS_INFO("Planning reference frame: %s", planning_reference_frame.c_str());
        ROS_INFO("Pose reference frame: %s", pose_reference_frame.c_str());
        ROS_INFO("End effector link: %s", end_effector_link.c_str());
        ROS_INFO("Group name: %s", group_name.c_str());
        ROS_INFO("Joint names:");
        for (const auto& joint_name : joint_names) {
            ROS_INFO("  %s", joint_name.c_str());
        }

    }

    // Destructor
    ~ArmControlClient() {
    }


    // Aim at a certain point defined in the base_link frame
    bool AimAtGroundPoint(const Eigen::Vector3d& point) {

        geometry_msgs::Pose end_effector_pose = move_group_interface->getCurrentPose("arm_link_4").pose;
        Eigen::Affine3d base_T_EE;
        tf2::fromMsg(end_effector_pose, base_T_EE);
        Eigen::Vector3d base_p_EE = base_T_EE.translation();
        Eigen::Matrix3d base_R_EE = base_T_EE.rotation();

        geometry_msgs::Pose arm_base_pose = move_group_interface->getCurrentPose("arm_link_0").pose;
        Eigen::Affine3d base_T_A0;
        tf2::fromMsg(arm_base_pose, base_T_A0);
        Eigen::Vector3d base_p_A0 = base_T_A0.translation();
        Eigen::Matrix3d base_R_A0 = base_T_A0.rotation();

        Eigen::Vector3d yaw_vector = (point - base_p_A0);
        yaw_vector.z() = 0.0;
        yaw_vector.normalize();
        double yaw = atan2(yaw_vector.y(), yaw_vector.x());

        Eigen::Matrix3d base_R_A0_yaw = Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()).toRotationMatrix();
        Eigen::Affine3d base_T_A0_yaw = Eigen::Affine3d::Identity();
        base_T_A0_yaw.translation() = base_p_A0;
        base_T_A0_yaw.linear() = base_R_A0_yaw;

        Eigen::Affine3d A0_T_EE = base_T_A0.inverse() * base_T_EE;
        Eigen::Affine3d base_T_EE_yaw = base_T_A0_yaw * A0_T_EE;

        Eigen::Vector3d pitch_vector = (point - base_T_EE_yaw.translation());
        pitch_vector.y() = 0.0;
        pitch_vector.normalize();
        double pitch = atan2(pitch_vector.z(), pitch_vector.x());

        Eigen::Matrix3d base_R_EE_yaw_pitch = Eigen::AngleAxisd(-pitch - M_PI_2, Eigen::Vector3d::UnitY()).toRotationMatrix() * base_T_EE_yaw.rotation();
        Eigen::Vector3d base_p_EE_yaw_pitch = base_T_EE_yaw.translation();

        Eigen::Vector3d end_effector_position = base_p_EE_yaw_pitch;
        Eigen::Quaterniond end_effector_orientation(base_R_EE_yaw_pitch);
        
        geometry_msgs::Pose target_pose_msg;
        target_pose_msg.position.x = end_effector_position.x();
        target_pose_msg.position.y = end_effector_position.y();
        target_pose_msg.position.z = end_effector_position.z();
        target_pose_msg.orientation.x = end_effector_orientation.x();
        target_pose_msg.orientation.y = end_effector_orientation.y();
        target_pose_msg.orientation.z = end_effector_orientation.z();
        target_pose_msg.orientation.w = end_effector_orientation.w();

        // Publish the target pose
        visual_tools.publishAxisLabeled(target_pose_msg , "target_pose");
        visual_tools.publishText(Eigen::Isometry3d::Identity(), "Goal Pose", rviz_visual_tools::WHITE, rviz_visual_tools::XLARGE);
        visual_tools.trigger();

        move_group_interface->setPoseTarget(target_pose_msg);
        move_group_interface->setPlanningTime(2.0);
        move_group_interface->setGoalTolerance(0.02);

        // Set constraints for the robot to reach the target pose
        std::vector<double> group_variable_values;
        move_group_interface->getCurrentState()->copyJointGroupPositions(move_group_interface->getCurrentState()->getRobotModel()->getJointModelGroup(move_group_interface->getName()), group_variable_values);

        moveit_msgs::JointConstraint joint_constraint_2;
        joint_constraint_2.joint_name = "arm_joint_2";
        joint_constraint_2.position = group_variable_values[1];
        joint_constraint_2.tolerance_above = 0.05;
        joint_constraint_2.tolerance_below = 0.05;
        joint_constraint_2.weight = 1.0;

        moveit_msgs::JointConstraint joint_constraint_3;
        joint_constraint_3.joint_name = "arm_joint_3";
        joint_constraint_3.position = group_variable_values[2];
        joint_constraint_3.tolerance_above = 0.05;
        joint_constraint_3.tolerance_below = 0.05;
        joint_constraint_3.weight = 1.0;

        moveit_msgs::Constraints joint_goal_constraints;
        joint_goal_constraints.joint_constraints.push_back(joint_constraint_2);
        joint_goal_constraints.joint_constraints.push_back(joint_constraint_3);
        move_group_interface->setPathConstraints(joint_goal_constraints);

        // Plan the motion
        moveit::planning_interface::MoveGroupInterface::Plan my_plan;
        bool success = (move_group_interface->plan(my_plan) == moveit::core::MoveItErrorCode::SUCCESS);
        if (success) {
            ROS_INFO("Planning successful.");
            // Execute the motion
            moveit::core::MoveItErrorCode execution_result = move_group_interface->execute(my_plan);
            if (execution_result == moveit::core::MoveItErrorCode::SUCCESS) {
                ROS_INFO("Motion execution successful.");
            } else {
                ROS_ERROR("Motion execution failed.");
            }
        } else {
            ROS_ERROR("Planning failed.");
        }
        return success;
    }

    // GoToPose method
    bool MoveEEToPose(const Eigen::Vector3d& position, const Eigen::Quaterniond& orientation) {
        geometry_msgs::Pose pose_msg;
        pose_msg.position.x = position.x() + 1e-6;
        pose_msg.position.y = position.y() + 1e-6;
        pose_msg.position.z = position.z() + 1e-6;
        pose_msg.orientation.x = orientation.x() + 1e-6;
        pose_msg.orientation.y = orientation.y() + 1e-6;
        pose_msg.orientation.z = orientation.z() + 1e-6;
        pose_msg.orientation.w = orientation.w()+ 1e-6;

        moveit_msgs::OrientationConstraint ocm;
        ocm.link_name = "arm_link_4";
        ocm.header.frame_id = "base_footprint";
        ocm.orientation = pose_msg.orientation;
        ocm.absolute_x_axis_tolerance = 0.03;
        ocm.absolute_y_axis_tolerance = 0.03;
        ocm.absolute_z_axis_tolerance = 0.12;
        ocm.weight = 1.0;
        moveit_msgs::Constraints constraints;
        constraints.orientation_constraints.push_back(ocm);
        
        
        // Publish the target pose
        visual_tools.deleteAllMarkers();
        visual_tools.publishAxisLabeled(pose_msg, "target_pose");
        visual_tools.publishText(Eigen::Isometry3d::Identity(), "Goal Pose", rviz_visual_tools::WHITE, rviz_visual_tools::XLARGE);
        visual_tools.trigger();
        
        move_group_interface->setGoalTolerance(0.02);
        move_group_interface->setPathConstraints(constraints);
        move_group_interface->setPoseTarget(pose_msg);

        // Plan the motion
        moveit::planning_interface::MoveGroupInterface::Plan my_plan;
        bool success = (move_group_interface->plan(my_plan) == moveit::core::MoveItErrorCode::SUCCESS);
        if (success) {
            ROS_INFO("Planning successful.");
            // Execute the motion
            moveit::core::MoveItErrorCode execution_result = move_group_interface->execute(my_plan);
            if (execution_result == moveit::core::MoveItErrorCode::SUCCESS) {
                ROS_INFO("Motion execution successful.");
            } else {
                ROS_ERROR("Motion execution failed.");
            }
        } else {
            ROS_ERROR("Planning failed.");
        }
        return success;
    }

    bool MoveToSavedState(const std::string& state_name) {
        move_group_interface->clearPathConstraints();
        move_group_interface->setNamedTarget(state_name);
        move_group_interface->setGoalTolerance(0.02);
        moveit::planning_interface::MoveGroupInterface::Plan my_plan;
        bool success = (move_group_interface->plan(my_plan) == moveit::core::MoveItErrorCode::SUCCESS);
        if (success) {
            ROS_INFO("Planning successful.");
            // Execute the motion
            moveit::core::MoveItErrorCode execution_result = move_group_interface->execute(my_plan);
            if (execution_result == moveit::core::MoveItErrorCode::SUCCESS) {
                ROS_INFO("Motion execution successful.");
            } else {
                ROS_ERROR("Motion execution failed.");
            }
        } else {
            ROS_ERROR("Planning failed.");
        }
        return success;
    }

    bool RotateYawBy(const float& angle){
        ROS_INFO("Rotating Arm by Yaw: %f", target);
        move_group_interface->clearPathConstraints();
        move_group_interface->setStartStateToCurrentState();
        moveit::core::RobotStatePtr current_state = move_group_interface->getCurrentState();
        std::vector<double> joint_group_positions;
        current_state->copyJointGroupPositions(joint_model_group, joint_group_positions);
        joint_group_positions[0]+=angle;
        move_group_interface->setJointValueTarget(joint_group_positions);
        move_group_interface->setGoalTolerance(0.005);
        // Plan the motion
        moveit::planning_interface::MoveGroupInterface::Plan my_plan;
        bool success = (move_group_interface->plan(my_plan) == moveit::core::MoveItErrorCode::SUCCESS);
        if (success) {
            ROS_INFO("Planning successful.");
            // Execute the motion
            moveit::core::MoveItErrorCode execution_result = move_group_interface->execute(my_plan);
            if (execution_result == moveit::core::MoveItErrorCode::SUCCESS) {
                ROS_INFO("Motion execution successful.");
            } else {
                ROS_ERROR("Motion execution failed.");
            }
        } else {
            ROS_ERROR("Planning failed.");
        }
        return success;
    }

    Eigen::Affine3d GetHosePose(){
        Eigen::Affine3d end_effector_state;
        tf2::fromMsg(move_group_interface->getCurrentPose().pose, end_effector_state);
        return end_effector_state;
    }

    void TargetCallback(const youbot_arm_moveit_config::TargetArm::ConstPtr& msg){
        double tmp = abs(msg->angle);
        if (tmp <0.0175 || tmp > 0.785){target = 0;}
        else {target = msg->angle;}

        if (!start_arm){
            return;
        }

        ROS_INFO("trying to move the arm");
        if (abs(target) > 0.01){
            if (!RotateYawBy(target)) {
                ROS_ERROR("Failed to rotate yaw");
            }
        }

        
    }

    double target;
    bool start_arm;


private:
    ros::NodeHandle nh_;
    std::unique_ptr<moveit::planning_interface::MoveGroupInterface> move_group_interface;
    std::unique_ptr<moveit::planning_interface::PlanningSceneInterface> planning_scene_interface;
    const moveit::core::JointModelGroup* joint_model_group;
    moveit_visual_tools::MoveItVisualTools visual_tools;
    ros::Subscriber targetSub;
};

int main(int argc, char** argv) {
    // ROS initialization
    ros::init(argc, argv, "ArmControlClient");
    // Create the client with a named NodeHandle
    ros::AsyncSpinner spinner(3);  // Two threads for spinning
    spinner.start();  // Start the async spinner
    ArmControlClient client("ArmControlClient");

    std::string state = "start";
    if (!client.MoveToSavedState(state)) {
        ROS_ERROR("Failed to move to state: %s", state.c_str());
    }

    client.start_arm = true;

    ros::waitForShutdown();  // Wait for shutdown to stop spinning


    // Example pose
    // Eigen::Vector3d position(0.3, 0.0, 0.3);
    // Eigen::Quaterniond orientation(1.0, 0.0, 0.0, 0);
    // Send pose
    // client.MoveToSavedState("spraying");
    // client.MoveToSavedState("reach_step_1");
    // client.MoveToSavedState("reach_step_2");
    // client.MoveToSavedState("start");
    // std::string input;
    // std::vector<std::string> states;

    // while (ros::ok()) {
    //     std::cout << "Enter the number corresponding to the desired action: " << std::endl;
    //     std::cout << "1. Aim at a point on the ground" << std::endl;
    //     std::cout << "2. Move end effector to a named target" << std::endl;
    //     std::cout << "3. Rotate yaw by an angle" << std::endl;
    //     std::cout << "4. Move end effector in a direction" << std::endl;
    //     std::cout << "5. Exit" << std::endl;
    //     int action;
    //     std::cin >> action;
    //     if (action == 1) {
    //         std::cout << "Enter the x, y, z coordinates of the point on the ground: " << std::endl;
    //         double x, y, z;
    //         std::cin >> x >> y >> z;
    //         Eigen::Vector3d point(x, y, z);
    //         client.AimAtGroundPoint(point);
    //     } else if (action == 2) {
    //         std::cout << "Enter the name of the target state: " << std::endl;
    //         std::string state;
    //         std::cin >> state;
    //         if (!client.MoveToSavedState(state)) {
    //             ROS_ERROR("Failed to move to state: %s", state.c_str());
    //         }
    //     } else if (action == 3) {
    //         std::cout << "Enter the angular displacement value " << std::endl;
    //         double angle;
    //         std::cin >> angle;
    //         if (!client.RotateYawBy(angle)) {
    //             ROS_ERROR("Failed to rotate yaw");
    //         }
    //     } else if (action == 4) {
    //         std::cout << "Enter the x, y, z coordinates of the direction: " << std::endl;
    //         double x, y, z;
    //         std::cin >> x >> y >> z;
    //         Eigen::Vector3d direction(x, y, z);
    //         Eigen::Affine3d ee_pose = client.GetHosePose();
    //         ee_pose.translation() += direction;
    //         client.MoveEEToPose(ee_pose.translation(), Eigen::Quaterniond(ee_pose.rotation()));
    //     } else if (action == 5) {
    //         break;
    //     } else {
    //         ROS_ERROR("Invalid action.");
    //     }
    // }

}