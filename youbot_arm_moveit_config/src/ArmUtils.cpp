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
        // Constructor
        ArmUtils(const std::string& node_name)
            : nh_(node_name),                                                           // NodeHandle with explicit name
            spinner(1),                                                               // Initialize AsyncSpinner
            PLANNING_GROUP("arm_1"),                                                    // Initialize move_group_name
            robot_model_loader(new robot_model_loader::RobotModelLoader("robot_description")), // Initialize robot_model_loader
            psm(new planning_scene_monitor::PlanningSceneMonitor(robot_model_loader)), // Initialize PlanningSceneMonitor
            planning_scene_interface(),                                               // Initialize PlanningSceneInterface
            visual_tools("arm_link_0"),                                              // Initialize MoveItVisualTools
            move_group(PLANNING_GROUP)                                                 // Initialize MoveGroupInterface
        {
            spinner.start(); // Start the AsyncSpinner

            // Initialize Robot Model and Robot_State from loader
            robot_model = robot_model_loader->getModel();
            robot_state = std::make_shared<robot_state::RobotState>(robot_model);
            joint_model_group = robot_state->getJointModelGroup(PLANNING_GROUP);

            // Planning Scene Interface. This uses the planningSceneMonitor Interface to update the arm context as this node receives new messages:
            //--->SceneMonitor: Monitors changes to the scene (objects added, removed, etc.)
            //--->WorldGeometryMonitor: Not sure what this monitors yet
            //--->StateMonitor: Monitors and stores the state of the robot (JointState)
            //---->PublishingPlanningScene: Publishes the current state of the scene to the planning scene topic
            // Other monitoring capabilities may need to be added by explicitely declaring Suscribers to respective topics
            // Anticipated Monitoring Capabilities;
            //---->CavityObjectClient: fetch a cavity object from the database on demand and store its info in an object
            //---->InsulationBlowingSuscriber: Suscribe to the insulation blowing topic to get the state of a particular cavity object
            // auto tf_buffer = std::make_shared<tf2_ros::Buffer>();
            // tf2_ros::TransformListener tf_listener(*tf_buffer);
            // psm = planning_scene_monitor::PlanningSceneMonitorPtr(new planning_scene_monitor::PlanningSceneMonitor(robot_model_loader, tf_buffer, "planning_scene_monitor"));
            psm->startSceneMonitor();
            psm->startWorldGeometryMonitor();
            psm->startStateMonitor();
            psm->startPublishingPlanningScene(planning_scene_monitor::PlanningSceneMonitor::UPDATE_STATE);

            //planning pipeline: Responsible for finding the plan that the arm must follow to execute the higher level task requested
            // It uses the planning plugin and request planner uploaded to the rosparam server and sepcified in (package)/config/<planner_name>_planning.yaml (e.g. ompl_planning.yaml)
            // The behavior of this object can be easily changed by modifying the planning plugin and request adapter values uploaded to the server or modifying the file directly
            planning_pipeline = planning_pipeline::PlanningPipelinePtr(new planning_pipeline::PlanningPipeline(robot_model, ros::NodeHandle("/move_group/planning_pipelines/ompl"), "planning_plugin", "request_adapters"));

            // Visual tools
            namespace rvt = rviz_visual_tools;
            visual_tools.deleteAllMarkers();


            //Scene storage variables
            cavity = moveit_msgs::CollisionObject();
            cavity.header.frame_id = "arm_link_0";


            ROS_INFO("%s initialized.", node_name.c_str());
        }

        // Destructor
        ~ArmUtils() {
            spinner.stop();
            //No need to use the delete keyword as the shared pointers (or custom object ending w Ptr) will automatically delete the objects when they go out of scope
        }

        moveit::core::RobotStatePtr getCurrentState(){
            std::shared_ptr<moveit::core::RobotState> current_state = std::make_shared<moveit::core::RobotState>(planning_scene_monitor::LockedPlanningSceneRO(psm)->getCurrentState());
            return current_state;
        }


        void deleteAllMarkers(){
            visual_tools.deleteAllMarkers();
        }

        //Not finished yet
        // bool goToSavedState(const std::string& state_name) {
        //     planning_interface::MotionPlanRequest req;
        //     planning_interface::MotionPlanResponse res;
        //     geometry_msgs::PoseStamped pose;
        //     pose.header.frame_id = "arm_link_0";
        // }


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
        void PublishCavityObjectToScene(const int& cavity_id,const Eigen::VectorXd& dimensions, const Eigen::Vector3d& position, const Eigen::Quaterniond& orientation,const Eigen::Vector3d hole_position){
            
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

        
        void removeCavityObjectFromScene(const int& cavity_id){
            moveit_msgs::CollisionObject cavity;
            cavity.id = std::to_string(cavity_id);
            cavity.operation = moveit_msgs::CollisionObject::REMOVE;
            planning_scene_interface.applyCollisionObject(cavity);


        }

        bool aimAtPoint(const Eigen::Vector3d& position){
            planning_interface::MotionPlanRequest req;
            planning_interface::MotionPlanResponse res;
            moveit_msgs::Constraints constraints;
            moveit_msgs::VisibilityConstraint aimingConstraint;
            aimingConstraint.target_radius = 0.20;
            aimingConstraint.target_pose.header.frame_id = "arm_link_0";
            aimingConstraint.target_pose.pose.position.x = position.x();
            aimingConstraint.target_pose.pose.position.y = position.y();
            aimingConstraint.target_pose.pose.position.z = position.z();
            aimingConstraint.cone_sides = 4;
            aimingConstraint.sensor_pose.header.frame_id = "hose_holder_hose_holder_tip_link";
            aimingConstraint.sensor_pose.pose.position.x = 1e-6;
            aimingConstraint.sensor_pose.pose.position.y = 1e-6;
            aimingConstraint.sensor_pose.pose.position.z = 1e-6;
            aimingConstraint.sensor_pose.pose.orientation.x = 1e-6;
            aimingConstraint.sensor_pose.pose.orientation.y = 1e-6;
            aimingConstraint.sensor_pose.pose.orientation.z = 1e-6;
            aimingConstraint.sensor_pose.pose.orientation.w = 1.0;
            aimingConstraint.max_view_angle = 5;
            aimingConstraint.max_range_angle = 10;
            aimingConstraint.sensor_view_direction = moveit_msgs::VisibilityConstraint::SENSOR_X;
            aimingConstraint.weight = 10;

            constraints.visibility_constraints.push_back(aimingConstraint);

            req.group_name = PLANNING_GROUP;
            req.goal_constraints.push_back(constraints);

            if (!generatePlan(req, res)) {
                ROS_ERROR("Failed to compute plan to position.");
                return false;
            }

            visualizeTrajectory(res);
            return executePlan(res);         
            
        }

        bool goToPosition(const Eigen::Vector3d& position){
            planning_interface::MotionPlanRequest req;
            planning_interface::MotionPlanResponse res;
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
            req.group_name = PLANNING_GROUP;
            moveit_msgs::Constraints pose_goal = kinematic_constraints::constructGoalConstraints("hose_holder_hose_holder_tip_link", pose, tolerance_pose, tolerance_angle);
            req.goal_constraints.push_back(pose_goal);

            if (!generatePlan(req, res)) {
                ROS_ERROR("Failed to compute plan to position.");
                return false;
            }

            visualizeTrajectory(res);
            return executePlan(res);
        }


        bool goToPose(const Eigen::Vector3d& position, const Eigen::Quaterniond& orientation){
            planning_interface::MotionPlanRequest req;
            planning_interface::MotionPlanResponse res;
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
            req.group_name = PLANNING_GROUP;
            visual_tools.publishAxis(pose.pose);
            visual_tools.trigger();
            moveit_msgs::Constraints pose_goal = kinematic_constraints::constructGoalConstraints("hose_holder_hose_holder_tip_link", pose, tolerance_pose, tolerance_angle);
            req.goal_constraints.push_back(pose_goal);

            if (!generatePlan(req, res)) {
                ROS_ERROR("Failed to compute plan to pose.");
                ROS_INFO_STREAM("Planning failed with error code: " << res.error_code_);
                return false;
            }

            visualizeTrajectory(res);
            return executePlan(res);    
        }


        bool generatePlan(const planning_interface::MotionPlanRequest& req, planning_interface::MotionPlanResponse& res){
            planning_scene_monitor::LockedPlanningSceneRO lscene(psm);
            planning_pipeline->generatePlan(lscene, req, res);
            return res.error_code_.val == moveit_msgs::MoveItErrorCodes::SUCCESS;
        }

        void visualizeTrajectory(const planning_interface::MotionPlanResponse& res){
            ROS_INFO("Visualizing the trajectory");
            visual_tools.deleteAllMarkers();
            moveit_msgs::MotionPlanResponse response;
            res.getMessage(response);
            moveit_msgs::DisplayTrajectory display_trajectory;
            display_trajectory.trajectory_start = response.trajectory_start;
            display_trajectory.trajectory.push_back(response.trajectory);
            visual_tools.publishTrajectoryLine(res.trajectory_, joint_model_group);
            visual_tools.trigger();
        }

        bool executePlan(const planning_interface::MotionPlanResponse& res){
            ROS_INFO("Executing the plan");
            moveit::planning_interface::MoveGroupInterface::Plan my_plan;
            moveit_msgs::MotionPlanResponse response;
            res.getMessage(response);
            my_plan.planning_time_ = response.planning_time;
            my_plan.start_state_ = response.trajectory_start;
            my_plan.trajectory_ = response.trajectory;
            moveit::core::MoveItErrorCode execution_result = move_group.execute(my_plan);
            if (execution_result == moveit::core::MoveItErrorCode::SUCCESS) {
                ROS_INFO("Motion execution successful.");
                return true;
            } else {
                ROS_ERROR("Motion execution failed.");
                return false;
            }
        }



            
            

        moveit_visual_tools::MoveItVisualTools visual_tools;

        


    private:
        ros::NodeHandle nh_;
        const std::string PLANNING_GROUP;
        robot_model_loader::RobotModelLoaderPtr robot_model_loader;
        planning_scene_monitor::PlanningSceneMonitorPtr psm;
        moveit::planning_interface::PlanningSceneInterface planning_scene_interface;
        robot_model::RobotModelPtr robot_model;
        robot_state::RobotStatePtr robot_state;
        const robot_state::JointModelGroup* joint_model_group;
        planning_pipeline::PlanningPipelinePtr planning_pipeline;
        moveit::planning_interface::MoveGroupInterface move_group;
        moveit_msgs::CollisionObject cavity;
        // Add a cavity object suscriber here and its callback should be PublishCavityObject
        ros::AsyncSpinner spinner; // Async spinner to handle ROS callbacks
        // Add a variable that stores the loaded planner here

};

int main(int argc, char** argv) {
    // ROS initialization
    ros::init(argc, argv, "MoveItArmTrajectoryFollowerClient");

    // Create the client with a named NodeHandle
    ArmUtils client("ArmInterfaceClient");

    // Move to a position
    robot_state::RobotStatePtr current_state = client.getCurrentState();
    Eigen::Affine3d end_effector_state = current_state->getGlobalLinkTransform("hose_holder_hose_holder_tip_link");
    Eigen::Vector3d position = end_effector_state.translation();
    ROS_INFO("Current position: %f, %f, %f", position.x(), position.y(), position.z());
    Eigen::Vector3d new_position = position;
    if (client.goToPosition(new_position)) {
        ROS_INFO("Successfully moved to the new position.");
    } else {
        ROS_ERROR("Failed to move to the new position.");
    }
    client.visual_tools.prompt("Press 'next' in the RvizVisualToolsGui window to continue the demo");
    end_effector_state.translation() = new_position;
    Eigen::Quaterniond orientation(end_effector_state.rotation());
    if (client.goToPose(new_position, orientation)) {
        ROS_INFO("Successfully moved to the new pose.");
    } else {
        ROS_ERROR("Failed to move to the new pose.");
    }
    client.visual_tools.prompt("Press 'next' in the RvizVisualToolsGui window to continue the demo");

    // Aim at a point
    Eigen::Vector3d aim_position = new_position + Eigen::Vector3d(2, 2, 0);
    if (client.aimAtPoint(aim_position)) {
        ROS_INFO("Successfully aimed at the point.");
    } else {
        ROS_ERROR("Failed to aim at the point.");
    }
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