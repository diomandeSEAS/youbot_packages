#!/usr/bin/env python
import numpy as np
import rospy
import signal
import sys
from std_msgs.msg import String
from tf2_msgs.msg import TFMessage
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from pydrake.systems.controllers import DiscreteTimeLinearQuadraticRegulator as DLQR
from pydrake.systems.controllers import LinearQuadraticRegulator as LQR
from pydrake.all import RotationMatrix,RollPitchYaw,PiecewisePolynomial
import matplotlib.pyplot as plt
from actionlib import SimpleActionClient
from youbot_local_trajectory_planner.msg import BaseTrajectoryFollowingAction, BaseTrajectoryFollowingFeedback, BaseTrajectoryFollowingResult, BaseTrajectoryFollowingGoal
from geometry_msgs.msg import Point
import time
from actionlib_msgs.msg import GoalStatus
import yaml
import tf2_ros
from youbot_local_trajectory_planner.msg import TargetBase


class BaseTrajectoryProcessingClient:
    def __init__(self):
        
        self.N = 100
        self.node = rospy.init_node('BaseTrajectoryProcessingClient')
        rospy.loginfo("Starting Client...")
        self.client = SimpleActionClient('BaseTrajectoryFollowerServer', BaseTrajectoryFollowingAction)
        self.client.wait_for_server()
        rospy.loginfo("Base Trajectory Processing Client connected")

        self.tfBuffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tfBuffer)

        self.odom_to_body_tf = np.eye(4)                    #transformation matrix from odom to body frame calculated in odometry callback
        self.odom_to_body_state = np.zeros(6)               #estimated state of the robot wrt to odom frame calculated in odometry callback
        self.odom_to_body_R = np.eye(3)   
        self.map_to_odom_tf = np.eye(4)                     #transformation matrix from map to odom frame calculated in map callback
        self.map_to_odom_state = np.zeros(6)                #estimated state of the robot wrt to map frame calculated in map callback
        self.map_to_odom_R = np.eye(3)
        self.InsulationDeliveryGoal = np.zeros(3)
        
        self.odom_sub = rospy.Subscriber('/odom', Odometry, self.OdomCallback)
        # self.tf_sub = rospy.Subscriber('/tf', TFMessage, self.TFcallback)
        self.insulation_goal_sub = rospy.Subscriber('/target_base', TargetBase, self.getNextInsulationGoal,queue_size=1)
        self.vertical_cavity_sub = rospy.Subscriber('/vertical_cavity_goal',Point,self.VerticalCavityGoalCallback,queue_size=1)
        self.last_feedback = 0
        
    #Callback function for the odometry subscriber
    #This function updates the state of the robot wrt to the odom frame
    def OdomCallback(self, data):
        orientation = data.pose.pose.orientation
        position = data.pose.pose.position
        linear_velocity = data.twist.twist.linear
        angular_velocity = data.twist.twist.angular
        
        self.odom_to_body_tf = msg_to_tf(position, orientation)
        self.odom_to_body_state[0] = self.odom_to_body_tf[0, 3]
        self.odom_to_body_state[1] = self.odom_to_body_tf[1, 3]
        self.odom_to_body_state[2] = quaternion_to_yaw(orientation)
        self.odom_to_body_state[3] = linear_velocity.x
        self.odom_to_body_state[4] = linear_velocity.y
        self.odom_to_body_state[5] = angular_velocity.z 
        self.odom_to_body_R = self.odom_to_body_tf[:3, :3]

    #Callback function for the map subscriber
    #This function updates the state of the robot wrt to the map frame
    def TFcallback(self, event):
        try:
            map_to_odom_tf = self.tfBuffer.lookup_transform("odom", "map", rospy.Time(0))

            self.map_to_odom_state[0] = self.map_to_odom_tf.transform.translation.x
            self.map_to_odom_state[1] = self.map_to_odom_tf.transform.translation.y
            orientation = self.map_to_odom_tf.transform.rotation
            self.map_to_odom_state[2] = quaternion_to_yaw(orientation)
            self.map_to_odom_R = self.transform_to_rotation_matrix(self.map_to_odom_tf.transform)

            map_to_odom_R = self.transform_to_rotation_matrix(map_to_odom_tf.transform)
            self.map_to_odom_state[3:6] = map_to_odom_R @ self.odom_to_odom_state[3:6]


        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as ex:
            rospy.logwarn("Transform not available: %s", ex)
    
    def VerticalCavityGoalCallback(self,data):
        CavityGoal = np.zeros(3)
        CavityGoal[0] = data.x
        CavityGoal[1] = data.y
        CavityGoal[2] = data.z
        speed = 0.2
        dist = np.linalg.norm(CavityGoal - self.odom_to_body_state[0:3])
        self.GoTo(CavityGoal,dist/speed)

    def feedback_cb(self,feedback):
        self.last_feedback = feedback.progress  
    
    def get_tag_pose_from_yaml(self, tag_id, yaml_file='/home/bricediomande/youbot_sim/src/brice_scripts/src/tags.yaml'):
        try:
            with open(yaml_file, 'r') as file:
                data = yaml.safe_load(file)
                
            # Navigate through the nested structure
            bodies = data.get('bodies', [])
            if not bodies:
                rospy.logwarn("No bodies found in YAML file.")
                return None, None
            
            home = bodies[0].get('home', {})
            tags = home.get('tags', [])
            
            # Find the tag with matching ID
            for tag in tags:
                if tag['id'] == tag_id:
                    pose = tag['pose']
                    position = pose['position']
                    rotation = pose['rotation']
                    return position, rotation
                    
            rospy.logwarn(f"Tag ID {tag_id} not found in YAML file.")
            return None, None
        
        except Exception as e:
            rospy.logerr(f"Error reading YAML file: {e}")
            return None, None
    
    def GoToTag(self,ID,time):
        position, orientation = self.get_tag_pose_from_yaml(ID)
        if position is not None and orientation is not None:
            target_position_x = position['x']
            target_position_y = position['y']
            target_position = np.array([target_position_x, target_position_y])
            # Rotate x and y around the z-axis of the origin frame by 180 degrees
            rotation_180 = np.array([[-1, 0],
                                     [0, -1]])
            rotated_position_180 = rotation_180 @ target_position
            target_position_x, target_position_y = rotated_position_180
            direction = np.array([target_position_x, target_position_y]) - np.array([0, 0])
            distance = np.linalg.norm(direction)
            direction = direction / distance  # Normalize the direction vector
            intermediate_position = direction * (distance - 1.5)
            target_position_x, target_position_y = intermediate_position
            theta = np.arctan2(target_position_y, target_position_x)
            
            self.GoTo([target_position_x, target_position_y, theta], time)
   
    def GoTo(self,endGoal,duration,VelocityProfile="cubic"):
        goal = BaseTrajectoryFollowingGoal()

        target_x = endGoal[0]
        target_y = endGoal[1]
        target_theta = endGoal[2]

        t = np.linspace(0, duration, self.N)
        s, ds_dt = self.getVelocityProfileScalingFactor(VelocityProfile, duration)

        x = self.odom_to_body_state[0] + s * (target_x - self.odom_to_body_state[0])
        y = self.odom_to_body_state[1] + s * (target_y - self.odom_to_body_state[1])
        theta = self.odom_to_body_state[2] + s * (target_theta - self.odom_to_body_state[2])

        xdot = (target_x - self.odom_to_body_state[0]) * ds_dt
        ydot = (target_y - self.odom_to_body_state[1]) * ds_dt
        thetadot = (target_theta - self.odom_to_body_state[2]) * ds_dt

        pos_list, vel_list, time_list = [], [], []
        for i in range(self.N):
            pos_list.append(Point(x=x[i], y=y[i], z=theta[i]))
            vel_list.append(Point(x=xdot[i], y=ydot[i], z=thetadot[i]))
            time_list.append(t[i])

        goal.states, goal.velocities, goal.times = pos_list, vel_list, time_list
        self.client.send_goal(goal)
        self.client.wait_for_result(rospy.Duration.from_sec(duration + 1))
        return self.client.get_result()


    def MoveBy(self, displacement, duration, VelocityProfile="cubic"):
        goal = BaseTrajectoryFollowingGoal()
        displacement_odom = np.zeros(3)
        displacement_odom[:2] = displacement[:2]@self.map_to_odom_R[:2,:2]
        displacement_odom[2] = displacement[2]
        t = np.linspace(0, duration, self.N)
        s, ds_dt = self.getVelocityProfileScalingFactor(VelocityProfile, duration)
        x = self.odom_to_body_state[0] + s * displacement_odom[0]
        y = self.odom_to_body_state[1] + s * displacement_odom[1]
        theta = self.odom_to_body_state[2] + s * displacement_odom[2]
        xdot = displacement_odom[0] * ds_dt
        ydot = displacement_odom[1] * ds_dt
        thetadot = displacement_odom[2] * ds_dt
        pos_list, vel_list, time_list = [], [], []
        for i in range(self.N):
            pos_list.append(Point(x=x[i], y=y[i], z=theta[i]))
            vel_list.append(Point(x=xdot[i], y=ydot[i], z=thetadot[i]))
            time_list.append(t[i])
        goal.states, goal.velocities, goal.times = pos_list, vel_list, time_list
        self.client.send_goal(goal)
        self.client.wait_for_result(rospy.Duration.from_sec(duration + 1))
        return self.client.get_result()


    
    #endGoal is the desired position of the insulation centroid wrt to the body frame
    def GoToInsulationDeliveryGoal(self):
        distance = np.linalg.norm(self.InsulationDeliveryGoal[:2]) + 1e-8
        speed = 0.5
        duration = max(distance/speed*1.5,0.5)
        rospy.loginfo(f"Going to Insulation Delivery Goal: {self.InsulationDeliveryGoal} with duration: {duration}")
        self.MoveBy(self.InsulationDeliveryGoal,duration,"cubic")
    
    def ExecuteLimitCycle(self,start_pos,cycle_time,amplitude):
        rospy.loginfo("Executing Limit Cycle...Enter 's' to stop the limit cycle")
        goal = BaseTrajectoryFollowingGoal()
        
        curr_state = self.odom_to_body_state
        curr_pos = curr_state[0:3]
        
        t = np.linspace(0,cycle_time,self.N)
        x = amplitude*np.sin(2*np.pi*t/cycle_time)
        y = start_pos[1]*np.ones(self.N)
        theta = np.zeros(self.N)
        xdot = np.gradient(x,t)
        ydot = np.gradient(y,t)
        thetadot = np.gradient(theta,t)
        
        pos_list = []
        vel_list = []
        time_list = []
        
        for i in range(self.N):
            pos_list.append(Point(x=x[i],y=y[i],z=theta[i]))
            vel_list.append(Point(x = xdot[i],y=ydot[i],z=thetadot[i]))
            time_list.append(t[i])
        
        goal.states = pos_list
        goal.velocities = vel_list
        goal.times = time_list
        
        dist = np.linalg.norm(curr_pos - start_pos)
        if dist > 0.075:
            self.GoTo(start_pos.flatten(),dist/0.3)

        userInput = ''    
        while userInput == '':
            self.client.send_goal(goal,feedback_cb=self.feedback_cb)
            if self.last_feedback > 97:
                rospy.loginfo("Feedback is close to 100, resending the goal")
                self.client.send_goal(goal)
            userInput = input("Enter 's' to stop the limit cycle: ")
                
            
        self.client.cancel_goal()
        
    def stopRobot(self):
        self.client.cancel_goal()
        self.client.wait_for_result()
        rospy.loginfo("Robot Stopped")
    
    def returnToStart(self,duration):
        self.GoTo([0,0,0],duration,"cubic")

    def getNextInsulationGoal(self, data):
        InsulationDeliveryGoalDistance = np.zeros(3)
        if np.linalg.norm([data.delta.x, data.delta.y, data.delta.z]) < 0.01 or np.linalg.norm([data.delta.x, data.delta.y, data.delta.z]) > 1:
            InsulationDeliveryGoalDistance[0] = 0
            InsulationDeliveryGoalDistance[1] = 0
            InsulationDeliveryGoalDistance[2] = 0
        else:
            InsulationDeliveryGoalDistance[0] = data.delta.x
            InsulationDeliveryGoalDistance[1] = data.delta.y
            InsulationDeliveryGoalDistance[2] = 0.0

        self.InsulationDeliveryGoal = InsulationDeliveryGoalDistance
        self.GoToInsulationDeliveryGoal()

    def getVelocityProfileScalingFactor(self, velocityProfileName, duration):
        t = np.linspace(0, duration, self.N)
        if velocityProfileName.lower() == "cubic":
            tau = t / duration
            s = 3 * tau**2 - 2 * tau**3
            ds_dt = (6 / duration) * tau - (6 / duration) * tau**2
        elif velocityProfileName.lower() == "quintic":
            tau = t / duration
            s = 10 * tau**3 - 15 * tau**4 + 6 * tau**5
            ds_dt = (30 * t**2 / duration**3) - (60 * t**3 / duration**4) + (30 * t**4 / duration**5)
        elif velocityProfileName.lower() == "linear":
            s = t / duration
            ds_dt = np.ones_like(t) / duration
        else:
            raise ValueError("Unknown velocity profile: " + velocityProfileName)
        
        return s, ds_dt
         


#---------------------------------Helper Functions---------------------------------#
def invert_tf(tf):
    tf_inv = np.eye(4)
    tf_inv[:3, :3] = tf[:3, :3].T
    tf_inv[:3, 3] = -tf[:3, :3].T @ tf[:3, 3]
    return tf_inv
def quaternion_to_yaw(q):
    return np.arctan2(2 * (q.z * q.w + q.x * q.y), 1 - 2 * (q.y ** 2 + q.z ** 2))
def quaternion_to_rotation_matrix(q):
    q0 = q.w
    q1 = q.x
    q2 = q.y
    q3 = q.z
    rot_matrix = np.array([[1 - 2 * q2 ** 2 - 2 * q3 ** 2, 2 * q1 * q2 - 2 * q0 * q3, 2 * q1 * q3 + 2 * q0 * q2],
                           [2 * q1 * q2 + 2 * q0 * q3, 1 - 2 * q1 ** 2 - 2 * q3 ** 2, 2 * q2 * q3 - 2 * q0 * q1],
                           [2 * q1 * q3 - 2 * q0 * q2, 2 * q2 * q3 + 2 * q0 * q1, 1 - 2 * q1 ** 2 - 2 * q2 ** 2]])
    return rot_matrix
def msg_to_tf(position, orientation):
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = quaternion_to_rotation_matrix(orientation)
    transformation_matrix[:3, 3] = [position.x, position.y, position.z]
    return transformation_matrix
def alarm_handler(signum, frame):
    raise TimeoutError
def input_with_timeout(prompt, timeout=3):
    """Prompt user for input. Raise TimeoutError if no input is given within `timeout` seconds."""
    signal.signal(signal.SIGALRM, alarm_handler)
    signal.alarm(timeout)
    try:
        answer = input(prompt)
        signal.alarm(0)  # Cancel the alarm
        return answer
    except TimeoutError:
        return None


if __name__ == '__main__':
    processor = BaseTrajectoryProcessingClient()
    # rospy.spin()
    # while not rospy.is_shutdown():
    #     continue




    while not rospy.is_shutdown():
        print("Choose an action:")
        print(f"0. Go to next insulation goal. Currently at {processor.InsulationDeliveryGoal}")
        print("1. Execute Limit Cycle (start_pos_x start_pos_y cycle_time amplitude)")
        print("2. Go To (end_goal_x end_goal_y end_goal_theta duration)")
        print("3. Move by (distance_x distance_y distance_theta duration)")
        print("4. Go To Tag (tag_id duration)")
        print("5. Return to Start (duration)")
        print("6. Stop Robot")
        print("7. Exit")
        
        choice = input_with_timeout("Enter your choice (timeout=2s): ", timeout=200)
        if choice == None:
            continue
        elif choice == '1':
            args = input("Enter start position, cycle time, and amplitude (e.g., 0 0 10 1): ").split()
            start_pos = np.array([float(args[0]), float(args[1]),0])
            cycle_time = float(args[2])
            amplitude = float(args[3])
            processor.ExecuteLimitCycle(start_pos, cycle_time, amplitude)
        elif choice == '2':
            args = input("Enter end goal and duration (e.g., 1 1 0 10): ").split()
            end_goal = np.array([float(args[0]), float(args[1]), float(args[2])])
            duration = float(args[3])
            processor.GoTo(end_goal, duration)
        elif choice == '3':
            args = input("Enter end goal and duration (e.g., 1 1 0 10): ").split()
            end_goal = np.array([float(args[0]), float(args[1]), float(args[2])])
            duration = float(args[3])
            processor.MoveBy(end_goal, duration)
        elif choice == '4':
            args = input("Enter tag ID and duration (e.g., 1 10): ").split()
            tag_id = int(args[0])
            duration = float(args[1])
            processor.GoToTag(tag_id, duration)
        elif choice == '5':
            duration = float(input("Enter duration (e.g., 10): "))
            processor.returnToStart(duration)
        elif choice == '6':
            processor.stopRobot()
        elif choice == '7':
            break
        elif choice == '0':
            processor.GoToInsulationDeliveryGoal()
        else:
            print("Invalid choice. Please try again.")


    
        
        
        
        