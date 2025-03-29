#!/usr/bin/env python
import numpy as np
import rospy
from std_msgs.msg import String
from tf2_msgs.msg import TFMessage
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import control
import matplotlib.pyplot as plt
from scipy.interpolate import CubicHermiteSpline
from actionlib import SimpleActionServer
from youbot_local_trajectory_planner.msg import BaseTrajectoryFollowingAction, BaseTrajectoryFollowingFeedback, BaseTrajectoryFollowingResult
from geometry_msgs.msg import Point
import time
import tf2_ros
import tf.transformations as tf_trans

class BaseTrajectoryFollowerServer:
    
    def __init__(self, dt):
        #Declare class variables
        self.dt = dt                                        #time step for the controller defined in the constructor
        self.traj_d = None                                  #desired trajectory wrt to global frame in the form [x, y, theta, x_dot, y_dot, theta_dot] defined in the constructor
        self.traj_a = np.empty((6,0))                        #actual trajectory wrt to global frame in the form [x, y, theta, x_dot, y_dot, theta_dot]
        self.odom_to_body_tf = np.eye(4)                     #transformation matrix from odom to body frame calculated in odometry callback
        self.odom_to_body_state = np.zeros(6)               #estimated state of the robot wrt to odom frame calculated in odometry callback
        self.odom_to_body_R = np.zeros((3, 3))              #rotation matrix from odom to body frame calculated in odometry callback
        self.map_to_body_tf = np.eye(4)                  #transformation matrix from map to body frame calculated in ground truth callback
        self.map_to_body_state =np.zeros(6)              #actual state of the robot wrt to map frame calculated in ground truth callback
        self.map_to_body_R = np.zeros((3, 3))            #rotation matrix from map to body frame calculated in ground truth callback
        self.x_offset = 0
        self.y_offset = 0
        self.theta_offset = 0
        
        rospy.init_node('BaseTrajectoryFollowerNode', anonymous=True)

        #Declare ROS node and publishers/subscribers
        self.tfBuffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tfBuffer)
        
        #Declare ROS node and publishers/subscribers used for trajectory following
        self.pub = rospy.Publisher('/cmd_vel', Twist,queue_size=10)
        self.subOdometry = rospy.Subscriber('/odom', Odometry, self.OdomsSubCallback)
        # self.subTF = rospy.Subscriber('/tf', TFMessage, self.TFcallback)

        rospy.sleep(1)
        self.set_offsets()
        
        #Declare server 
        rospy.loginfo("Starting Server...")
        self.actionServer = SimpleActionServer('BaseTrajectoryFollowerServer', BaseTrajectoryFollowingAction, execute_cb=self.execute_cb, auto_start = False)
        self.actionServer.start()
        rospy.loginfo("Base Trajectory Tracking server started")
        
        
    def execute_cb(self, goal):
        rospy.loginfo("Received new goal")
        states = goal.states
        velocities = goal.velocities
        times = np.array(goal.times)

        # Convert states and velocities to numpy arrays
        state_matrix = np.array([[s.x, s.y, s.z] for s in states])  # shape: (N, 3)
        velocity_matrix = np.array([[v.x, v.y, v.z] for v in velocities])  # shape: (N, 3)

        # Create interpolators for x, y, theta
        interpolators = []
        for dim in range(3):
            interpolators.append(
                CubicHermiteSpline(times, state_matrix[:, dim], velocity_matrix[:, dim])
            )

        # Discretize the trajectory
        discrete_times = np.arange(times[0], times[-1], self.dt)
        discrete_states = np.vstack([interp(discrete_times) for interp in interpolators])
        discrete_velocities = np.vstack([interp.derivative()(discrete_times) for interp in interpolators])

        self.traj_d = np.vstack((discrete_states, discrete_velocities))

        # Run the controller
        result = self.FollowTrajectoryDiscrete()

        if result.success:
            rospy.loginfo("Goal succeeded")
            self.actionServer.set_succeeded()
        else:
            rospy.loginfo("Goal failed")
            self.actionServer.set_aborted()

        for _ in range(10):
            if not self.actionServer.is_active():
                self.sendVelocity(0.0, 0.0, 0.0)
                rospy.sleep(0.01)

        
      
    def set_offsets(self):
        self.x_offset = self.odom_to_body_state[0]
        self.y_offset = self.odom_to_body_state[1]
        self.theta_offset = self.odom_to_body_state[2]  
        
    #Callback function for the odometry subscriber
    #This function updates the state of the robot wrt to the odom frame
    def OdomsSubCallback(self, data):
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
        
    #Callback function for the ground truth subscriber
    #This function updates the actual state of the robot wrt to the global frame
    #In simulation, the global frame is the gazebo world frame
    #In the real world, the global frame is the map frame and some estimation algorithm is used to estimate the robot's position in the map frame
    # TODO: Implement an estimation algorithm to estimate the robot's position in the map frame
    def TFcallback(self, event):
        try:
            self.map_to_body_tf = self.tfBuffer.lookup_transform("base_link", "map", rospy.Time(0))
            map_to_odom_tf = self.tfBuffer.lookup_transform("odom", "map", rospy.Time(0))

            self.map_to_body_state[0] = self.map_to_body_tf.transform.translation.x
            self.map_to_body_state[1] = self.map_to_body_tf.transform.translation.y
            orientation = self.map_to_body_tf.transform.rotation
            self.map_to_body_state[2] = quaternion_to_yaw(orientation)
            self.map_to_body_R = self.transform_to_rotation_matrix(self.map_to_body_tf.transform)

            map_to_odom_R = self.transform_to_rotation_matrix(map_to_odom_tf.transform)
            self.map_to_body_state[3:6] = map_to_odom_R @ self.odom_to_body_state[3:6]


        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as ex:
            rospy.loginfo("An error occurred")
        
    #Function to send velocity commands to the robot. The velocity commands are with respect to the body frame
    #Inputs: u_x, u_y, u_theta
    #u_x: linear velocity in the x direction
    #u_y: linear velocity in the y direction
    #u_theta: angular velocity about the z-axis
    def sendVelocity(self, u_x, u_y, u_theta):
        vel_msg = Twist()
        vel_msg.linear.x = u_x
        vel_msg.linear.y = u_y
        vel_msg.linear.z = 0
        vel_msg.angular.x = 0
        vel_msg.angular.y = 0
        vel_msg.angular.z = u_theta
        self.pub.publish(vel_msg)    
    
    #Function to wrap the angle between -pi and pi
    def wrap_angle(self, angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def linearize_around_desired_state(self, vx_d, vy_d, theta_d):
        A = np.array([
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
            [0, 0, -vx_d * np.sin(theta_d) - vy_d * np.cos(theta_d), 0, 0, 0],
            [0, 0, vx_d * np.cos(theta_d) - vy_d * np.sin(theta_d), 0, 0, 0],
            [0, 0, 0, 0, 0, 0]
        ])
        
        B = np.array([
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ])
        
        return A, B
    
    def CalcError(self, actual, desired):
        actual = actual.flatten()
        desired = desired.flatten()
        error = np.zeros(6)

        # Position and velocity differences
        error[:2] = actual[:2] - desired[:2]
        error[3:5] = actual[3:5] - desired[3:5]

        # Orientation and angular velocity differences using quaternions
        quat_actual = tf_trans.quaternion_from_euler(0, 0, actual[2])
        quat_desired = tf_trans.quaternion_from_euler(0, 0, desired[2])
        quat_error = tf_trans.quaternion_multiply(
            quat_actual, tf_trans.quaternion_conjugate(quat_desired)
        )
        _, _, yaw_error = tf_trans.euler_from_quaternion(quat_error)
        error[2] = yaw_error

        quat_actual_vel = tf_trans.quaternion_from_euler(0, 0, actual[5])
        quat_desired_vel = tf_trans.quaternion_from_euler(0, 0, desired[5])
        quat_error_vel = tf_trans.quaternion_multiply(
            quat_actual_vel, tf_trans.quaternion_conjugate(quat_desired_vel)
        )
        _, _, yaw_dot_error = tf_trans.euler_from_quaternion(quat_error_vel)
        error[5] = yaw_dot_error

        return error
        
    def GetControlInputs(self, traj_d):
        x_d, y_d, theta_d, xdot_d, ydot_d, thetadot_d = traj_d
        
        A, B = self.linearize_around_desired_state(xdot_d, ydot_d, theta_d)
        
        Q = np.diag([0.2, 0.2, 0.2, 0.1, 0.1, 0.1])
        R = np.diag([1000.0, 1000.0, 1000.0])
        
        K, _, _ = control.lqr(A, B, Q, R)
        K = np.array(K)  # Convert from control.matlab matrix to numpy array if needed

        state = self.odom_to_body_state.flatten()
        error = self.CalcError(state[:6], traj_d[:6])
        
        u = -K @ error
        u = np.array([xdot_d, ydot_d, thetadot_d]) + u

        # Transform the world coordinate control input to body frame
        R_xy = self.odom_to_body_R[:2, :2]
        u[:2] = R_xy.T @ u[:2]

        return u

    # Function to follow the desired trajectory (TEST)
    # This function calculates the control inputs at each time step and sends the velocity commands to the robot
    # def FollowTrajectory(self):
    #     # while np.norm(self.odom_to_body_state[:3] - self.traj_d[:3, -1]) > 0.05:
    #     # for i in range(self.traj_d.shape[1]):
    #     try:
    #         i = 0
    #         while np.linalg.norm(self.CalcError(self.odom_to_body_state,self.traj_d[:,-1])[:3]) > 0.03 or i < self.traj_d.shape[1]-1:
    #             traj_d = self.traj_d[:, i]
    #             u = self.GetControlInputs(traj_d)
    #             self.sendVelocity(u[0], u[1], u[2])
    #             rospy.sleep(self.dt)
    #             if i < self.traj_d.shape[1]-1:
    #                 i = i + 1
    #                 self.traj_a = np.hstack((self.traj_a, self.odom_to_body_state.reshape(-1, 1)))
    #             if not(i < self.traj_d.shape[1]-1):
    #                 print(np.linalg.norm(self.CalcError(self.odom_to_body_state,self.traj_d[:,-1])))
    #     except rospy.ROSInterruptException as e:
    #         rospy.loginfo("An error occurred")
    #         print(e)
    #     finally:
    #         self.traj_a = np.hstack((self.traj_a, self.odom_to_body_state.reshape(-1, 1)))
    #         for i in range(5):
    #             self.sendVelocity(0, 0, 0)
    #             rospy.sleep(0.01)

    def FollowTrajectoryDiscrete(self):
        tolerance = np.array([0.02,0.02,0.03])
        result_ = BaseTrajectoryFollowingResult()
        try:
            i = 0
            reached = False
            while i < self.traj_d.shape[1]-1:
                if i < self.traj_d.shape[1]-1:
                    i = i + 1
                    self.traj_a = np.hstack((self.traj_a, self.odom_to_body_state.reshape(-1, 1)))
                    traj_d = self.traj_d[:, i]
                if i == self.traj_d.shape[1]-1:
                    traj_d = self.traj_d[:, i]
                    traj_d[3:] = np.zeros(3)
                    reached = np.all(self.CalcError(self.odom_to_body_state,self.traj_d[:,-1])[:3] < tolerance)
                    if reached:
                        break
                    
                u = self.GetControlInputs(traj_d)
                self.sendVelocity(u[0], u[1], u[2])
                rospy.sleep(self.dt)
                
        except rospy.ROSInterruptException as e:
            rospy.loginfo("An error occurred")
            print(e)
        finally:
            self.traj_a = np.hstack((self.traj_a, self.odom_to_body_state.reshape(-1, 1)))
            for i in range(5):
                self.sendVelocity(0, 0, 0)
                rospy.sleep(0.01)
                
        result_.success = True
        print("CURRENT POSITION:  ",self.odom_to_body_state.flatten())
        self.actionServer.set_succeeded(result_)
        return result_         
    
    # def FollowTrajectoryContinuous(self,traj):
    #     feedback_ = BaseTrajectoryFollowingFeedback()
    #     result_ = BaseTrajectoryFollowingResult()
    #     last_time = traj.get_segment_times()[-1]
    #     last_pos = traj.value(last_time).flatten()
    #     last_velocity = traj.EvalDerivative(last_time, 1).flatten()
    #     last_state = np.concatenate((last_pos, last_velocity))
    #     start_time = time.time()
    #     while np.any(self.CalcError(self.odom_to_body_state,last_state)[:3] > np.ones(3)*0.03) or  time.time() - start_time < last_time + self.dt:
    #         if self.actionServer.is_preempt_requested():
    #             rospy.loginfo("Goal preempted")
    #             self.actionServer.set_preempted()
    #             break
    #         t = time.time() - start_time
    #         state_d = traj.value(t)
    #         vel_d = traj.EvalDerivative(t, 1)
    #         traj_d = np.concatenate((state_d, vel_d)).flatten()
    #         u = self.GetControlInputs(traj_d)
    #         self.sendVelocity(u[0], u[1], u[2])
    #         feedback_.progress = float(t/last_time * 100)
    #         self.actionServer.publish_feedback(feedback_)
    #         rospy.sleep(self.dt)
            
    #     for i in range(3):
    #         self.sendVelocity(0,0,0)

        
        # result_.success = True
        # print("CURRENT POSITION:  ",self.odom_to_body_state.flatten())
        # self.actionServer.set_succeeded(result_)
        # return result_,feedback_
                    
    #Function to plot the desired and actual trajectories
    
    def plot_trajectory(self):
        plt.figure()
        plt.plot(self.traj_d[0, 0], self.traj_d[1, 0], 'go', label='Initial Desired Point')
        plt.plot(self.traj_d[0, -1], self.traj_d[1, -1], 'ro', label='Final Desired Point')
        
        plt.plot(self.traj_d[0, :], self.traj_d[1, :], label='Desired Trajectory')
        plt.quiver(self.traj_d[0, ::5], self.traj_d[1, ::5], self.traj_d[3, ::5], self.traj_d[4, ::5], angles='xy', scale_units='xy', scale=10, color='b', label='Desired Velocity')
        
        plt.plot(self.traj_a[0], self.traj_a[1], label='Actual Trajectory')
        plt.quiver(self.traj_a[0, ::5], self.traj_a[1, ::5], self.traj_a[3, ::5], self.traj_a[4, ::5], angles='xy', scale_units='xy', scale=10, color='r', label='Actual Velocity')
        
        plt.xlabel('X position')
        plt.ylabel('Y position')
        plt.title('Trajectory Comparison')
        plt.grid(True)
        plt.show()
        
        
        # Position tracking error
        plt.subplot(2, 1, 1)
        plt.plot(self.traj_d[0, :] - self.traj_a[0, :], label='X Error')
        plt.plot(self.traj_d[1, :] - self.traj_a[1, :], label='Y Error')
        plt.plot(self.wrap_angle(self.traj_d[2, :] - self.traj_a[2, :]), label='Theta Error')
        plt.xlabel('Time step')
        plt.ylabel('Position Error')
        plt.title('Trajectory Tracking Error')
        plt.legend()
        plt.grid(True)
        
        # Velocity tracking error
        plt.subplot(2, 1, 2)
        plt.plot(self.traj_d[3, :] - self.traj_a[3, :], label='X Velocity Error')
        plt.plot(self.traj_d[4, :] - self.traj_a[4, :], label='Y Velocity Error')
        plt.plot(self.traj_d[5, :] - self.traj_a[5, :], label='Theta Velocity Error')
        plt.xlabel('Time step')
        plt.ylabel('Velocity Error')
        plt.title('Velocity Tracking Error')
        plt.legend()
        plt.grid(True)
        plt.show()


#----------------------------Utility functions-----------------------------------#
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
#  ----------------------------Main function-----------------------------------#
if __name__ == '__main__':
    server = BaseTrajectoryFollowerServer(dt=0.1)
    rospy.spin()


# ---------------------------TEST MAIN FUNCTION---------------------------------#
# if __name__ == '__main__':
    
#     rospy.loginfo("Script Started")
#     print("script started")
 
#     dt = 0.1
#     N = 100
#     traj_follower = BaseTrajectoryFollowerServer(dt)
#     # # Define a circular trajectory
#     # r = 14
#     # theta = np.linspace(-np.pi/2,-np.pi/2 +2 * np.pi, N)
#     # x = 0 + r  np.cos(theta) + traj_follower.state[0]
#     # y = r + r * np.sin(theta) + traj_follower.state[1]
#     # x_dot = -r * np.sin(theta)
#     # y_dot = r * np.cos(theta)
#     # theta = np.linspace(0, 2*np.pi, N) + traj_follower.state[2] 
#     # theta_dot = np.gradient(theta, dt)
    
#     #Define a straight line trajectory
#     t = np.linspace(0,2*np.pi,N)
#     x = -np.sin(t) * 1 
#     x_dot = np.gradient(x, dt)
#     x_dot[-1] = 0
#     y =  np.zeros(N) 
#     y_dot = np.gradient(y,dt)
#     theta = np.zeros(N) 
#     theta_dot = np.gradient(theta,dt)
#     theta_dot[-1] = 0
#     traj_d_line_x = np.vstack((x, y, theta, x_dot, y_dot, theta_dot))
    
#     # Define a square trajectory with a given side length
#     side_length = 1 # You can change this value to set the side length of the square
#     x = np.zeros(N) 
#     x_dot = np.zeros(N)
#     y = np.zeros(N) 
#     y_dot = np.zeros(N)
#     theta = np.zeros(N)
#     theta_dot = np.gradient(theta,dt)
#     for i in range(N):
#         if i < N/4:
#             x[i] = (i / (N/4)) * side_length
#             x_dot[i] = side_length / (N/4)
#             y[i] = 0
#             y_dot[i] = 0
#         elif i < N/2:
#             x[i] = side_length
#             x_dot[i] = 0
#             y[i] = ((i - N/4) / (N/4)) * side_length
#             y_dot[i] = side_length / (N/4)
#         elif i < 3*N/4:
#             x[i] = side_length - ((i - N/2) / (N/4)) * side_length
#             x_dot[i] = -side_length / (N/4)
#             y[i] = side_length
#             y_dot[i] = 0
#         else:
#             x[i] = 0
#             x_dot[i] = 0
#             y[i] = side_length - ((i - 3*N/4) / (N/4)) * side_length
#             y_dot[i] = -side_length / (N/4)
#     traj_d_square = np.vstack((x, y, theta, x_dot, y_dot, theta_dot))
    
#     # # Define 8-shaped trajectory
#     A = 2
#     x = A * np.sin(t)
#     y = A * np.sin(t) * np.cos(t)
#     x_dot = np.gradient(x, dt)
#     y_dot = np.gradient(y, dt)
#     theta = np.arctan2(y_dot, x_dot)
#     correction_angle = -theta[0]
#     cos_correction = np.cos(correction_angle)
#     sin_correction = np.sin(correction_angle)
#     correction_matrix = np.array([[cos_correction, -sin_correction], [sin_correction, cos_correction]])
#     corrected_points = correction_matrix @ np.vstack((x, y))
#     x_corrected = corrected_points[0]
#     y_corrected = corrected_points[1]
#     x_dot_corrected = np.gradient(x_corrected, dt)
#     y_dot_corrected = np.gradient(y_corrected, dt)
#     theta_corrected = np.arctan2(y_dot_corrected, x_dot_corrected)
#     theta_corrected -= theta_corrected[0]
#     theta_dot_corrected = np.gradient(theta_corrected, dt)
#     traj_d_eight = np.vstack((x_corrected, y_corrected, theta_corrected, x_dot_corrected, y_dot_corrected, theta_dot_corrected))
    
#     # Define a circle with constant velocity
#     r = 1
#     num_points = N
#     time_step = dt
#     t = np.linspace(0, 2*np.pi, num_points)
#     x = r * np.cos(t- np.pi/2)
#     y = r * np.sin(t- np.pi/2) + r
#     x_dot = np.gradient(x, time_step)
#     y_dot = np.gradient(y, time_step)
#     theta = np.zeros(num_points)
#     theta_dot = np.gradient(theta, time_step)

    
#     traj_d_circle = np.vstack((x, y, theta, x_dot, y_dot, theta_dot))
    
#     #Rotation around the z axis
#     x = np.zeros(N)
#     y = np.zeros(N)
#     x_dot = np.zeros(N)
#     y_dot = np.zeros(N)
#     theta = np.linspace(0,2*np.pi,N)
#     theta_dot = np.gradient(theta,dt)
#     theta_dot[-1] = 0
#     traj_d_rotation = np.vstack((x, y, theta, x_dot, y_dot, theta_dot))
    

    
    
#     #Execute trajectory
#     traj_follower.traj_d = traj_d_line_x
#     traj_follower.FollowTrajectory()
    
    
#     #Stop the Robot
#     for i in range(10):
#             traj_follower.sendVelocity(0.0, 0.0, 0.0)
            
#     #Plot results
#     traj_follower.plot_trajectory()

