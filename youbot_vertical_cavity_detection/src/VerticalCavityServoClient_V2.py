#!/usr/bin/env python
import sys
import rospy
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image,CompressedImage,CameraInfo
from cv_bridge import CvBridge, CvBridgeError
import matplotlib.pyplot as plt
import numpy as np
import message_filters
import pyrealsense2 as rs
import time
from PipelineFromJson import PipelineFromJSON
from geometry_msgs.msg import Point
from nav_msgs.msg import Odometry
import rospkg



class VerticalCavityServoClient:
    def __init__(self):
        self.image_pub_1 = rospy.Publisher("image_1",Image, queue_size=3)
        self.image_pub_2 = rospy.Publisher("image_2",Image, queue_size=3)
        self.cavity_goal_pub = rospy.Publisher("vertical_cavity_goal",Point,queue_size=1)

        self.bridge = CvBridge()
        self.image_sub = message_filters.Subscriber("/camera/color/image_raw/compressed", CompressedImage)
        self.depth_sub = message_filters.Subscriber("/camera/aligned_depth_to_color/image_raw", Image)
        self.camera_info_sub = message_filters.Subscriber("/camera/color/camera_info", CameraInfo)
        self.odom_sub = rospy.Subscriber("odom",Odometry,self.odomCallback,queue_size=3)
        self.ts = message_filters.TimeSynchronizer([self.image_sub, self.depth_sub,self.camera_info_sub], 10)
        self.ts.registerCallback(self.imageCallback)

        self.cv_image = None
        self.cv_image_visual = None
        rospack = rospkg.RosPack()
        package_path = rospack.get_path('youbot_vertical_cavity_detection')
        edge_config_path = package_path + "/config/pipeline_config_edge_detection.json"
        delimiter_config_path = package_path + "/config/pipeline_config delimiter_detection.json"
        self.cavity_edge_pipeline = PipelineFromJSON(edge_config_path)
        self.cavity_delimiter_pipeline = PipelineFromJSON(delimiter_config_path)

        #camera_params
        self.K = None
        self.R = None

        #Local robot position info
        self.odom_to_body_tf = np.eye(4)                    #transformation matrix from odom to body frame calculated in odometry callback
        self.odom_to_body_state = np.zeros(6)               #estimated state of the robot wrt to odom frame calculated in odometry callback
        self.odom_to_body_R = np.zeros((3, 3))
        self.odom_to_cavity_start_tf = np.eye(4)            #transformation matrix from odom to cavity_start frame calculated in odometry callback
        self.odom_to_cavity_start_state = np.zeros(3)        #estimated state of the robot wrt to cavity_start frame calculated in odometry callback
        self.odom_to_cavity_start_R = np.zeros((3, 3))     #rotation matrix from odom to cavity_start frame calculated in odometry callback
        self.cavity_start_to_body_tf = np.eye(4)           #transformation matrix from cavity_start to body frame calculated in odometry callback
        self.cavity_start_to_body_state = np.zeros(3)       #estimated state of the robot wrt to cavity_start frame calculated in odometry callback
        self.cavity_start_to_body_R = np.zeros((3, 3))     #rotation matrix from cavity_start to body frame calculated in odometry callback


        #state machine params
        self.state = "edge_detection" #Stores state at which controller is
        self.odom_to_edge_start = None #Stores the start edge pixel position for reasoning about the next edge
        self.odom_to_edge_end = None #Stores the newly detected end edge position for reasoning about the cavity center or cluster end
        self.odom_to_next_base_goal = None #Stores the next base goal to be reached
        self.dir = None #Stores the direction to which the cavity extends (True = to the right, False = to the left)

    def odomCallback(self,data):
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

        if np.all(self.odom_to_cavity_start_R == np.zeros((3, 3))):
            return
        
        self.cavity_start_to_body_tf = invert_tf(self.odom_to_cavity_start_tf) @ self.odom_to_body_tf
        self.cavity_start_to_body_state[0] = self.cavity_start_to_body_tf[0, 3]
        self.cavity_start_to_body_state[1] = self.cavity_start_to_body_tf[1, 3]
        self.cavity_start_to_body_state[2] = quaternion_to_yaw(orientation)
        self.cavity_start_to_body_R = self.cavity_start_to_body_tf[:3, :3]

    def imageCallback(self, imageData, depthData,CameraData):
        curr = time.time()
        try:
            cv_image_color = self.bridge.compressed_imgmsg_to_cv2(imageData, desired_encoding="passthrough")
            cv_image_depth = self.bridge.imgmsg_to_cv2(depthData, desired_encoding="passthrough")
            self.K = np.array(CameraData.K).reshape(3, 3)
            self.R = np.array(CameraData.P).reshape(3, 4)
        except CvBridgeError as e:
            print("ERROR: ",e)

        
   
        if self.state == "edge_detection":
            print(self.state)
            #Detect the edge of the cavity cluster
            self.cv_image_visual = cv_image_color.copy()
            cv_image_depth,dist= self.filterByDepth(cv_image_color,cv_image_depth,1)
            cv_image_edge = cv_image_depth.copy()
            edge_pos,dir = self.detectCavityClusterEdge(cv_image_edge)

            #Detect the next delimiter of the cavity cluster
            cv_image_delim = cv_image_depth.copy()
            cv_image_delim,delim_pos = self.detectCavityDelimiter(cv_image_delim,edge_pos,dir,dist)

            #store state variables before state transition
            self.odom_to_cavity_start_tf = self.odom_to_body_tf
            self.odom_to_cavity_start_state = self.odom_to_body_state[:3]
            self.odom_to_cavity_start_R = self.odom_to_body_R

            body_to_edge_start = np.append(self.pixel_to_body_coord(edge_pos, dist), 0)
            body_to_edge_end = np.append(self.pixel_to_body_coord(delim_pos, dist), 0)
            self.odom_to_edge_start = self.odom_to_body_tf @ np.append(body_to_edge_start, 1)
            self.odom_to_edge_start = self.odom_to_edge_start[:3]
            self.odom_to_edge_end = self.odom_to_body_tf @ np.append(body_to_edge_end, 1)
            self.odom_to_edge_end = self.odom_to_edge_end[:3]

            self.dir = dir
            self.next_base_goal = 0.5*(self.odom_to_edge_start + self.odom_to_edge_end)
            self.next_base_goal[0] = 0
            self.next_base_goal[2] = self.odom_to_cavity_start_state[2]

            input(f"Press Enter to align to cavity center at: {self.next_base_goal}")
            self.requestMoveTo(self.next_base_goal)
            self.state = "Approach cavity"
            cv2.imshow('image',self.cv_image_visual)
            cv2.waitKey(0)
            return
        
        if self.state == "Approach cavity":
            print(self.state)
            #convert edge end position in odom frame to cavity frame
            cavity_start_to_current_goal = invert_tf(self.odom_to_cavity_start_tf) @ np.append(self.odom_to_edge_end, 1)
            cavity_start_to_current_goal[2] = 0
            cavity_start_to_current_goal = cavity_start_to_current_goal[:3]

            #convert goal back to odom frame as the next goal
            self.next_base_goal = self.odom_to_cavity_start_tf @ np.append(cavity_start_to_current_goal, 1)
            self.next_base_goal[0] = 0
            self.next_base_goal[2] = self.odom_to_cavity_start_state[2]
            self.next_base_goal = self.next_base_goal[:3]
    
            input(f"Press Enter to align to delimiter at: {self.next_base_goal}")
            self.requestMoveTo(self.next_base_goal)
            self.state = "Seek new delimiter"
            rospy.sleep(3.0)
            cv2.imshow('image',self.cv_image_visual)
            cv2.waitKey(0)
            return
        
        if self.state == "Seek new delimiter":
            print(self.state)
            self.cv_image_visual = cv_image_color.copy()
            cv_image_depth,dist= self.filterByDepth(cv_image_color,cv_image_depth,1)
            h,w=cv_image_delim.shape[:2]
            curr_edge_pos = w//2
            cv_image_edge = cv_image_depth.copy()
            if (edge_pos is not None) and (dir is not None) and (dir != self.dir):
                if abs(edge_pos - curr_edge_pos) < 20:
                    input("Robot is aligned with end of cluster. Press Enter to end")
                    self.state = "End"

            cv_image_delim = cv_image_depth.copy()
            cv_image_delim,delim_pos = self.detectCavityDelimiter(cv_image_delim,curr_edge_pos,self.dir,dist)

            #store state variables before state transition
            self.odom_to_cavity_start_tf = self.odom_to_body_tf
            self.odom_to_cavity_start_state = self.odom_to_body_state[:3]
            self.odom_to_cavity_start_R = self.odom_to_body_R
            body_to_edge_start = np.append(self.pixel_to_body_coord(curr_edge_pos, dist), 0)
            body_to_edge_end = np.append(self.pixel_to_body_coord(delim_pos, dist), 0)
            self.odom_to_edge_start = self.odom_to_body_tf @ np.append(body_to_edge_start, 1)
            self.odom_to_edge_start = self.odom_to_edge_start[:3]
            self.odom_to_edge_end = self.odom_to_body_tf @ np.append(body_to_edge_end, 1)
            self.odom_to_edge_end = self.odom_to_edge_end[:3]

            self.dir = dir
            self.next_base_goal = 0.5*(self.odom_to_edge_start + self.odom_to_edge_end)
            self.next_base_goal[2] = self.odom_to_cavity_start_state[2]

            input("Press Enter to align to cavity center at : ",self.next_base_goal)
            self.requestMoveTo(self.next_base_goal)
            self.state = "Approach cavity"
            cv2.imshow('image',self.cv_image_visual)
            cv2.waitKey(1)
            rospy.sleep(3.0)
            return

        if self.state == "End":
            print(self.state)
            return

        try:
            print(self.cv_image_visual.shape)
            image_og_msg = self.bridge.cv2_to_imgmsg(self.cv_image_visual, "passthrough")
            self.image_pub_1.publish(image_og_msg)
            cv2.imshow('image',self.cv_image_visual)
            cv2.waitKey(1)
        except CvBridgeError as e:
            print(e)
        
        rospy.sleep(3.0)
        print("Image processed: Time taken: ", time.time() - curr)

    def detectCavityClusterEdge(self,cv_image):
        cv_image, x_pos = self.cavity_edge_pipeline.apply(cv_image)

        if x_pos:
            h, w = cv_image.shape[:2]
            grad_x = cv2.Sobel(cv_image,cv2.CV_16S, 1, 0 ,ksize=21, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
            avg_dir = np.mean(grad_x[:,x_pos])
            dir = avg_dir >= 0
            cv2.line(self.cv_image_visual, (x_pos, 0), (x_pos, h - 1), (0, 255, 255), 3)
            arrow_y = h // 2
            arrow_len = 200
            half_len = arrow_len // 2
            if dir:
                start_pt = (x_pos - half_len, arrow_y)
                end_pt = (x_pos + half_len, arrow_y)
            else:
                start_pt = (x_pos + half_len, arrow_y)
                end_pt = (x_pos - half_len, arrow_y)
            cv2.arrowedLine(self.cv_image_visual, start_pt, end_pt, (0,255,255), 20, tipLength=0.3)
            return x_pos,dir
        
        return x_pos,None
        
    def detectCavityDelimiter(self,cv_image,x_pos,dir,dist):

        #ROI DEFINITION

        #Remove the side that is not the textile cover
        h,w=cv_image.shape[:2]
        mask_edge=np.ones((h,w),dtype=np.uint8)*255
        if dir:mask_edge[:,:x_pos]=0
        else:mask_edge[:,x_pos:]=0

        #Define where to look, a cavity is usually 16 to 24 inches. We use dist as depth and the camera matrices to place a range of areas
        #to determine the area to keep:
        fx = self.K[0, 0]
        cx = self.K[0, 2]
        x_edge_camera = (dist/fx)*(x_pos - cx)
        if dir:
            direction = 1
            x_lower_camera = x_edge_camera + direction*300
            x_upper_camera = x_edge_camera + direction*600
            x_lower_pixel = np.min([w,fx*(x_lower_camera/dist)+cx])
            x_upper_pixel = np.min([w,fx*(x_upper_camera/dist)+cx])
        else:
            direction = -1
            x_lower_camera = x_edge_camera + direction*600
            x_upper_camera = x_edge_camera + direction*300
            x_lower_pixel = np.max([0,fx*(x_lower_camera/dist)+cx])
            x_upper_pixel = np.max([0,fx*(x_upper_camera/dist)+cx])
        
        
        mask_roi = np.zeros((h,w),dtype=np.uint8)
        mask_roi[:,int(x_lower_pixel):int(x_upper_pixel)] = 255
        mask = mask_roi*mask_edge
        cv_image=cv2.bitwise_and(cv_image,cv_image,mask=mask)
        cv_image,x_pos = self.cavity_delimiter_pipeline.apply(cv_image)

        x1 = int(min(x_lower_pixel, x_upper_pixel))
        x2 = int(max(x_lower_pixel, x_upper_pixel))
        cv2.rectangle(self.cv_image_visual, (x1, 0), (x2, h - 1), (0, 0, 0), 2)
        cv2.line(self.cv_image_visual, (x_pos, 0), (x_pos, h - 1), (255, 255, 0), 3)
        return cv_image,x_pos   

    def filterByDepth(self,cv_image_color,cv_image_depth,threshold):
        mask_depth_dist=(cv_image_depth/1000.0<threshold).astype(np.uint8)
        cv_image_color=cv2.bitwise_and(cv_image_color,cv_image_color,mask=mask_depth_dist)
        dist = self.getDist(cv_image_depth)
        return cv_image_color,dist
    
    def getDist(self,cv_image_depth):
        h,w=cv_image_depth.shape[:2]
        patch_size=20
        roi=cv_image_depth[h//2-patch_size//2:h//2+patch_size//2,w//2-patch_size//2:w//2+patch_size//2]
        return np.mean(roi)

    def requestMoveTo(self,goal):
        pt = Point(x = goal[0],y=goal[1],z=goal[2])
        self.cavity_goal_pub.publish(pt)
    
    def pixel_to_body_coord(self,x_pos,dist):
        fx = self.K[0, 0]
        cx = self.K[0, 2]
        y = -(dist/fx)*(x_pos - cx)
        x = dist
        return np.array([x,y])/1000


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



if __name__ == '__main__':
    rospy.init_node('vertical_cavity_servo_client', anonymous=True)
    vcs = VerticalCavityServoClient()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")

        