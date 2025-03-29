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
        self.ts = message_filters.TimeSynchronizer([self.image_sub, self.depth_sub,self.camera_info_sub ], 3)
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

        #state machine params
        self.state = "edge_detection"

    def imageCallback(self, imageData, depthData,CameraData):
        curr = time.time()
        try:
            cv_image_color = self.bridge.compressed_imgmsg_to_cv2(imageData, desired_encoding="passthrough")
            cv_image_depth = self.bridge.imgmsg_to_cv2(depthData, desired_encoding="passthrough")
            self.K = np.array(CameraData.K).reshape(3, 3)
            self.R = np.array(CameraData.P).reshape(3, 4)
        except CvBridgeError as e:
            print("ERROR: ",e)
            return
        

        self.cv_image_visual = cv_image_color.copy()
        cv_image_depth,dist= self.filterByDepth(cv_image_color,cv_image_depth,1)

        cv_image_edge = cv_image_depth.copy()
        edge_pos,dir = self.detectCavityClusterEdge(cv_image_edge)

        cv_image_delim = cv_image_depth.copy()
        cv_image_delim,delim_pos = self.detectCavityDelimiter(cv_image_delim,edge_pos,dir,dist)


        try:

            image_og_msg = self.bridge.cv2_to_imgmsg(self.cv_image_visual, "passthrough")
            self.image_pub_1.publish(image_og_msg)
            image_og_msg = self.bridge.cv2_to_imgmsg(cv_image_delim, "passthrough")
            self.image_pub_2.publish(image_og_msg)
        except CvBridgeError as e:
            print(e)
        
        print("Image processed: Time taken: ", time.time() - curr)


    def detectCavityClusterEdge(self,cv_image):
        cv_image, x_pos = self.cavity_edge_pipeline.apply(cv_image)

        if x_pos:
            h, w = cv_image.shape[:2]
            grad_x = cv2.Sobel(cv_image,cv2.CV_16S, 1, 0 ,ksize=11, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
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
        



        

    def simplify_image_colors(self,cv_image, K=5):
        image_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        pixels = image_rgb.reshape(-1, 3)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        _, labels, centers = cv2.kmeans(np.float32(pixels), K, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
        centers = np.uint8(centers)
        segmented_image = centers[labels.flatten()]
        segmented_image = segmented_image.reshape(image_rgb.shape)
        segmented_image_bgr = cv2.cvtColor(segmented_image, cv2.COLOR_RGB2BGR)
        return segmented_image_bgr

    def filterByDepth(self,cv_image_color,cv_image_depth,threshold):
        h,w=cv_image_depth.shape[:2]
        mask_depth_dist=(cv_image_depth/1000.0<threshold).astype(np.uint8)
        cv_image_color=cv2.bitwise_and(cv_image_color,cv_image_color,mask=mask_depth_dist)
        patch_size=20
        roi=cv_image_depth[h//2-patch_size//2:h//2+patch_size//2,w//2-patch_size//2:w//2+patch_size//2]
        dist=np.mean(roi)
        return cv_image_color,dist




if __name__ == '__main__':
    rospy.init_node('vertical_cavity_servo_client', anonymous=True)
    vcs = VerticalCavityServoClient()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
        cv2.destroyAllWindows()  

        