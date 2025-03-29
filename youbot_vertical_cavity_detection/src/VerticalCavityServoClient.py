#!/usr/bin/env python
import sys
import rospy
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image,CompressedImage
from cv_bridge import CvBridge, CvBridgeError
import matplotlib.pyplot as plt
import numpy as np
import message_filters
import pyrealsense2 as rs
import time


class VerticalCavityServoClient:
    def __init__(self):
        self.image_pub_1 = rospy.Publisher("image_1",Image, queue_size=1)
        self.image_pub_2 = rospy.Publisher("image_2",Image, queue_size=1)
        self.image_pub_3 = rospy.Publisher("image_3",Image, queue_size=1)

        self.bridge = CvBridge()
        self.image_sub = message_filters.Subscriber("/camera/color/image_raw", Image)
        self.depth_sub = message_filters.Subscriber("/camera/aligned_depth_to_color/image_raw", Image)
        self.ts = message_filters.TimeSynchronizer([self.image_sub, self.depth_sub], 10)
        self.ts.registerCallback(self.imageCallback)

    def imageCallback(self, imageData, depthData):
        curr = time.time()
        try:
            cv_image_color = self.bridge.imgmsg_to_cv2(imageData, desired_encoding="bgr8")
            cv_image_depth = self.bridge.imgmsg_to_cv2(depthData, desired_encoding="CV_16UC1")
        except CvBridgeError as e:
            print(e)
            return

        # Use the depth information to filter out points not immediately in front of the camera
        cv_image_og = cv_image_color.copy()
        mask_depth_dist = cv_image_depth/1000.0 < 2.0
        height, width = cv_image_color.shape[:2]
        mask_sides = np.zeros((height, width), dtype=np.uint8)
        mask_sides[:, int(0.2 * width):int(0.8 * width)] = 1
        mask = mask_depth_dist * mask_sides
        cv_image_color = cv2.bitwise_and(cv_image_color, cv_image_color, mask=mask)

        # Filter out the background and only keep the white textile cover
        hsv = cv2.cvtColor(cv_image_color, cv2.COLOR_BGR2HSV)
        lower = np.array([0, 0, 0], dtype=np.uint8)
        upper = np.array([179, 55, 200], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)
        cv_image_color = cv2.bitwise_and(cv_image_color, cv_image_color, mask=mask)

        # # Find the frontier between textile cover and backgroud
        cv_image_greyscale = cv2.cvtColor(cv_image_color, cv2.COLOR_BGR2GRAY)
        cv_image_greyscale = cv2.GaussianBlur(cv_image_greyscale, (17, 17), 0)
        cv_image_greyscale = cv2.Canny(cv_image_greyscale, 100, 200)
        # cnt,_ = cv2.findContours(cv_image_greyscale, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # cv2.drawContours(cv_image_og, cnt, -1, (0, 255, 0), 3)
        # filtered_contours = []
        # for contour in cnt:
        #     M = cv2.moments(contour)
        #     if M["m00"] != 0:
        #         cX = int(M["m10"] / M["m00"])
        #         cY = int(M["m01"] / M["m00"])
        #         if 0.22 * width < cX < 0.78 * width:
        #             filtered_contours.append(contour)


        # # Draw filtered contours
        # biggest = max(filtered_contours, key=cv2.contourArea)
        # # Remove points from biggest that are at the edge of the image
        # biggest = [point for point in biggest if 0.25*width < point[0][0] < 0.75*width and 0.05*height < point[0][1] < 0.95*height]
        # cv2.drawContours(cv_image_og, biggest, -1, (0, 255, 255), 10)

        # if len(biggest) > 0:
        #     [vx, vy, x, y] = cv2.fitLine(np.array(biggest), cv2.DIST_L2, 0, 0.01, 0.01)
        #     lefty = int((-x * vy / vx) + y)
        #     righty = int(((width - x) * vy / vx) + y)
        #     cv2.line(cv_image_og, (width - 1, righty), (0, lefty), (0, 255, 255), 10)

        rho = 1  # distance resolution in pixels of the Hough grid
        theta = np.pi / 180  # angular resolution in radians of the Hough grid
        threshold = 50  # minimum number of votes (intersections in Hough grid cell)
        min_line_length = 200  # minimum number of pixels making up a line
        max_line_gap = 50  # maximum gap in pixels between connectable line segments

        # Run Hough on edge detected image
        lines = cv2.HoughLinesP(cv_image_greyscale, rho, theta, threshold, np.array([]),
                    min_line_length, max_line_gap)
        filtered_lines = []
        if lines is None:
            print("No lines detected")
            return
        for points in lines:
            x1,y1,x2,y2=points[0]
            if (0.25*width < x1 < 0.75*width) or (0.25*width < x2 < 0.75*width):
                slope = (y2-y1)/(x2-x1)
                if abs(slope) > 1:
                    filtered_lines.append(points)
                    cv2.line(cv_image_og, (x1, y1), (x2, y2), (0, 255, 255), 10)

        # Convert CV images to ROS and Publish relevant visualizations
        try:
            image_color_msg = self.bridge.cv2_to_imgmsg(cv_image_og, "bgr8")
            self.image_pub_1.publish(image_color_msg)
        except CvBridgeError as e:
            print(e)
        
        print("Image processed: Time taken: ", time.time() - curr)


if __name__ == '__main__':
    rospy.init_node('vertical_cavity_servo_client', anonymous=True)
    vcs = VerticalCavityServoClient()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
        cv2.destroyAllWindows()  

        