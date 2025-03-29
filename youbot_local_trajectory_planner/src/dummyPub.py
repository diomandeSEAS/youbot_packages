#!/usr/bin/env python

import rospy
import random
from geometry_msgs.msg import Point
from youbot_local_trajectory_planner.msg import TargetBase

def publish_random_point():
    # Initialize the ROS node
    rospy.init_node('random_point_publisher')

    # Create a publisher for the target topic
    pub = rospy.Publisher('/target_base', TargetBase, queue_size=10)

    # Loop to continuously publish messages
    while not rospy.is_shutdown():
        # Create a Point message
        msg =TargetBase();
        time = rospy.Time.now()
        point = Point()
        point.x = 0.0  # x = 0.05
        point.y = 0.0   # y = 0.0
        point.z = 0.10   # z = 0.0
        msg.delta =point
        msg.header.stamp = time

        # Publish the point message
        pub.publish(msg)

        # Log info about the message
        rospy.loginfo("Published Point: [%f, %f, %f]", point.x, point.y, point.z)

        # Sleep for a random time between 1 and 10 seconds
        rate = random.uniform(1, 0.1)  # Random rate between 1 and 10 Hz
        rospy.sleep(1.0 / rate)  # Sleep for the calculated period

if __name__ == '__main__':
    try:
        publish_random_point()
    except rospy.ROSInterruptException:
        pass
