#!/usr/bin/env python

import rospy
import random
from geometry_msgs.msg import Point
from youbot_arm_moveit_config.msg import TargetArm

def publish_random_point():
    # Initialize the ROS node
    rospy.init_node('random_point_publisher')

    # Create a publisher for the target topic
    pub = rospy.Publisher('/target_arm', TargetArm, queue_size=10)

    # Loop to continuously publish messages
    while not rospy.is_shutdown():
        # Create a Point message
        msg =TargetArm()
        time = rospy.Time.now()
        msg.angle =0
        msg.header.stamp = time

        # Publish the point message
        pub.publish(msg)

        # Log info about the message
        rospy.loginfo("Published angel: [%f]", 0.15)

        # Sleep for a random time between 1 and 10 seconds
        rate = random.uniform(1, 0.1)  # Random rate between 1 and 10 Hz
        rospy.sleep(1.0 / rate)  # Sleep for the calculated period

if __name__ == '__main__':
    try:
        publish_random_point()
    except rospy.ROSInterruptException:
        pass