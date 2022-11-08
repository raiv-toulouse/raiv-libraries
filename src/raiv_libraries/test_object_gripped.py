#!/usr/bin/env python
# coding: utf-8

# This program is used to determine if the vacuum gripper gets an object or not.
# This information is provided throw a service called /object_gripped which return an ObjectGripped (True or False)
# The idea is to position the object under the camera and compute the number of objects pixels (here : gray pixels for the cylinders)
# If this number is above THRESHOLD_NB_PIXELS, we conclude that an object has been gripped.

import numpy as np
import cv2
import rospy
from cv_bridge import CvBridge
from raiv_libraries.robotUR import RobotUR
import geometry_msgs.msg as geometry_msgs
from sensor_msgs.msg import Image
from raiv_libraries.srv import ObjectGripped, ObjectGrippedResponse

X = 0.3
Y = 0
Z = 0.12
DELAY_TO_MOVE = 4
THRESHOLD_NB_PIXELS = 500  # If this number of gray pixels are detected, we conclude that an object is present

tool_test_gripped_pose = geometry_msgs.Quaternion(0, 0.924, 0, 0.383)
my_robot = RobotUR()

def handle_object_gripped(req):
    initial_pose = my_robot.get_current_pose()

    #Move the robot to the test position
    my_robot.go_to_pose(geometry_msgs.Pose(
                    geometry_msgs.Vector3(0.21, -0.27, 0.12), tool_test_gripped_pose
                ), DELAY_TO_MOVE)

    #Read image
    rgb = rospy.wait_for_message("/camera/color/image_raw",  Image)
    cv_image_rgb = CvBridge().imgmsg_to_cv2(rgb, desired_encoding ='bgr8')

    #Crop the image and convert BGR to HSV
    #cropped_image = cv_image_rgb[350:402, 297:402]
    hsv = cv2.cvtColor(cv_image_rgb, cv2.COLOR_BGR2HSV)

    #Color strength parameters in HSV
    weaker = np.array([40, 40, 40])
    stronger = np.array([180, 105, 180])

    #Threshold HSV image to obtain input color
    mask = cv2.inRange(hsv, weaker, stronger)

    nb_pixels = cv2.countNonZero(mask)
    print(nb_pixels, "grey pixels")

    # Return to initial pose
    my_robot.go_to_pose(initial_pose, DELAY_TO_MOVE)

    gripped = nb_pixels >= THRESHOLD_NB_PIXELS
    return ObjectGrippedResponse(gripped)

if __name__ == "__main__":
    rospy.init_node('test_object_gripped')
    s = rospy.Service('object_gripped', ObjectGripped, handle_object_gripped)
    print("Ready to test_object_gripped.")
    rospy.spin()