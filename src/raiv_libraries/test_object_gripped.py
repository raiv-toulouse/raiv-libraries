#!/usr/bin/env python
# coding: utf-8

import numpy as np
import cv2
import math
import rospy
from cv_bridge import CvBridge
from raiv_libraries.robotUR import RobotUR
import geometry_msgs.msg as geometry_msgs
from sensor_msgs.msg import Image
from raiv_libraries.srv import ObjectGripped, ObjectGrippedResponse

X = 0.3
Y = 0
Z = 0.12

def handle_object_gripped(req):
    #Move the robot
    my_robot=RobotUR
    my_robot.go_to_xyz_position(X, Y, Z, duration = 2)
    my_robot.go_to_pose(geometry_msgs.Pose(
                    geometry_msgs.Vector3(0.442, -0.022, 0.125), RobotUR.tool_test_gripped_pose
                ),8)
    #Read image
    rospy.init_node('Test')
    rgb = rospy.wait_for_message("/RGBClean",  Image)
    cv_image_rgb = CvBridge().imgmsg_to_cv2(rgb, desired_encoding ='bgr8')

    #Crop the image and convert BGR to HSV
    cropped_image = cv_image_rgb[350:402,297:402]
    hsv = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2HSV)

    #cv2.imshow('object hsv', hsv)
    cv2.imshow('object cropped',cropped_image)

    #Color strength parameters in HSV
    weaker = np.array([40,40,40])
    stronger = np.array([180,105,180])

    #Threshold HSV image to obtain input color
    mask = cv2.inRange(hsv, weaker, stronger)
    #cv2.imshow('Result',mask)

    nb_pixels = cv2.countNonZero(mask)
    print(nb_pixels, "grey pixels")
    my_robot.go_to_pose(geometry_msgs.Pose(
                geometry_msgs.Vector3(X, Y, Z), RobotUR.tool_down_pose
            ),10)
    gripped = nb_pixels >= 500
    return ObjectGrippedResponse(gripped)

if __name__ == "__main__":
    rospy.init_node('test_object_gripped')
    s = rospy.Service('object_gripped', ObjectGripped, handle_object_gripped)
    print("Ready to add two ints.")
    rospy.spin()