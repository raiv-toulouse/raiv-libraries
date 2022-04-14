#!/usr/bin/env python
# coding: utf-8

import cv2
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from raiv_libraries.srv import get_coordservice
from raiv_libraries.get_coord_node import InBoxCoord


CROP_WIDTH = CROP_HEIGHT = 50

rospy.init_node('test_get_coord_node')
rgb = rospy.wait_for_message("/RGBClean",  Image)
rgb = CvBridge().imgmsg_to_cv2(rgb, desired_encoding ='bgr8')

cv2.namedWindow('debug')

rospy.wait_for_service('In_box_coordService')
coord_serv = rospy.ServiceProxy('In_box_coordService', get_coordservice)

ct = 0
while True:
    try:
        resp = coord_serv('random', InBoxCoord.PICK, InBoxCoord.ON_OBJECT, CROP_WIDTH, CROP_HEIGHT, None, None)
        cv2.circle(rgb, (resp.x_pixel, resp.y_pixel), radius=2, color=(0, 0, 255), thickness=-1)
        resp = coord_serv('random', InBoxCoord.PLACE, InBoxCoord.IN_THE_BOX, CROP_WIDTH, CROP_HEIGHT, None, None)
        cv2.circle(rgb, (resp.x_pixel, resp.y_pixel), radius=2, color=(255, 0, 0), thickness=-1)
        ct += 1
        print(ct)
        if ct % 10 == 0:
            print("============================================")
            cv2.imshow('debug', rgb)
            cv2.waitKey(1)

    except rospy.ServiceException as e:
        print("Service call failed: %s"%e)