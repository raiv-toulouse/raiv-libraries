#!/usr/bin/env python
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from std_msgs.msg import UInt8
from cv_bridge import CvBridge
from Node_Test_Pkg.srv import validationzoneservice, validationzoneserviceResponse
import argparse


class ImageProcessing:
    def __init__(self):
        self.image = None

        #Initiating the node 'RGB gatherer'
        rospy.init_node('RGBgatherer', anonymous='True')

        #Define the publisher
        self.pub = rospy.Publisher('RGBClean', Image, queue_size=1)

        self.srv = rospy.ServiceProxy('ZoneValidationService', validationzoneservice)
        #Define the subscriber
        rospy.Subscriber('/camera/color/image_raw', Image, self.callback)
        self.r = rospy.Rate(30)
        rospy.wait_for_service('Zone Validation Service')
        rospy.spin()

    def callback(self,msg):
        #Defining 'bridge', which is necessary to transform the image from ROS msg to UInt8 cv2 image
        bridge = CvBridge()
        self.image = bridge.imgmsg_to_cv2(msg, desired_encoding= 'bgr8')

        msg.header.stamp = rospy.Time.now()
        msg.header.stamp.nsecs = int(msg.header.stamp.nsecs/100000000)*100000000

        #Displaying the image live at 30FPS

        key = cv2.waitKey(1)
        cv2.imshow('Image RGB', np.array(self.image))
        cv2.setMouseCallback("Image RGB", self.Click)

        #Publishing the frames into the publisher previously defined
        self.pub.publish(msg)
        self.r.sleep()

    def Click(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            print(x, "et", y)
            ServResponse = self.srv(x,y)
            print('Serv Response', ServResponse)
            if ServResponse.Rep == True:
                print('CE PIXEL EST DANS LA ZONE VALIDE !')
            if ServResponse.Rep == False:
                print("CE PIXEL N'EST PAS DANS LA ZONE VALIDE")                

IP = ImageProcessing()