#!/usr/bin/env python
import cv2
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import numpy as np


class PointCloud:

    def __init__(self):

        #Initiating the node 'Distance_Gatherer'
        rospy.init_node('Distance_Gatherer')

        #Define the publisher
        self.pub = rospy.Publisher('Distance_Here', Image, queue_size=1)
        self.pub_filtered = rospy.Publisher('Filtered_Distance_Here', Image, queue_size = 1)

        #Define the Subscriber
        rospy.Subscriber('/camera/aligned_depth_to_color/image_raw', Image, self.callback)
        self.r = rospy.Rate(30)
        rospy.spin()

    #This function is used to normalize the image in order to make it more displayable
    def normalize(self, image, bins=255):
        image_histogram, bins = np.histogram(image.flatten(), bins, density=True)
        cdf = image_histogram.cumsum()  # cumulative distribution function
        cdf = cdf / cdf[-1]  # normalize

        # use linear interpolation of cdf to find new pixel values
        image_equalized = np.interp(image.flatten(), bins[:-1], cdf)

        return image_equalized.reshape(image.shape), cdf

    def callback(self,msg):

        #Defining 'bridge', which is necessary to transform the image from ROS msg to cv2 UInt16 with one channel
        bridge = CvBridge()

        cv2.namedWindow('MedianFilter & Normalization')
        cv2.createTrackbar('ksize', 'MedianFilter & Normalization', 1, 19, self.nothing)

        #get depth image
        depth_image = bridge.imgmsg_to_cv2(msg, desired_encoding="16UC1")
        print(depth_image)

        #This is used to inverse the black and white colors
        depth_image = 255 - depth_image * 255

        print(depth_image)

        depth_image_median = depth_image
        depth_image_median = depth_image_median.astype(np.uint8)

        ksize = cv2.getTrackbarPos('ksize', 'MedianFilter & Normalization')

        #The Kernel size can only be an odd number in order to have a center pixel. So if it's even, we add one.
        if (ksize%2) == 0:
            ksize += 1

        print(f'ksize = {ksize}')

        #We apply the Median Blur filter to the image
        depth_image_median = cv2.medianBlur(depth_image_median, ksize)

        #We normalize the image
        depth_image_median = self.normalize(depth_image_median)[0]

        depth_image = depth_image.astype(np.uint8)

        #Display the images
        cv2.imshow('MedianFilter & Normalization', depth_image_median)
        cv2.imshow('Without Filter', depth_image)
        cv2.waitKey(10)

        #Publishing the now dated Msg into the publisher previously defined
        self.pub.publish(msg)
        self.r.sleep()

    def nothing(self):
        pass

PC = PointCloud()