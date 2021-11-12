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

        #Define the Subscriber
        rospy.Subscriber('/camera/aligned_depth_to_color/image_raw', Image, self.callback)
        self.r = rospy.Rate(30)
        rospy.spin()

    def histeq(self, image, bins=255):
        image_histogram, bins = np.histogram(image.flatten(), bins, density=True)
        cdf = image_histogram.cumsum()  # cumulative distribution function
        cdf = cdf / cdf[-1]  # normalize

        # use linear interpolation of cdf to find new pixel values
        image_equalized = np.interp(image.flatten(), bins[:-1], cdf)

        return image_equalized.reshape(image.shape), cdf

    def callback(self,msg):
        #Defining 'bridge', which is necessary to transform the image from ROS msg to cv2 UInt16 with one channel
        bridge = CvBridge()

        cv2.namedWindow('MedianFilter')
        cv2.createTrackbar('ksize', 'MedianFilter', 1, 19, self.nothing)

        depth_image = bridge.imgmsg_to_cv2(msg, desired_encoding="16UC1")
        depth_image = 255 - depth_image * 255

        depth_image_median = depth_image
        depth_image_median = depth_image_median.astype(np.uint8)

        ksize = cv2.getTrackbarPos('ksize', 'MedianFilter')

        if (ksize%2) == 0:
            ksize += 1

        print(f'ksize = {ksize}')
        depth_image_median = cv2.medianBlur(depth_image_median, ksize)
        depth_image_median = self.histeq(depth_image_median)[0]

        depth_image = depth_image.astype(np.uint8)

        cv2.imshow('MedianFilter', depth_image_median)
        cv2.imshow('Without Filter', depth_image)
        cv2.waitKey(10)

        #Creating the header of the Msg. It's divided and then multiply by the same value to reduce the number of decimals after the coma in the time stamp
        msg.header.stamp = rospy.Time.now()
        msg.header.stamp.nsecs = int(msg.header.stamp.nsecs/100000000)*100000000

        #Publishing the now dated Msg into the publisher previously defined
        self.pub.publish(msg)
        self.r.sleep()

    def nothing(self):
        pass


PC = PointCloud()