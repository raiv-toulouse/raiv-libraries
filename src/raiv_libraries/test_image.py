#!/usr/bin/env python
# coding: utf-8


import numpy as np
import rospy
import sys
from sensor_msgs.msg import Image
from PyQt5.QtWidgets import *
from PyQt5 import uic
from cv_bridge import CvBridge
import cv2

class TestImage(QWidget):
    """
    XXXX
    """

    def __init__(self, image_topic='/usb_cam/image_raw'):
        super().__init__()
        uic.loadUi("test_image.ui",self) #needs the canvas_create_image_dataset.py file in the current directory
        # Event handlers
        self.image_sub = rospy.Subscriber(image_topic, Image, self.display_image)
        self.sl_value.valueChanged.connect(self.maj)
        self.btn_go.clicked.connect(self.go)
        self.bridge = CvBridge()
        self.lbl_value.setText(str(self.sl_value.value()))

    #
    # Event handlers
    #

    def display_image(self, msg):
        # Convert the ROS image from the topic to a OpenCV Image
        self.image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='16UC1')
        # Display this image in the canvas_in place
        self.canvas_in.set_image(self.image)
        self.maj()

    def maj(self):
        value = self.sl_value.value()
        self.lbl_value.setText(str(self.sl_value.value()))
        # Calculate the histogram of the depth image
        histogram = cv2.calcHist([self.image], [0], None, [1000], [1, 1000])
        distance_camera_to_table = histogram.argmax()
        image_depth_without_table = np.where(self.image <= distance_camera_to_table - value, self.image, 0)
        self.canvas_out.set_image(image_depth_without_table)

    def go(self):
        pass

    def process_click(self, px, py):
        print(px, "   ", py)

    # #Funciton used to normalize the image
    # def _histeq(self,bins=255):
    #     bridge = CvBridge()
    #     image_histogram, bins = np.histogram(self.depth.flatten(), bins, density=True)
    #     cdf = image_histogram.cumsum()  # cumulative distribution function
    #     cdf = cdf / cdf[-1]  # normalize
    #
    #     # use linear interpolation of cdf to find new pixel values
    #     image_equalized = np.interp(self.depth.flatten(), bins[:-1], cdf)
    #     image_equalized = image_equalized.reshape(self.depth.shape)
    #     self.depth = image_equalized*255
    #     self.depth = bridge.cv2_to_imgmsg(self.depth, encoding = 'passthrough')

#
# Main program
#/camera/color/image_raw
if __name__ == '__main__':
    rospy.init_node('test_image')
    rate = rospy.Rate(0.5)
    app = QApplication(sys.argv)
    gui = TestImage(image_topic='/Distance_Here')
    gui.show()
    sys.exit(app.exec_())