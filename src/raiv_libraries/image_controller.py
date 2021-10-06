#!/usr/bin/env python
# coding: utf-8

import os
import time

import rospy
from PIL import Image as PILImage
from sensor_msgs.msg import Image
from raiv_libraries.simple_image_controller import SimpleImageController

"""
This class is used to manage sensor_msgs Images.
"""

class ImageController(SimpleImageController):
    def __init__(self, path=os.path.dirname(os.path.realpath(__file__)), image_topic='/usb_cam/image_raw'):
        super().__init__(image_topic)
        self.ind_saved_images = 0  # Index which will tell us the number of images that have been saved
        self.success_path = "{}/success".format(path)  # Path where the images are going to be savedF
        self.fail_path = "{}/fail".format(path)  # Path where the images are going to be saved
        self._create_if_not_exist(path)
        self._create_if_not_exist(self.success_path)
        self._create_if_not_exist(self.fail_path)

    def _create_if_not_exist(self, path):
        # If it does not exist, we create the path folder in our workspace
        try:
            os.stat(path)
        except:
            os.mkdir(path)

    def record_image(self, img, success):
        path = self.success_path if success else self.fail_path  # The path were we want to save the image is
        image_path = '{}/img{}.png'.format(  # Saving image
            path,  # Path
            time.time())  # FIFO queue

        img.save(image_path)

        self.ind_saved_images += 1  # Index increment


if __name__ == '__main__':
    rospy.init_node('image_recorder')  # ROS node initialization
    image_controller = ImageController(path='/home/phil/ros_pictures', image_topic='/usb_cam/image_raw')
    while True:
        img, width, height = image_controller.get_image()
        image_controller.record_image(img, True)