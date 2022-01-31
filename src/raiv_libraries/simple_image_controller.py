#!/usr/bin/env python
# coding: utf-8

import rospy
from PIL import Image as PILImage
from sensor_msgs.msg import Image

"""
The get_image allows to retrieve the image and its dimensions from a topic 
"""

class SimpleImageController:
    def __init__(self, image_topic='/usb_cam/image_raw'):
        self.image_topic = image_topic

    def get_image(self):
        msg = rospy.wait_for_message(self.image_topic, Image)
        return self.to_pil(msg), msg.width, msg.height

    def to_pil(self, msg, display=False):
        size = (msg.width, msg.height)  # Image size
        img = PILImage.frombytes('RGB', size, msg.data)  # sensor_msg to Image
        return img

if __name__ == '__main__':
    rospy.init_node('simple_image_controller')  # ROS node initialization
    simple_image_controller = SimpleImageController(image_topic='/usb_cam/image_raw')

    while True:
        img, width, height = simple_image_controller.get_image()
        print("image width = {}, height = {}".format(width,height))