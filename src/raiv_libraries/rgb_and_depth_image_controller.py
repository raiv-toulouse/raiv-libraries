#!/usr/bin/env python
# coding: utf-8

import rospy
from PIL import Image as PILImage

from raiv_libraries.image_tools import ImageTools
from sensor_msgs.msg import Image

"""
The get_image allows to retrieve the image and its dimensions from a topic 
"""

class RgbAndDepthImageController:
    def __init__(self, rgb_topic='/camera/color/image_raw', depth_topic='/camera/aligned_depth_to_color/image_raw'):
        self.rgb_topic = rgb_topic
        self.depth_topic = depth_topic

    def get_image(self):
        msg_rgb = rospy.wait_for_message(self.rgb_topic, Image)
        msg_depth = rospy.wait_for_message(self.depth_topic, Image)
        pil_rgb = ImageTools.ros_msg_to_pil(msg_rgb)
        pil_depth = ImageTools.ros_msg_to_pil(msg_depth)
        return pil_rgb, pil_depth, pil_rgb.width, pil_rgb.height


if __name__ == '__main__':
    rospy.init_node('rgb_and_depth_image_controller')  # ROS node initialization
    rgb_and_depth_image_controller = RgbAndDepthImageController(rgb_topic='/camera/color/image_raw', depth_topic='/camera/aligned_depth_to_color/image_raw')
    while True:
        pil_rgb, pil_depth, width, height = rgb_and_depth_image_controller.get_image()
        print("image width = {}, height = {}".format(width,height))
        print(pil_rgb.getextrema())
        pil_rgb.show()
        input('Continuer?')
        print(pil_depth.getextrema())
        pil_depth.show()
        input('Continuer?')
