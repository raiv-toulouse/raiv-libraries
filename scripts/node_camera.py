#!/usr/bin/env python3
# coding: utf-8
from raiv_libraries.image_viewer_opencv import ImageViewerOpenCV
import rospy

#
# Node used to test the ImageViewerOpenCV class.
#
# Plug a webcam then :
# rosrun usb_cam usb_cam_node   (to provide a /usb_cam/image_raw topic)
# rosrun raiv_libraries node_camera.py  (to convert the topic image to a OpenCV image and display it in a OpenCV window)
#
# Use : rosservice call /record_image "data: '/home/philippe'"   to save an image in the /home/philippe folder.
#
rospy.init_node('myCamera')
myViewer = ImageViewerOpenCV()
rospy.spin()
