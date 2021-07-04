#!/usr/bin/env python3
# coding: utf-8
from raiv_libraries.image_viewer_opencv import ImageViewerOpenCV
import rospy

rospy.init_node('myCamera')
myViewer = ImageViewerOpenCV()
rospy.spin()
