#!/usr/bin/env python

import numpy as np
import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from raiv_libraries.srv import get_coordservice, get_coordserviceResponse
from raiv_camera_calibration.perspective_calibration import PerspectiveCalibration
import math
import random
import sys
#from PIL import Image

THRESHOLD_ABOVE_TABLE = 10  # Used to select all the pixels above the table
BOX_ELEVATION = 40  # 23   height in mm above the table for the bottom of a box
OBJECTS_HEIGHT = 18
OBJECTS_HEAP = 100 # height of the heap of objects in a box
THRESHOLD_EMPTY_BOX = 50 # A box is empty if the maximum number of pixels < this value
PICK_BOX_IS_LEFT = 1
PICK_BOX_IS_RIGHT = 2
PART_HEIGHT = 25  #height of a part in mm

class InBoxCoord:
    # Used to specify the type of point to generate
    PICK = 1
    PLACE = 2
    # Used to specify if we want a point on an object or just a point in the box (but not necessary on an object)
    ON_OBJECT = True
    IN_THE_BOX = False

    def __init__(self, perspective_calibration):

        self.perspective_calibration = perspective_calibration
        self.image_rgb = None
        self.image_depth = None
        self.distance_camera_to_table = 0
        self.service = rospy.Service('/In_box_coordService', get_coordservice, self.process_service)
        self.init_pick_and_place_boxes()
        rospy.spin()

    ###################################################################################################################
    # Initialisation methods
    ###################################################################################################################

    # Search for boxes in depth image and initialize the pick and place boxes
    def init_pick_and_place_boxes(self):
        self.refresh_rgb_and_depth_images()
        self.image_width = self.image_rgb.shape[1]
        self.image_height = self.image_rgb.shape[0]
        # Calculate the histogram of the depth image
        histogram = cv2.calcHist([self.image_depth], [0], None, [1000], [1, 1000])
        # Take the index with the maximum values (i.e. the value of the table's distance to the camera) e
        # Every pixel with a value under the table value +BOX_ELEVATION milimeters is set to zero.
        self.distance_camera_to_table = histogram.argmax()
        image_depth_without_table = np.where(self.image_depth <= self.distance_camera_to_table - THRESHOLD_ABOVE_TABLE, self.image_depth, 0)
        # Then we obtain clear contours of the boxes
        self.init_left_and_right_boxes(image_depth_without_table)
        vide_left = self.is_box_empty(self.leftbox, image_depth_without_table)
        vide_right = self.is_box_empty(self.rightbox, image_depth_without_table)
        rospy.loginfo(f'La boite gauche est vide : {vide_left}')
        rospy.loginfo(f'La boite droite est vide : {vide_right}')
        # The empty box is for placing, the other one is for picking
        if vide_right:
            rospy.loginfo('Left box for picking')
            self.pick_box = self.scale_contour(self.leftbox, 0.8)
            self.pick_box_angle = self.angleleft
            self.place_box = self.scale_contour(self.rightbox, 0.5)
            self.place_box_angle = self.angleright
            self.picking_box = PICK_BOX_IS_LEFT
        elif vide_left:
            rospy.loginfo('Right box for picking')
            self.pick_box = self.scale_contour(self.rightbox, 0.8)
            self.pick_box_angle = self.angleright
            self.place_box = self.scale_contour(self.leftbox, 0.5)
            self.place_box_angle = self.angleleft
            self.picking_box = PICK_BOX_IS_RIGHT
        else:
            rospy.loginfo('Be sure to have one empty box')

    # Get the contours of the two boxes and init the left and right boxes
    def init_left_and_right_boxes(self, image_depth_without_table):

        image = np.divide(image_depth_without_table, np.amax(image_depth_without_table))
        image = image * 255
        image = np.array(image, dtype=np.uint8)
        contours, hierarchy = cv2.findContours(image, 1, 2)
        imagergb = cv2.merge((image, image, image))
        cntlist = []
        for cnt in contours:
            box = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(box)
            box = np.int0(box)
            #cv2.drawContours(imagergb, [box], 0, (0, 0, 255), 5)
            cntlist.append((cnt, cv2.contourArea(cnt)))

        cntlist = sorted(cntlist, key=lambda x: x[1])

        box1contour = cntlist[-1][0]
        box2contour = cntlist[-2][0]

        box1 = cv2.minAreaRect(box1contour)
        anglebox1 = box1[-1]
        box2 = cv2.minAreaRect(box2contour)
        anglebox2 = box2[-1]

        box1 = cv2.boxPoints(box1)
        box2 = cv2.boxPoints(box2)

        box1 = np.int0(box1)
        box2 = np.int0(box2)

        compteurbox1 = 0
        compteurbox2 = 0

        for pt in box1:
            if pt[0] < self.image_width / 2:
                compteurbox1 += 1

        for pt in box2:
            if pt[0] < self.image_width / 2:
                compteurbox2 += 1

        if compteurbox1 > compteurbox2:
            self.leftbox = box1
            self.angleleft = anglebox1
            self.rightbox = box2
            self.angleright = anglebox2
        else:
            self.leftbox = box2
            self.angleleft = anglebox2
            self.angleright = anglebox1
            self.rightbox = box1

        # if DEBUG:
        #     cv2.drawContours(imagergb, [self.leftbox], 0, (0, 255, 0), 3)
        #     cv2.drawContours(imagergb, [self.rightbox], 0, (255, 0, 0), 3)
        #     cv2.imshow('debug', imagergb)
        #     cv2.waitKey(0)


    ###################################################################################################################
    # Common methods
    ###################################################################################################################

    # Function to refresh the RGB and Depth image
    def refresh_rgb_and_depth_images(self):
        image_rgb = rospy.wait_for_message('/RGBClean', Image)
        image_depth = rospy.wait_for_message('/Distance_Here', Image)
        self.image_rgb = CvBridge().imgmsg_to_cv2(image_rgb, desired_encoding='bgr8')
        self.image_depth = CvBridge().imgmsg_to_cv2(image_depth, desired_encoding='16UC1')
        #cv2.imwrite('/common/guilem/test.png', self.image_rgb)                #Sauver l'image rgb pour guilem
        #np.savetxt('/common/guilem/test_depth.txt',self.image_depth,fmt='%.2f')   #Ecrire la matrice pour guilem


    # Test if this box is empty
    def is_box_empty(self, box, image):
        mask = np.zeros((self.image_height, self.image_width, 1), np.uint8)
        cv2.drawContours(mask, [box], 0, 255, -1)
        element = cv2.getStructuringElement(0, (2 * 20 + 1, 2 * 20 + 1), (20, 20))
        mask = cv2.erode(mask, element)

        masked_image = cv2.bitwise_and(image, image, mask=mask)
        dist_min = int(self.distance_camera_to_table - BOX_ELEVATION - OBJECTS_HEAP)
        dist_max = int(self.distance_camera_to_table - BOX_ELEVATION)
        hist = cv2.calcHist([masked_image], [0], mask, [OBJECTS_HEAP], [dist_min, dist_max])

        # if DEBUG:
        #     cv2.imshow('debug',  masked_image)
        #     cv2.waitKey(0)
        #     print('MAX = ', hist.max())

        return hist.max() < THRESHOLD_EMPTY_BOX


    #To downsize the contour size inside the boxes to avoid the suction cup to come in contact too often with the boxes walls
    #Downsize to 0.8 for the active box and 0.5 for the inactive box
    @staticmethod
    def scale_contour(cnt, scale):
        m = cv2.moments(cnt)
        cx = int(m['m10'] / m['m00'])
        cy = int(m['m01'] / m['m00'])
        cnt_norm = cnt - [cx, cy]
        cnt_scaled = cnt_norm * scale
        cnt_scaled = cnt_scaled + [cx, cy]
        cnt_scaled = cnt_scaled.astype(np.int32)
        return cnt_scaled

    ###################################################################################################################
    # Methods used by service
    ###################################################################################################################

    # Treat the request received by the service
    def process_service(self, req):

        if req.mode == 'random':
            x_pixel, y_pixel = self.generate_random_pick_or_place_points(req.type_of_point, req.on_object)
        elif req.mode == 'fixed':
            x_pixel, y_pixel = req.x, req.y
        elif req.mode == 'random_no_refresh':
            x_pixel, y_pixel = self.generate_random_pick_or_place_points(req.type_of_point, req.on_object, refresh = False)

        depth = self.image_depth
        x, y, z = self.perspective_calibration.from_2d_to_3d([x_pixel, y_pixel], depth)
        print(x, y, z)
        if req.type_of_point == InBoxCoord.PICK:
            rgb_crop, depth_crop = self.generate_cropped_images(x_pixel, y_pixel, self.image_rgb, self.image_depth, req.crop_width, req.crop_height)
            bridge = CvBridge()
            rgb_crop = bridge.cv2_to_imgmsg(rgb_crop, encoding='passthrough')
            depth_crop = bridge.cv2_to_imgmsg(depth_crop, encoding='passthrough')
        else : # for PLACE, we don't need to ci=ompute crop images
            rgb_crop, depth_crop = None, None

        return get_coordserviceResponse(
            rgb_crop=rgb_crop,
            depth_crop=depth_crop,
            x_pixel=x_pixel,
            y_pixel=y_pixel,
            x_robot=x,
            y_robot=y,
            z_robot=z
        )


    #Generate random point inside the box contour
    def generate_random_point_in_box(self, box, angle, point_type, on_object):
        #This part of the code allows us to know what is the angle we are given by OpenCV
        o_i = int(math.sqrt((box[-1][0]-box[2][0])**2+(box[-1][1]-box[2][1])**2))
        oi = int(math.sqrt((box[-1][0]-box[0][0])**2+(box[-1][1]-box[0][1])**2))
        if o_i < oi:
            beta = 90-angle
            pt_ref = box[0]
            largeur = o_i
            longueur = oi
        else:
            beta = -angle
            pt_ref = box[1]
            largeur = oi
            longueur = o_i

        point_ok = False
        while not point_ok:
            x = random.randrange(0, largeur)
            y = random.randrange(0, longueur)
            x2 = int(y * math.sin(math.pi / 180 * beta) + x * math.cos(math.pi / 180 * beta))
            y2 = int(y * math.cos(math.pi / 180 * beta) - x * math.sin(math.pi / 180 * beta))
            x2 = x2 + int(pt_ref[0])
            y2 = y2 + int(pt_ref[1])
            print('profondeur du pixel----------------------', self.image_depth[y2][x2])
            #h_min = int(self.distance_camera_to_table - BOX_ELEVATION - OBJECTS_HEIGHT)
            #h_max = int(self.distance_camera_to_table - BOX_ELEVATION - OBJECTS_HEAP)
            #print('distance table =  ', self.distance_camera_to_table)
            #print ('h_min = : ',h_min, 'h_max = : ', h_max)
            if on_object == InBoxCoord.IN_THE_BOX:
                point_ok = True
            elif 405 < self.image_depth[y2][x2] < 510:  # PICK case
                point_ok = True

        if not self.image_depth[y2][x2] in range(1, self.distance_camera_to_table - 3):
            rospy.loginfo(f'Generating another randpoint due to bad value of depth: {self.image_depth[y2][x2]}')
            return self.generate_random_point_in_box(box, angle, point_type, on_object)

        return x2, y2


    # Generate cropped images of rgb and depth around the selected point
    def generate_cropped_images(self, x, y, image_rgb, image_depth, crop_width, crop_height):
        rgb_crop = image_rgb[
                        int(y - crop_width/2):int(y + crop_width/2),
                        int(x - crop_height/2):int(x + crop_height/2)
                        ]
        depth_crop = image_depth[
                          int(y - crop_width/2):int(y + crop_width/2),
                          int(x - crop_height/2):int(x + crop_height/2)
                          ]
        depth_crop = np.where(depth_crop == 0, self.distance_camera_to_table, depth_crop)

        return rgb_crop, depth_crop


    # Generate a cropped image which center is a random point in the pick or place box
    def generate_random_pick_or_place_points(self, point_type, on_object, refresh = True):
        if refresh == True :
            self.refresh_rgb_and_depth_images()
            self.swap_pick_and_place_boxes_if_needed(self.image_depth)

        if point_type == InBoxCoord.PICK:
            return self.generate_random_point_in_box(self.pick_box, self.pick_box_angle, point_type, on_object)
        else:
            return self.generate_random_point_in_box(self.place_box, self.place_box_angle, point_type, on_object)


    # Determine if the pick box is empty, if so, the pick box becomes the place one and the place box becomes the pick one
    def swap_pick_and_place_boxes_if_needed(self, image):

        # In this part of the function we calculate the mean height inside the designated box, we then compare this
        # mean height to value determined empirically to decide if the box is still full or empty
        image_depth_without_table = np.where(self.image_depth <= self.distance_camera_to_table - THRESHOLD_ABOVE_TABLE, self.image_depth, 0)
        vide_left = self.is_box_empty(self.leftbox, image_depth_without_table)
        vide_right = self.is_box_empty(self.rightbox, image_depth_without_table)

        # left -> right and left becomes empty => right is the new picking box
        if vide_left and self.picking_box == PICK_BOX_IS_LEFT:
            rospy.loginfo('Right box for picking')
            self.pick_box = self.scale_contour(self.rightbox, 0.8)
            self.pick_box_angle = self.angleright
            self.place_box = self.scale_contour(self.leftbox, 0.5)
            self.place_box_angle = self.angleleft
            self.picking_box = PICK_BOX_IS_RIGHT
            # left <- right and right becomes empty => left is the new picking box
        elif vide_right and self.picking_box == PICK_BOX_IS_RIGHT:
            rospy.loginfo('Left box for picking')
            self.pick_box = self.scale_contour(self.leftbox, 0.8)
            self.pick_box_angle = self.angleleft
            self.place_box = self.scale_contour(self.rightbox, 0.5)
            self.place_box_angle = self.angleright
            self.picking_box = PICK_BOX_IS_LEFT

###################################################################################################################
# Main program
###################################################################################################################

if __name__ == '__main__':
    rospy.init_node('In_box_coord')
    pc = PerspectiveCalibration('/common/calibration/camera/camera_data')
    IBC = InBoxCoord(pc)
