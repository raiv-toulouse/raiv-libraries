#!/usr/bin/env python

import numpy as np
import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from raiv_libraries.srv import get_coordservice, get_coordserviceResponse
import math
import random

# Used to specify the type of point to generate
PICK = 1
PLACE = 2
BOX_ELEVATION = 40  # height in mm above the table for the bottom of a box
OBJECTS_HEIGHT = 100 # height of the heap of obejcts in a box

class InBoxCoord:

    def __init__(self):

        #Declaration of nodes and global variables

        rospy.init_node('In_box_coord')

        self.image_rgb = None
        self.image_depth = None

        self.rgb_crop = None
        self.depth_crop = None

        self.pub = rospy.Publisher('Point_in_box', Image, queue_size=1)

        self.leftbox = None
        self.angleleft = 0

        self.rightbox = None
        self.angleright = 0

        self.r = rospy.Rate(0.3)

        self.table_distance = 0

        self.init_pick_and_place_boxes()

        # As a start, we check if the left box is empty or full, this is mainly used to define all the variables,
        # you should always start with the left box full and the right one empty

        self.service = rospy.Service('/In_box_coordService', get_coordservice, self.process_coord)
        rospy.spin()

    # Searchh for boxes in depth image and initialize the pick and place boxes
    def init_pick_and_place_boxes(self):
        self.refresh_rgb_and_depth_images()
        self.image_width = self.image_rgb.shape[1]
        self.image_height = self.image_rgb.shape[0]
        # Calculate the histogram of the depth image
        histogram = cv2.calcHist([self.image_depth], [0], None, [1000], [1, 1000])
        # Take the index with the maximum values (i.e. the value of the table's distance to the camera) e
        # Every pixel with a value under the table value +BOX_ELEVATION milimeters is set to zero.
        self.table_distance = histogram.argmax()
        image_depth_without_table = np.where(self.image_depth <= self.table_distance - BOX_ELEVATION, self.image_depth, 0)
        # Then we obtain clear contours of the boxes
        self.init_left_and_right_boxes(image_depth_without_table)
        vide_left = self.is_box_empty(self.leftbox, image_depth_without_table)
        vide_right= self.is_box_empty(self.rightbox, image_depth_without_table)
        rospy.loginfo(f'La boite gauche est vide : {vide_left}')
        rospy.loginfo(f'La boite droite est vide : {vide_right}')
        mean_height_left = self.compute_mean_box_height(self.leftbox, image_depth_without_table)
        mean_height_right = self.compute_mean_box_height(self.rightbox, image_depth_without_table)
        rospy.loginfo(f'Left mean : {mean_height_left}')
        rospy.loginfo(f'Right mean : {mean_height_right}')
        if mean_height_left < mean_height_right:
            rospy.loginfo('Left box for picking')
            self.pick_box = self.scale_contour(self.leftbox, 0.8)
            self.pick_box_angle = self.angleleft
            self.place_box = self.scale_contour(self.rightbox, 0.5)
            self.place_box_angle = self.angleright
        else:
            rospy.loginfo('Right box for picking')
            self.pick_box = self.scale_contour(self.rightbox, 0.8)
            self.pick_box_angle = self.angleright
            self.place_box = self.scale_contour(self.leftbox, 0.5)
            self.place_box_angle = self.angleleft

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
            cv2.drawContours(imagergb, [box], 0, (0, 0, 255), 5)
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

        cv2.drawContours(imagergb, [self.leftbox], 0, (0, 255, 0), 3)
        cv2.drawContours(imagergb, [self.rightbox], 0, (255, 0, 0), 3)


    # Function to refresh the RGB and Depth image
    def refresh_rgb_and_depth_images(self):

        image_rgb = rospy.wait_for_message('/RGBClean', Image)
        image_depth = rospy.wait_for_message('/Distance_Here', Image)

        self.image_rgb = CvBridge().imgmsg_to_cv2(image_rgb, desired_encoding='bgr8')
        self.image_depth = CvBridge().imgmsg_to_cv2(image_depth, desired_encoding='16UC1')

    # Test if this box is empty
    def is_box_empty(self, box, image):
        mask = np.zeros((self.image_height, self.image_width, 1), np.uint8)
        cv2.drawContours(mask, [box], 0, 255, -1)
        element = cv2.getStructuringElement(0, (2 * 20 + 1, 2 * 20 + 1), (20, 20))
        mask = cv2.erode(mask, element)

        #image = cv2.bitwise_and(image * 255, image * 255, mask=mask)  GENERATE A BUG
        image = cv2.bitwise_and(image, image, mask=mask)
        dist_min = int(self.table_distance-BOX_ELEVATION-OBJECTS_HEIGHT)
        dist_max = int(self.table_distance-BOX_ELEVATION)
        hist = cv2.calcHist([image], [0], mask, [dist_max-dist_min], [dist_min, dist_max])
        print(hist.max())
        return hist.max() < 50


    # Calculate the mean height inside the designated box
    def compute_mean_box_height(self, box, image):
        mask = np.zeros((self.image_height, self.image_width, 1), np.uint8)
        cv2.drawContours(mask, [box], 0, 255, -1)
        element = cv2.getStructuringElement(0, (2 * 20 + 1, 2 * 20 + 1), (20, 20))
        mask = cv2.erode(mask, element)

        #image = cv2.bitwise_and(image * 255, image * 255, mask=mask)  GENERATE A BUG
        image = cv2.bitwise_and(image, image, mask=mask)
        hist = cv2.calcHist([image], [0], mask, [999], [1, 1000])

        mean_height = 0
        pool = 1

        for idx, val in enumerate(hist):
            if int(val) != 0:
                mean_height += int(idx)*int(val)
                pool += val
        mean_height = mean_height/pool

        return mean_height

    # Determine if the active box is empty, if so, the active box becomes the inactive one and the inactive box becomes the active one
    def check_emptyness(self, image):

        # In this part of the function we calculate the mean height inside the designated box, we then compare this
        # mean height to value determined empirically to decide if the box is still full or empty


        # Change the value of the height mean depending on which is the active box, this is due to the unequal layer of cork in the boxes
        """
        if self.pick_box_angle is self.angleleft:
            dh = 25

        elif self.pick_box_angle is self.angleright:
            dh = 29
        """
        dh = 27  # TODO  résoudre

        rospy.loginfo(f'self background index : { self.table_distance}')
        rospy.loginfo(f'self background index - 24 : {self.table_distance - 24}')
        rospy.loginfo(f'mean : { mean_height}')

        #In this part of the function we change global variables depending on the result of our mean height comparaison

        if mean_height < self.table_distance - dh and self.pick_box_angle is self.angleleft:
            rospy.loginfo('Left box for picking')
            self.pick_box = self.scale_contour(self.leftbox, 0.8)
            self.pick_box_angle = self.angleleft
            self.place_box = self.scale_contour(self.rightbox, 0.5)
            self.place_box_angle = self.angleright

        elif mean_height < self.table_distance - dh and self.pick_box_angle is self.angleright:
            rospy.loginfo('Right box for picking')
            self.pick_box = self.scale_contour(self.rightbox, 0.8)
            self.pick_box_angle = self.angleright
            self.place_box = self.scale_contour(self.leftbox, 0.5)
            self.place_box_angle = self.angleleft

        elif mean_height >= self.table_distance - dh and self.pick_box_angle is self.angleleft:
            rospy.loginfo('Right box for picking')
            self.pick_box = self.scale_contour(self.rightbox, 0.8)
            self.pick_box_angle = self.angleright
            self.place_box = self.scale_contour(self.leftbox, 0.5)
            self.place_box_angle = self.angleleft

        elif mean_height >= self.table_distance - dh and self.pick_box_angle is self.angleright:
            rospy.loginfo('Left box for picking')
            self.pick_box = self.scale_contour(self.leftbox, 0.8)
            self.pick_box_angle = self.angleleft
            self.place_box = self.scale_contour(self.rightbox, 0.5)
            self.place_box_angle = self.angleright
        else:
            rospy.loginfo(f'box place : {self.place_box}')
            rospy.loginfo(f'self background -20 : {self.table_distance - 20}')

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

    #
    # Methods used by service
    #
    
    #Generate random point inside the box contour
    def generate_random_point_in_box(self, box, angle):
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

        x = random.randrange(0, largeur)
        y = random.randrange(0, longueur)

        x2 = int(y*math.sin(math.pi/180*beta) + x*math.cos(math.pi/180*beta))
        y2 = int(y*math.cos(math.pi/180*beta) - x*math.sin(math.pi/180*beta))

        x2 = x2 + int(pt_ref[0])
        y2 = y2 + int(pt_ref[1])

        if not self.image_depth[y2][x2] in range(1, self.table_distance - 3):
            rospy.loginfo(f'Generating another randpoint due to bad value of depth: {self.image_depth[y2][x2]}')
            return self.generate_random_point_in_box(box, angle)

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

        depth_crop = np.where(depth_crop == 0, self.table_distance, depth_crop)

        return rgb_crop, depth_crop


    def generate_random_pick_or_place_points(self, point_type):

        self.refresh_rgb_and_depth_images()
        self.check_emptyness(self.image_depth)
        if point_type == PICK:
            return self.generate_random_point_in_box(self.pick_box, self.pick_box_angle)
        else:
            return self.generate_random_point_in_box(self.place_box, self.place_box_angle)


    # Treat the request received by the service
    def process_coord(self, req):

        # Mode 'refresh', doesn't generate anything, just refresh the RGB and depth images
        if req.mode == 'refresh':
            self.refresh_rgb_and_depth_images()

            return get_coordserviceResponse(
                rgb=None,
                depth=None,
                xpick=None,
                ypick=None,
                xplace=None,
                yplace=None,
                hist_max=None
            )
        #Mode 'Random', used to generate randomly the picking point and the placing point
        elif req.mode == 'random':
            xpick, ypick = self.generate_random_pick_or_place_points(PICK)
            xplace, yplace = self.generate_random_pick_or_place_points(PLACE)
            rgb_crop, depth_crop = self.generate_cropped_images(xpick, ypick, self.image_rgb, self.image_depth, req.width, req.height)
            bridge = CvBridge()

            return get_coordserviceResponse(
                rgb=bridge.cv2_to_imgmsg(rgb_crop, encoding='passthrough'),
                depth=bridge.cv2_to_imgmsg(depth_crop, encoding='passthrough'),
                xpick=xpick,
                ypick=ypick,
                xplace=xplace,
                yplace=yplace,
                hist_max=self.table_distance
            )
        #Mode "fixed", used to generate a crop around a non-random point given in the request
        elif req.mode == 'fixed':
            xplace, yplace = self.generate_random_pick_or_place_points(PLACE)
            rgb_crop, depth_crop = self.generate_cropped_images(req.x, req.y, self.image_rgb, self.image_depth, req.width, req.height)
            bridge = CvBridge()

            return get_coordserviceResponse(
                rgb=bridge.cv2_to_imgmsg(rgb_crop, encoding='passthrough'),
                depth=bridge.cv2_to_imgmsg(depth_crop, encoding='passthrough'),
                xpick=req.x,
                ypick=req.y,
                xplace=xplace,
                yplace=yplace,
                hist_max=self.table_distance
            )

###
# Main program
###
if __name__ == '__main__':
    IBC = InBoxCoord()
