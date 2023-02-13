#!/usr/bin/env python

import numpy as np
import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from raiv_libraries.srv import get_coordservice, get_coordserviceResponse
from raiv_libraries.srv import GetPickingBoxCentroid, GetPickingBoxCentroidResponse
from raiv_libraries.srv import PickingBoxIsEmpty, PickingBoxIsEmptyResponse
from raiv_camera_calibration.perspective_calibration import PerspectiveCalibration
from raiv_libraries.image_tools import ImageTools
import math
import random
from PIL import Image as PILImage


BOX_ELEVATION = 30  # 23   height in mm above the table for the bottom of a box
OBJECTS_HEIGHT = 18
OBJECTS_HEAP = 100 # height of the heap of objects in a box
THRESHOLD_EMPTY_BOX = 50 # A box is empty if the maximum number of pixels < this value
PICK_BOX_IS_LEFT = 1
PICK_BOX_IS_RIGHT = 2
KERNEL_SIZE_FOR_BOX_EXTRACTION = 40
DEBUG = False


class InBoxCoord:
    THRESHOLD_ABOVE_TABLE = 10  # Used to select all the pixels above the table
    # Used to specify the type of point to generate
    PICK = 1
    PLACE = 2
    # Used to specify if we want a point on an object or just a point in the box (but not necessary on an object)
    ON_OBJECT = True
    IN_THE_BOX = False


    def __init__(self, perspective_calibration):

        self.perspective_calibration = perspective_calibration
        self.bgr_cv = None
        self.depth_cv = None
        self.distance_camera_to_table = 0
        rospy.Service('/In_box_coordService', get_coordservice, self.process_service)
        rospy.Service('/Is_Picking_Box_Empty', PickingBoxIsEmpty, self.is_picking_box_empty)
        rospy.Service('/Get_picking_box_centroid', GetPickingBoxCentroid, self._get_picking_box_centroid)


    ###################################################################################################################
    # Initialisation methods
    ###################################################################################################################

    # Search for boxes in depth image and initialize the pick and place boxes
    def init_pick_and_place_boxes(self):
        self.refresh_rgb_and_depth_images()
        self.image_width = self.bgr_cv.shape[1]
        self.image_height = self.bgr_cv.shape[0]
        # Calculate the histogram of the depth image
        dist_max = np.max(self.depth_cv).item()
        histogram = cv2.calcHist([self.depth_cv], [0], None, [dist_max], [1, dist_max])
        # Take the index with the maximum values (i.e. the value of the table's distance to the camera) e
        # Every pixel with a value under the table value +BOX_ELEVATION milimeters is set to zero.
        self.distance_camera_to_table = histogram.argmax()
        image_depth_without_table = np.where(self.depth_cv <= self.distance_camera_to_table - InBoxCoord.THRESHOLD_ABOVE_TABLE, self.depth_cv, 0)
        # Then we obtain clear contours of the boxes
        self.init_left_and_right_boxes(image_depth_without_table)
        vide_left = self.is_box_empty(self.leftbox, image_depth_without_table)
        vide_right = self.is_box_empty(self.rightbox, image_depth_without_table)
        rospy.loginfo(f'Left box is {"empty" if vide_left else "full"}')
        rospy.loginfo(f'Right box is {"empty" if vide_right else "full"}')
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

    def _angle_and_nb_pts_on_left(self, box_contour):
        box_2D = cv2.minAreaRect(box_contour)  # return center(x, y), (width, height), angle of rotation
        angle_box = box_2D[-1]  # angla of rotation
        box_pts_float = cv2.boxPoints(box_2D)
        box_pts = np.int0(box_pts_float)
        nb_pts_on_left = 0
        for pt in box_pts:
            if pt[0] < self.image_width / 2:
                nb_pts_on_left += 1
        return angle_box, nb_pts_on_left, box_pts

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

        (angle_box1, nb_pts_on_left_box1, box1_pts) = self._angle_and_nb_pts_on_left(box1contour)
        (angle_box2, nb_pts_on_left_box2, box2_pts) = self._angle_and_nb_pts_on_left(box2contour)

        if nb_pts_on_left_box1 > nb_pts_on_left_box2:
            self.leftbox = box1_pts
            self.angleleft = angle_box1
            self.rightbox = box2_pts
            self.angleright = angle_box2
            moments = cv2.moments(box1contour)
        else:
            self.leftbox = box2_pts
            self.angleleft = angle_box2
            self.rightbox = box1_pts
            self.angleright = angle_box1
            moments = cv2.moments(box1contour)
        # Compute the centroid of the picking box
        cx = int(moments['m10']/moments['m00'])
        cy = int(moments['m01']/moments['m00'])
        self.pick_box_centroid = (cx, cy)
        if DEBUG:
            cv2.drawContours(imagergb, [self.leftbox], 0, (0, 255, 0), 3)
            cv2.drawContours(imagergb, [self.rightbox], 0, (255, 0, 0), 3)
            # Visualize imagergb with debugger / view as image, cv2.imshow doesn't work
            # cv2.imshow('debug', imagergb)
            # cv2.waitKey(10)


    ###################################################################################################################
    #  Common methods
    ###################################################################################################################

    def _get_picking_box_centroid(self, req):
        return GetPickingBoxCentroidResponse(x_centroid=self.pick_box_centroid[0],
                                             y_centroid=self.pick_box_centroid[1],
                                             z_centroid=0)

    # Function to refresh the RGB and Depth image
    def refresh_rgb_and_depth_images(self):
        rgb_msg = rospy.wait_for_message('/camera/color/image_raw', Image)
        depth_msg = rospy.wait_for_message('/camera/aligned_depth_to_color/image_raw', Image)
        self.bgr_cv = CvBridge().imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8')
        self.depth_cv = CvBridge().imgmsg_to_cv2(depth_msg, desired_encoding='16UC1')

    def is_picking_box_empty(self, req):
        dist_max = np.max(self.depth_cv).item()
        histogram = cv2.calcHist([self.depth_cv], [0], None, [dist_max], [1, dist_max])
        distance_camera_to_table = histogram.argmax()
        image_depth_without_table = np.where(self.depth_cv <= distance_camera_to_table - InBoxCoord.THRESHOLD_ABOVE_TABLE, self.depth_cv, 0)
        return PickingBoxIsEmptyResponse(self.is_box_empty(self.pick_box, image_depth_without_table))

    # Test if this box is empty
    def is_box_empty(self, box, image):
        mask = np.zeros((self.image_height, self.image_width, 1), np.uint8)
        cv2.drawContours(mask, [box], 0, 255, -1)
        #element = cv2.getStructuringElement(0, (2 * 20 + 1, 2 * 20 + 1), (20, 20))
        kernel = (2 * KERNEL_SIZE_FOR_BOX_EXTRACTION + 1, 2 * KERNEL_SIZE_FOR_BOX_EXTRACTION + 1)
        element = cv2.getStructuringElement(cv2.MORPH_RECT, kernel)
        mask = cv2.erode(mask, element)
        masked_image = cv2.bitwise_and(image, image, mask=mask)
        dist_min = int(self.distance_camera_to_table - BOX_ELEVATION - OBJECTS_HEAP)
        dist_max = int(self.distance_camera_to_table - BOX_ELEVATION)
        hist = cv2.calcHist([masked_image], [0], mask, [OBJECTS_HEAP], [dist_min, dist_max])
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
        """
        In-box_coord_service have 4 modes:

        * random : This mode launch the service with a rgb, deepth image refresh and he control the swap
        * Fixed : This mode is the same as the random mode but the pixel is defined in the call of the service
        * random_no_refresh : This mode launch the service with the same rgb and deepth image, no refresh is processed
        * random_no_swap : This mode launch the service with just a rgb and deepth refresh but no swap

        """
        if req.mode == 'random':
            x_pixel, y_pixel = self.generate_random_pick_or_place_points(req.type_of_point, req.on_object, color=False)
        elif req.mode == 'fixed':
            x_pixel, y_pixel = req.x, req.y
            self.refresh_rgb_and_depth_images()
        elif req.mode == 'random_no_refresh':
            x_pixel, y_pixel = self.generate_random_pick_or_place_points(req.type_of_point, req.on_object, refresh=False, swap=False, color=False)
        elif req.mode == 'random_no_swap':
            x_pixel, y_pixel = self.generate_random_pick_or_place_points(req.type_of_point, req.on_object, swap=False, color=False)
        elif req.mode == 'color':
            x_pixel, y_pixel = self.generate_random_pick_or_place_points(req.type_of_point, req.on_object, color=True)

        depth = self.depth_cv
        x, y, z = self.perspective_calibration.from_2d_to_3d([x_pixel, y_pixel], depth)
        if req.type_of_point == InBoxCoord.PICK:
            rgb_pil = ImageTools.numpy_to_pil(cv2.cvtColor(self.bgr_cv, cv2.COLOR_BGR2RGB))
            depth_pil = ImageTools.numpy_to_pil(self.depth_cv)
            rgb_crop_pil = ImageTools.crop_xy(rgb_pil, x_pixel, y_pixel, req.crop_width, req.crop_height)
            depth_crop_pil = ImageTools.crop_xy(depth_pil, x_pixel, y_pixel, req.crop_width, req.crop_height)
            rgb_crop = ImageTools.pil_to_numpy(rgb_crop_pil)
            depth_crop = ImageTools.pil_to_numpy(depth_crop_pil)
            bridge = CvBridge()
            rgb_crop = bridge.cv2_to_imgmsg(rgb_crop, encoding='passthrough')
            depth_crop = bridge.cv2_to_imgmsg(depth_crop, encoding='passthrough')
        else : # for PLACE, we don't need to compute crop images
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
            if on_object == InBoxCoord.IN_THE_BOX:
                point_ok = True
            elif 405 < self.depth_cv[y2][x2] < 510:  # PICK case
                point_ok = True
        if not self.depth_cv[y2][x2] in range(1, self.distance_camera_to_table - 3):
            rospy.loginfo(f'Generating another randpoint due to bad value of depth: {self.depth_cv[y2][x2]}')
            return self.generate_random_point_in_box(box, angle, point_type, on_object)
        return x2, y2

    # Generate a cropped image which center is a random point in the pick or place box
    def generate_random_pick_or_place_points(self, point_type, on_object, refresh = True, swap = True, color=False):
        if refresh == True :
            self.refresh_rgb_and_depth_images()
        if swap == True :
            self.swap_pick_and_place_boxes_if_needed(self.depth_cv)
        if color==False and point_type == InBoxCoord.PICK:
            return self.generate_random_point_in_box(self.pick_box, self.pick_box_angle, point_type, on_object)
        if color==False and point_type == InBoxCoord.PLACE:
            return self.generate_random_point_in_box(self.place_box, self.place_box_angle, point_type, on_object)
        if color==True and point_type == InBoxCoord.PICK:
            print("generate_random_pick_or_place_points : color==True and point_type == InBoxCoord.PICK")
            return self.generate_random_point_in_box_color(self.pick_box, self.pick_box_angle, point_type, on_object)

    def generate_random_point_in_box_color(self, box, angle, point_type, on_object):
        # This part of the code allows us to know what is the angle we are given by OpenCV
        o_i = int(math.sqrt((box[-1][0] - box[2][0]) ** 2 + (box[-1][1] - box[2][1]) ** 2))
        oi = int(math.sqrt((box[-1][0] - box[0][0]) ** 2 + (box[-1][1] - box[0][1]) ** 2))
        if o_i < oi:
            beta = 90 - angle
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
            image_rgb = cv2.cvtColor(self.bgr_cv, cv2.COLOR_BGR2RGB)
            image_pil = ImageTools.numpy_to_pil(image_rgb)
            image_pil.save("/common/work/stockage_image_test/test.png")
            print("*******************************************")
            print("******************************************************* C EST QUOI CE TRUC?????")
            print("*******************************************")
            i = PILImage.open("/common/work/stockage_image_test/test.png")
            (rouge, vert, bleu) = i.getpixel((x2, y2))
            if on_object == InBoxCoord.IN_THE_BOX:
                point_ok = True
            elif rouge < 15 and vert > 25 and bleu < 40:
                point_ok = True
        if not self.depth_cv[y2][x2] in range(1, self.distance_camera_to_table - 3):
            rospy.loginfo(f'Generating another randpoint due to bad value of depth: {self.depth_cv[y2][x2]}')
            return self.generate_random_point_in_box_color(box, angle, point_type, on_object)
        return x2, y2


    # Determine if the pick box is empty, if so, the pick box becomes the place one and the place box becomes the pick one
    def swap_pick_and_place_boxes_if_needed(self, image_depth_without_table):

        # In this part of the function we calculate the mean height inside the designated box, we then compare this
        # mean height to value determined empirically to decide if the box is still full or empty

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
    pc = PerspectiveCalibration('/common/save/calibration/camera/camera_data')
    IBC = InBoxCoord(pc)
    IBC.init_pick_and_place_boxes()
    rospy.spin()


