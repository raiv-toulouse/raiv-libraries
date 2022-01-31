#!/usr/bin/env python

import numpy as np
import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from raiv_libraries.srv import get_coordservice, get_coordserviceResponse
import math
import random

class InBoxCoord:

    def __init__(self):

        #Declaration of nodes and global variables

        rospy.init_node('In_box_coord')

        self.image_rgb = None
        self.image_depth = None

        self.rgb_crop = None
        self.depth_crop = None

        self.image_width = 640
        self.image_height = 480

        self.histogram = None
        self.req = None
        self.pub = rospy.Publisher('Point_in_box', Image, queue_size=1)

        self.leftbox = None
        self.angleleft = 0

        self.rightbox = None
        self.angleright = 0

        self.activebox = None
        self.activeangle = None

        self.inactivebox = None
        self.inactiveangle = None

        self.r = rospy.Rate(0.3)

        self.background_index = 0

        self.crop_height = 30
        self.crop_width = 30

        self.image_leftbox = None
        self.image_rightbox = None

        self.xpick = 0
        self.ypick = 0
        self.xplace = 563
        self.yplace = 226

        self.refresh_image_and_distance()
        self.remove_background()

        self.distance = rospy.wait_for_message('/Distance_Here', Image)
        self.distance = CvBridge().imgmsg_to_cv2(self.distance, desired_encoding='16UC1')

        self.activebox = self.leftbox
        self.activeangle = self.angleleft

        self.inactivebox = self.rightbox
        self.inactiveangle = self.angleright

        #As a start, we check if the left box is empty or full, this is mainly used to define all the variables, you should always
        #start with the left box full and the right one empty
        self.check_emptyness(self.activebox, self.distance)
        service = rospy.Service('/In_box_coordService', get_coordservice, self.process_coord)
        rospy.spin()

    def rand_process(self):
        self.distance = rospy.wait_for_message('/Distance_Here', Image)
        self.distance = CvBridge().imgmsg_to_cv2(self.distance, desired_encoding='16UC1')
        self.check_emptyness(self.activebox, self.distance)
        self.rand_point(self.activebox, self.activeangle)
        self.rand_point(self.inactivebox, self.inactiveangle)
        self.images_crop()

    # Function to refresh the RGB and Depth image
    def refresh_image_and_distance(self):

        self.image_rgb = rospy.wait_for_message('/RGBClean', Image)

        self.distance = rospy.wait_for_message('/Distance_Here', Image)

        self.image_rgb = CvBridge().imgmsg_to_cv2(self.image_rgb, desired_encoding='bgr8')

        self.distance = CvBridge().imgmsg_to_cv2(self.distance, desired_encoding='16UC1')

    #Calculate the histogram of the depth image and take the max value (i.e. the value of the table) every pixel with a value
    #under the table value +10 milimeters is set to zero. Then we obtain clear contours of the boxes
    def remove_background(self):

        self.histogram = cv2.calcHist([self.distance], [0], None, [1000], [1, 1000])
        self.background_index = self.histogram.argmax()
        self.distance = np.where(self.distance <= self.background_index-10, self.distance, 0)
        self.get_contour(self.distance)

    #Function used to get the contours of the two boxes
    def get_contour(self, image):

        image = np.divide(image, np.amax(image))
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

        cv2.drawContours(imagergb, [self.leftbox], 0, (0, 0, 255), 3)
        cv2.drawContours(imagergb, [self.rightbox], 0, (255, 0, 0), 3)

    # Determine if the active box is empty, if so, the active box becomes the inactive one and the inactive box becomes the active one
    def check_emptyness(self, box, image):
        self.refresh_image_and_distance()

        #In this part of the function we calculate the mean height inside the designated box, we then compare this
        #mean height to value determined empirically to decide if th ebox is still full or empty
        mask = np.zeros((self.image_height, self.image_width, 1), np.uint8)

        cv2.drawContours(mask, [box], 0, 255, -1)
        element = cv2.getStructuringElement(0, (2 * 20 + 1, 2 * 20 + 1), (20, 20))
        mask = cv2.erode(mask, element)

        image = cv2.bitwise_and(image*255, image*255, mask=mask)
        hist = cv2.calcHist([image], [0], mask, [999], [1, 1000])

        mean = 0
        pool = 1

        dh = 0

        for idx, val in enumerate(hist):
            if int(val) != 0:
                mean += int(idx)*int(val)
                pool += val
        mean = mean/pool

        # Change the value of the height mean depending on which is the active box, this is due to the unequal layer of cork in the boxes

        if self.activeangle is self.angleleft:
            dh = 25

        elif self.activeangle is self.angleright:
            dh = 29

        print('self background index : ', self.background_index)
        print('self background index - 24 : ', self.background_index - 24)
        print('mean : ', mean)

        #In this part of the function we change global variables depending on the result of our mean height comparaison

        if mean < self.background_index - dh and self.activeangle is self.angleleft:
            print('La boite gauche est celle active')
            self.activebox = self.scale_contour(self.leftbox, 0.8)
            self.activeangle = self.angleleft
            self.inactivebox = self.scale_contour(self.rightbox, 0.5)
            self.inactiveangle = self.angleright

        elif mean < self.background_index - dh and self.activeangle is self.angleright:
            print('La boite droite est celle active')
            self.activebox = self.scale_contour(self.rightbox, 0.8)
            self.activeangle = self.angleright
            self.inactivebox = self.scale_contour(self.leftbox, 0.5)
            self.inactiveangle = self.angleleft

        elif mean >= self.background_index - dh and self.activeangle is self.angleleft:
            print('La boite droite est celle active')
            self.activebox = self.scale_contour(self.rightbox, 0.8)
            self.activeangle = self.angleright
            self.inactivebox = self.scale_contour(self.leftbox, 0.5)
            self.inactiveangle = self.angleleft

        elif mean >= self.background_index - dh and self.activeangle is self.angleright:
            print('La boite gauche est celle active')
            self.activebox = self.scale_contour(self.leftbox, 0.8)
            self.activeangle = self.angleleft
            self.inactivebox = self.scale_contour(self.rightbox, 0.5)
            self.inactiveangle = self.angleright
        else:
            print('box inactive : ', self.inactivebox)
            print('self background -20 : ', self.background_index - 20)
            print('Kleines probleme meine kommandant')

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

    #Generate random point inside the active box contour
    def rand_point(self, box, angle):
        #This part of the code allows us to know what is the angle we are given by OpenCV
        #
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

        if box is self.activebox:
            self.xpick = x2
            self.ypick = y2
        if box is self.inactivebox:
            self.xplace = x2
            self.yplace = y2

        if not self.distance[y2][x2] in range(1, self.background_index-3):
            print(f'Generating another randpoint due to bad value of depth: {self.distance[y2][x2]}')
            self.rand_point(box, angle)
    #Crop an image around the selected point
    def images_crop(self, state=0):
        if state == 0:
            x = int(self.xpick)
            y = int(self.ypick)
        if state == 1:
            x = self.req.x
            y = self.req.y
        self.rgb_crop = self.image_rgb[
                        int(y - self.crop_width/2):int(y + self.crop_width/2),
                        int(x - self.crop_height/2):int(x + self.crop_height/2)
                        ]
        self.depth_crop = self.distance[
                          int(y - self.crop_width/2):int(y + self.crop_width/2),
                          int(x - self.crop_height/2):int(x + self.crop_height/2)
                          ]

        self.depth_crop = np.where(self.depth_crop == 0,self.background_index, self.depth_crop)
        print(type(self.rgb_crop))

    #Treat the request received by the service
    def process_coord(self, req):
        self.req = req
        self.crop_height = req.height
        self.crop_width = req.width

        #Mode 'refresh', doesn't generate anything, just refresh the RGB and depth image
        if self.req.mode == 'refresh':
            self.refresh_image_and_distance()

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
        if self.req.mode == 'random':
            bridge = CvBridge()
            self.rand_process()
            self.rgb_crop = bridge.cv2_to_imgmsg(self.rgb_crop, encoding='passthrough')

            self.depth_crop = bridge.cv2_to_imgmsg(self.depth_crop, encoding='passthrough')

            return get_coordserviceResponse(
                rgb=self.rgb_crop,
                depth=self.depth_crop,
                xpick=self.xpick,
                ypick=self.ypick,
                xplace=self.xplace,
                yplace=self.yplace,
                hist_max=self.background_index
            )
        #Mode "fixed", used to generate a crop around a non-random point given in the request
        if self.req.mode == 'fixed':
            bridge = CvBridge()
            self.rand_process()
            self.images_crop(state = 1)
            self.rgb_crop = bridge.cv2_to_imgmsg(self.rgb_crop, encoding='passthrough')
            self.depth_crop = bridge.cv2_to_imgmsg(self.depth_crop, encoding='passthrough')
            return get_coordserviceResponse(
                rgb=self.rgb_crop,
                depth=self.depth_crop,
                xpick=None,
                ypick=None,
                xplace=None,
                yplace=None,
                hist_max=None
            )

IBC = InBoxCoord()
