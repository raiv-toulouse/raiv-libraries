#!/usr/bin/env python

import numpy as np
import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from raiv_libraries.srv import get_coordservice, get_coordserviceResponse
import math
import random


class In_box_coord:

    def __init__(self):
        rospy.init_node('In_box_coord')

        self.image_rgb = None
        self.image_depth = None

        self.rgb_crop = None
        self.depth_crop = None

        self.image_width = 640
        self.image_height = 480

        self.Histogram = None

        self.pub = rospy.Publisher('Point_in_box', Image, queue_size=1)

        self.BoxGauche = None
        self.AngleGauche = 0

        self.BoxDroite = None
        self.AngleDroite = 0

        self.BoxActive = None
        self.AngleActif = None

        self.BoxInactive = None
        self.AngleInactif = None

        self.r = rospy.Rate(0.3)

        self.background_index = 0

        self.crop_height = 60
        self.crop_width = 60

        self.image_BoxGauche = None
        self.image_BoxDroite = None

        self.xpick = 0
        self.ypick = 0
        self.xplace= 563
        self.yplace= 226

        print('Class Initiated')

        self.RefreshImageAndDistance()
        self.RemoveBackground()

        self.distance = rospy.wait_for_message('/Distance_Here', Image)
        self.distance = CvBridge().imgmsg_to_cv2(self.distance, desired_encoding='16UC1')
        self.Check_Emptyness(self.BoxGauche, self.distance)

        self.RandProcess()

        service = rospy.Service('/In_box_coordService', get_coordservice, self.ProcessCoord)
        rospy.spin()

    def RandProcess(self):
        self.distance = rospy.wait_for_message('/Distance_Here', Image)
        self.distance = CvBridge().imgmsg_to_cv2(self.distance, desired_encoding='16UC1')
        self.Check_Emptyness(self.BoxGauche, self.distance)
        print('Emptyness check Done')
        self.rand_point(self.BoxActive, self.AngleActif)
        print('Rand_point Done')
        # self.rand_point(self.BoxInactive, self.AngleInactif)
        self.Images_crop()



    def RefreshImageAndDistance(self):

        self.image_rgb = rospy.wait_for_message('/RGBClean', Image)

        self.distance = rospy.wait_for_message('/Distance_Here', Image)

        self.image_rgb = CvBridge().imgmsg_to_cv2(self.image_rgb, desired_encoding='bgr8')

        self.distance = CvBridge().imgmsg_to_cv2(self.distance, desired_encoding='16UC1')


    def RemoveBackground(self):

        self.Histogram = cv2.calcHist([self.distance], [0], None, [1000], [1,1000])

        np.set_printoptions(threshold=np.inf)

        self.background_index = self.Histogram.argmax()

        self.distance = np.where(self.distance <= self.background_index-3, self.distance, 0)

        self.GetContour(self.distance)


    def GetContour(self, image):

        image = np.divide(image, np.amax(image))
        image = image * 255

        image = np.array(image, dtype = np.uint8)

        contours, hierarchy = cv2.findContours(image,1,2)

        imageRGB = cv2.merge((image, image, image))

        cntlist = []

        for cnt in contours:
            Box = cv2.minAreaRect(cnt)
            Box = cv2.boxPoints(Box)
            Box = np.int0(Box)
            cv2.drawContours(imageRGB, [Box], 0, (0, 255, 255), 5)
            cntlist.append((cnt,cv2.contourArea(cnt)))

        cntlist = sorted(cntlist, key=lambda x:x[1])

        Box1Contour = cntlist[-1][0]
        Box2Contour = cntlist[-2][0]

        Box1 = cv2.minAreaRect(Box1Contour)
        AngleBox1 = Box1[-1]
        Box2 = cv2.minAreaRect(Box2Contour)
        AngleBox2 = Box2[-1]

        Box1 = cv2.boxPoints(Box1)
        Box2 = cv2.boxPoints(Box2)

        Box1 = np.int0(Box1)
        Box2 = np.int0(Box2)

        CompteurBox1 = 0
        CompteurBox2 = 0

        for pt in Box1:
            if pt[0] < self.image_width /2:
                CompteurBox1 +=1

        for pt in Box2:
            if pt[0] < self.image_width /2:
                CompteurBox2 +=1

        if CompteurBox1 > CompteurBox2:
            self.BoxGauche = Box1
            self.AngleGauche = AngleBox1
            self.BoxDroite = Box2
            self.AngleDroite = AngleBox2
        else:
            self.BoxGauche = Box2
            self.AngleGauche = AngleBox2
            self.AngleDroite = AngleBox1
            self.BoxDroite = Box1

        cv2.drawContours(imageRGB, [self.BoxGauche], 0, (0, 0, 255), 3)
        cv2.drawContours(imageRGB, [self.BoxDroite], 0, (255, 0, 0), 3)

    def Check_Emptyness(self, Box, image):
        self.RefreshImageAndDistance()

        mask = np.zeros((self.image_height,self.image_width,1), np.uint8)

        cv2.drawContours(mask,[Box], 0, (1), -1)

        element = cv2.getStructuringElement(0, (2 * 20 + 1, 2 * 20 + 1),(20, 20))
        mask = cv2.erode(mask, element)

        image = cv2.bitwise_and(image, image, mask = mask )

        Hist = cv2.calcHist([image], [0], mask, [999], [1,1000])
        mean = 0
        pool = 1

        for idx, val in enumerate(Hist):
            if int(val) != 0:
                mean += int(idx)*int(val)
                pool += val
        print(pool)
        mean = mean/pool

        self.BoxActive = self.BoxGauche
        self.AngleActif = self.AngleGauche

        self.BoxInactive = self.BoxDroite
        self.AngleInactif = self.AngleDroite

        if mean < self.background_index - 20 and Box is self.BoxGauche:
            print('La boite gauche est celle active')
            self.BoxActive = self.BoxGauche
            self.AngleActif = self.AngleGauche
            self.BoxInactive = self.BoxDroite
            self.AngleInactif = self.AngleDroite

        elif mean < self.background_index - 20 and Box is self.BoxDroite:
            print('La boite droite est celle active')
            self.BoxActive = self.BoxDroite
            self.AngleActif = self.AngleDroite
            self.BoxInactive = self.BoxGauche
            self.AngleInactif = self.AngleGauche

        elif mean >= self.background_index - 20 and Box is self.BoxGauche:
            print('La boite droite est celle active')
            self.BoxActive = self.BoxDroite
            self.AngleActif = self.AngleDroite
            self.BoxInactive = self.BoxGauche
            self.AngleInactif = self.AngleGauche

        elif mean >= self.background_index - 20 and Box is self.BoxDroite:
            print('La boite droite est celle active')
            self.BoxActive = self.BoxGauche
            self.AngleActif = self.AngleGauche
            self.BoxInactive = self.BoxDroite
            self.AngleInactif = self.AngleDroite
        else:
            print('Box inactive : ', self.BoxInactive)
            print('mean : ', mean)
            print('self background -20 : ', self.background_index - 20)
            print(Box)
            print(self.BoxGauche)
            print(self.BoxDroite)
            print('Kleines probleme meine kommandant')

    # def scale_contour(cnt, scale):
    #     M = cv2.moments(cnt)
    #     cx = int(M['m10'] / M['m00'])
    #     cy = int(M['m01'] / M['m00'])
    #
    #     cnt_norm = cnt - [cx, cy]
    #     cnt_scaled = cnt_norm * scale
    #     cnt_scaled = cnt_scaled + [cx, cy]
    #     cnt_scaled = cnt_scaled.astype(np.int32)
    #
    #     return cnt_scaled

    def rand_point(self, box, angle):

        O_I = int(math.sqrt((box[-1][0]-box[2][0])**2+(box[-1][1]-box[2][1])**2))
        OI = int(math.sqrt((box[-1][0]-box[0][0])**2+(box[-1][1]-box[0][1])**2))

        if O_I < OI:
            Beta = 90-angle
            PtRef = box[0]
            Largeur = O_I
            Longueur = OI
        else:
            Beta = -angle
            PtRef = box[1]
            Largeur = OI
            Longueur = O_I

        x = random.randrange(0, Largeur)
        y = random.randrange(0, Longueur)
        print(Beta)
        x2 = int(y*math.sin(math.pi/180*Beta) + x*math.cos(math.pi/180*Beta))
        y2 = int(y*math.cos(math.pi/180*Beta) - x*math.sin(math.pi/180*Beta))

        x2 = x2 + int(PtRef[0])
        y2 = y2 + int(PtRef[1])

        if box is self.BoxActive:
            self.xpick = x2
            self.ypick = y2
        if box is self.BoxInactive:
            self.xplace = x2
            self.yplace = y2

    def Images_crop(self):

        x = int(self.xpick)
        y = int(self.ypick)
        self.rgb_crop = self.image_rgb[int(y - self.crop_width/2):int(y + self.crop_width/2),int(x - self.crop_height/2):int(x + self.crop_height/2)]
        self.depth_crop = self.distance[int(y - self.crop_width/2):int(y + self.crop_width/2),int(x - self.crop_height/2):int(x + self.crop_height/2)]

    def ProcessCoord(self, req):
        bridge = CvBridge()


        self.crop_height = req.height
        self.crop_width = req.width

        if req.mode == 'refresh':
            self.RefreshImageAndDistance()
            return get_coordserviceResponse(
                rgb=None,
                depth=None,
                xpick=None,
                ypick=None,
                xplace=None,
                yplace=None,
            )
        if req.mode == 'random':
            bridge = CvBridge()
            self.RandProcess()

            self.rgb_crop = bridge.cv2_to_imgmsg(self.rgb_crop, encoding ='passthrough')

            self.depth_crop = bridge.cv2_to_imgmsg(self.depth_crop, encoding ='passthrough')

            # cv2.namedWindow("Supervision Debug", cv2.WINDOW_NORMAL)
            # cv2.drawContours(self.image_rgb, [self.BoxGauche], 0, (0, 0, 255), 3)
            # cv2.drawContours(self.image_rgb, [self.BoxDroite], 0, (255, 0, 0), 3)
            # cv2.circle(self.image_rgb, (self.xpick, self.ypick), 5, (255, 0, 0), -1)
            # cv2.circle(self.image_rgb, (self.xplace, self.yplace), 5, (0, 0, 255), -1)
            # cv2.imshow('Supervision Debug', self.image_rgb)
            # cv2.waitKey(5000)
            # cv2.destroyWindow('Supervision Debug')
            return get_coordserviceResponse(
                rgb =self.rgb_crop,
                depth = self.depth_crop,
                xpick = self.xpick,
                ypick = self.ypick,
                xplace =self.xplace,
                yplace =self.yplace,
            )

IBC = In_box_coord()