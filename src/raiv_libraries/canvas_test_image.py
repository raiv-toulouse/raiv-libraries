from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
import numpy as np
from PIL import Image
from cv_bridge import CvBridge, CvBridgeError


class canvas_test_image(QLabel):

    def __init__(self,parent):
        super().__init__(parent)
        self.parent = parent
        self.image = None

    def set_image(self, img):
        """ called by parent widget to specify a new image to display """
        # Below code to display a 16 bits gray image in a QImage
        img_8bit = (img / 256.0).astype(np.uint8) # use the high 8bit
        img_8bit = ((img - img.min()) / (img.ptp() / 255.0)).astype(np.uint8) # map the data range to 0 - 255
        self.image = QImage(img_8bit.repeat(4), 640, 480, QImage.Format_RGB32)
        # Below code for RGB image
        #img = Image.fromarray(img)
        #self.image = QImage(img.tobytes("raw", "RGB"), img.width, img.height, QImage.Format_RGB888)  # Convert PILImage to QImage
        self.setMinimumSize(self.image.width(), self.image.height())
        self.update()

    def mousePressEvent(self, event):
        """ when we click on the image of this canvas, call the process_click method in test_image.py """
        pos = event.pos()
        self.parent.process_click(pos.x(), pos.y()) # ask the create_image_dataset object to send the robot to this position

    def paintEvent(self, event):
        """ Use to draw the image"""
        if self.image:
            qp = QPainter(self)
            rect = event.rect()
            qp.drawImage(rect, self.image, rect)
