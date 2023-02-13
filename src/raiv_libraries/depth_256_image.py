import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import cv2
import numpy as np

TABLE_DISTANCE = 540
MIN = 195

class Depth256Image():
    def __init__(self):
        self.publisher = rospy.Publisher('/depth_256_image', Image, queue_size=10)
        self.bridge = CvBridge()
        rospy.init_node('depth_256_image_node', anonymous=True)
        rospy.Subscriber('/camera/aligned_depth_to_color/image_raw', Image, self.convert_to_256)
        rospy.spin()

    def convert_to_256(self, msg_depth):
        depth_cv = self.bridge.imgmsg_to_cv2(msg_depth, desired_encoding='16UC1')
        depth_cv = np.where(depth_cv > TABLE_DISTANCE, TABLE_DISTANCE, depth_cv) # All values under the table are changed to table distance
        depth_cv = np.where(depth_cv <= TABLE_DISTANCE - 255, TABLE_DISTANCE - 255, depth_cv) # Keep only values in [TABLE_DISTANCE-255, TABLE_DISTANCE]
        depth_cv-= TABLE_DISTANCE - 255  # All values are now in [0,255]
        depth_cv = np.where(depth_cv == 0, 255, depth_cv) # Suppress all black points, they become white points (table points)
        a = 255 / (255 - MIN)  # Compute a and b for y = a.x + b to convert image with low contrast to image with high contrast
        b = -a * MIN
        depth_cv = a * depth_cv + b  # Image with high contrast
        depth_cv = depth_cv.astype('uint8')  # Convert to GRAYSCALE 8bits image
        depth_rgb_cv = cv2.cvtColor(depth_cv,cv2.COLOR_GRAY2BGR)
        msg_depth_256 = self.bridge.cv2_to_imgmsg(depth_rgb_cv, encoding="rgb8")
        self.publisher.publish(msg_depth_256)


if __name__ == '__main__':
    try:
        depth_image = Depth256Image()
    except rospy.ROSInterruptException:
        pass