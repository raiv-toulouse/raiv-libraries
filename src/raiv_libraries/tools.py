from pathlib import Path
from datetime import datetime
import cv2
import numpy as np
from cv_bridge import CvBridge
from raiv_libraries.image_tools import ImageTools
from raiv_libraries.robotUR import RobotUR
import geometry_msgs.msg as geometry_msgs

# Ajout pour un test
#TODO : Move in a constant file
THRESHOLD_ABOVE_TABLE = 10  # Used to select all the pixels above the table
BIG_CROP_WIDTH = 100  # Crop a big image to be able to perform rotations before final small crop
BIG_CROP_HEIGHT = 100


bridge = CvBridge()

def create_rgb_depth_folders(parent_folder):
    """
    Create (if they don't exist) the following folders :
    rgb / success; rgb / fail; depth / success; depth / fail
    """
    for rd_folder in ['rgb', 'depth']:
        for sf_folder in ['success', 'fail']:
            folder = parent_folder / rd_folder / sf_folder
            Path.mkdir(folder, parents=True, exist_ok=True)
            folder.chmod(0o777)  # Write permission for everybody

def save_pil_images(parent_folder, success_or_fail, rgb_images_pil, depth_images_pil):
    """
    Save 2 collections of PIL images in folders named :
    * <parent_folder>/rgb/<'success' or 'fail'>/<date>.png
    * <parent_folder>/depth/<'success' or 'fail'>/<date>.png
    """
    image_name_prefix = str(datetime.now())
    for image_type, images_pil in zip(['rgb', 'depth'], [rgb_images_pil, depth_images_pil]):
        for ind, image_pil in enumerate(images_pil):
            image_path = (parent_folder / image_type / success_or_fail / (image_name_prefix + '_' + str(ind) + '.png')).resolve()
            image_pil.save(str(image_path))
            image_path.chmod(0o777) # Write permission for everybody

def generate_and_save_rgb_depth_images(resp_pick, parent_image_folder, is_object_gripped):

    rgb_images_pil = []
    depth_images_pil = []
    pil_rgb = ImageTools.ros_msg_to_pil(resp_pick.rgb_crop)

    depth_crop_cv = bridge.imgmsg_to_cv2(resp_pick.depth_crop, desired_encoding='passthrough')

    # Calculate the histogram of the depth image
    histogram = cv2.calcHist([depth_crop_cv], [0], None, [1000], [1, 1000])
    # Take the index with the maximum values (i.e. the value of the table's distance to the camera) e
    # Every pixel with a value under the table value +BOX_ELEVATION milimeters is set to zero.
    distance_camera_to_table = histogram.argmax()
    image_depth_without_table = np.where(depth_crop_cv == 0, distance_camera_to_table, depth_crop_cv)
    alt_pt_plus_haut = np.min(image_depth_without_table)
    image_depth_without_table = np.where(image_depth_without_table >= distance_camera_to_table - THRESHOLD_ABOVE_TABLE, distance_camera_to_table,
                                         image_depth_without_table)

    cv2.normalize(image_depth_without_table, image_depth_without_table, 0, 255, cv2.NORM_MINMAX)
    image_depth_without_table = np.round(image_depth_without_table).astype(np.uint8)
    pil_depth = ImageTools.numpy_to_pil(image_depth_without_table)

    # Generate a set of images with rotation transform
    for deg in range(0, 360, 10):
        rgb_images_pil.append(ImageTools.center_crop(pil_rgb.rotate(deg), ImageTools.CROP_WIDTH, ImageTools.CROP_HEIGHT))
        depth_images_pil.append(ImageTools.center_crop(pil_depth.rotate(deg), ImageTools.CROP_WIDTH, ImageTools.CROP_HEIGHT))

    save_pil_images(parent_image_folder, 'success' if is_object_gripped else 'fail', rgb_images_pil, depth_images_pil) # Save images in success folders

def xyz_to_pose(x, y, z):
    return geometry_msgs.Pose(geometry_msgs.Vector3(x, y, z), RobotUR.tool_down_pose)
