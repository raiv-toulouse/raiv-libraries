from pathlib import Path
from datetime import datetime
import cv2
from cv_bridge import CvBridge
from raiv_libraries.image_tools import ImageTools

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

def save_images(parent_folder, success_or_fail, rgb_images_pil, depth_images_pil):
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