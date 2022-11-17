from pathlib import Path


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