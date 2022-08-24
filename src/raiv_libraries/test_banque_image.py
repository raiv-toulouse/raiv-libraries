import cv2
from pathlib import Path
import subprocess,os

nb_fail = 250
nb_success = 250
path_src = Path('/common/stockage_banque_image/h_68cm/0.5_souflet/pb')
path_dst = Path('/common/stockage_banque_image/h_68cm/0.5_souflet/250im_v2')


def copier_fichiers(rep_src, rep_dst, nb):
    # Copie des fichiers
    pathlist = list(rep_src.glob('*.png'))
    list_of_files = ''
    for count, path in enumerate(pathlist):
        if count == nb:
            #subprocess.run(["cp", list_of_files, str(rep_dst)])
            cmd = 'cp ' + list_of_files + ' ' + str(rep_dst)
            os.system(cmd)
            break
        list_of_files += "'" + str(path) + "' "

# Suppression des répertoires
subprocess.run(["rm", "-rf", str(path_dst)])

# Création des repertoires
path_dst.mkdir()
for rep_dr in ["depth", "rgb"]:
    (path_dst / rep_dr).mkdir()
    for rep_fs in ["fail", "success"]:
        (path_dst / rep_dr / rep_fs).mkdir()

# Copie des fichiers
copier_fichiers(path_src / "depth" / "fail", path_dst / "depth" / "fail", nb_fail)
copier_fichiers(path_src / "rgb" / "fail", path_dst / "rgb" / "fail", nb_fail)
copier_fichiers(path_src / "depth" / "success", path_dst / "depth" / "success", nb_success)
copier_fichiers(path_src / "rgb" / "success", path_dst / "rgb" / "success", nb_success)
print('fin')
