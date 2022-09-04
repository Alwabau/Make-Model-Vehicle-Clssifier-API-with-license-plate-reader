"""
This script will be used to analize input images with Detectron2 and get the 
result saved to a pickle file.
We create a new folder (2nd argument) to store this files, following exactly the same 
directory structure with its subfolders and file names but with pickle 
extension .pkl.

Parameters
----------
data_folder : str
    Full path to the directory having all the cars images.

output_data_folder : str
    Full path to the directory in which we will store the resulting
    pickle files.    
"""

import cv2
import os
import pickle
import argparse
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from torchvision.ops import box_area
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Train your model.")
    parser.add_argument(
        "data_folder",
        type=str,
        help=(
            "Full path to the directory having all the cars images already. "
            "selected from the entire dataset to be the sample dataset"
            "E.g. `/data_volume/dataset/sample_dataset_images/`."
        ),
    )
    parser.add_argument(
        "output_data_folder",
        type=str,
        help=(
            "Full path to the directory in which we will store the resulting "
            "pckle files. E.g. `/data_volume/dataset/pickled_images`."
        ),
    )

    args = parser.parse_args()

    return args

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")
DET_MODEL = DefaultPredictor(cfg)

def main(data_folder, output_data_folder):
    """
    Parameters
    ----------
    data_folder : str
        Full path to the directory having all the cars images.

    output_data_folder : str
        Full path to the directory in which we will store the resulting
        pickle files.
    """
    for root, _, files in tqdm(list(os.walk(data_folder))):
        for file in files:
            output_path = os.path.normpath(os.path.join(output_data_folder, os.path.relpath(root, data_folder)))
            os.makedirs(output_path, exist_ok=True)
            image_path = os.path.join(root, file)
            image = cv2.imread(image_path)
            if image is None:
                continue
            object_detecron2 = DET_MODEL(image)
            pickle_path = os.path.splitext(file)[0]+'.pkl'
            full_pickle_path = os.path.join(output_path, pickle_path)
            if object_detecron2:
                with open(full_pickle_path, 'wb') as f:
                    pickle.dump(object_detecron2, f)
            
if __name__ == "__main__":
    args = parse_args()
    main(args.data_folder, args.output_data_folder)