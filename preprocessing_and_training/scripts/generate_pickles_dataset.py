"""
Description:
----------
This script will generate a picke file for each valid image in a given dataset, keeping folders structure and filenames
so we can infere the image path using the pickle path

Example:
-for:
/data_volume/dataset/images/mini_one_2018/23655_9_e452da2f-945a-4a6c-bcfd-822d26977093_921b97ef-f067-45fd-bfaa-579e89e1b426.jpg
-we will generate:
/data_volume/dataset/pickled_images/mini_one_2018/23655_9_e452da2f-945a-4a6c-bcfd-822d26977093_921b97ef-f067-45fd-bfaa-579e89e1b426.pkl

Pickle file will contain the bounding box coordinates of the vehicle
"""

import os 
import cv2
import pickle 
import argparse
import torch
from tqdm import tqdm
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from torchvision.ops import box_area

DETECTRON_CAR_CODE = 2
DETECTRON_TRUCK_CODE = 7
DETECTRON_BUS_CODE = 5
AREA_TRESHHOLD = 0.2

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")
DET_MODEL = DefaultPredictor(cfg)

def get_car_bbox(img):
    """
    Parameters
    ----------
    Receives an image and returns the bounding box coordinates of the car if exists.
    Otherwise, returns None.

    RETURNS: 
        box_coordinates if car exists, None otherwise.
    """
    outputs = DET_MODEL(img)
    cars_trucks_buses = (outputs["instances"].pred_classes == DETECTRON_CAR_CODE) | (outputs["instances"].pred_classes == DETECTRON_BUS_CODE |
                        (outputs["instances"].pred_classes == DETECTRON_TRUCK_CODE))
    car_trucks_boxes = outputs["instances"][cars_trucks_buses]

    if(outputs["instances"][cars_trucks_buses].pred_classes.size()[0] != 0):
        vehicle_areas = box_area(outputs["instances"][cars_trucks_buses].pred_boxes.tensor)
        max_arg_index = vehicle_areas.argmax().item()
        max_frame_area = vehicle_areas[max_arg_index].item()
        full_pic_area = img.shape[0] * img.shape[1]
        bbox_ratio = max_frame_area / full_pic_area
        box_coordinates = car_trucks_boxes.pred_boxes.tensor[max_arg_index].to(torch.int).tolist()
        if(bbox_ratio > AREA_TRESHHOLD):
            return box_coordinates   
    return None

def parse_args():
    parser = argparse.ArgumentParser(description="Generate pickles for valid images in a dataset")
    parser.add_argument(
        "data_folder",
        type=str,
        help=(
            "Full path to the directory having all the cars images. Already "
            "orgenized in folders by class."
            "E.g. `/data_volume/dataset/images`."
        ),
    )
    parser.add_argument(
        "output_data_folder",
        type=str,
        help=(
            "Full path to the directory in which we will store the resulting "
            "pickles. E.g. `/home/app/src/pickled_images/`."
        ),
    )
    args = parser.parse_args()

    return args


def main(data_folder, output_data_folder):
    """
    Parameters
    ----------
    data_folder : str
        Full path to Raw images Dataset folder.

    output_data_folder : str
        Full path to the directory in which we will store the resulting
        valid car/truck pickles.
    """
    for root, dirs, files in os.walk(data_folder):
        for file in tqdm(files):
            output_path = os.path.join(output_data_folder,root.split('/')[-1])
            if not os.path.exists(output_path):
                os.makedirs(output_path, mode=0o777)
            image_path = os.path.join(root, file)
            image = cv2.imread(image_path)
            if image is None:
                print('Corrupted image:', image_path)
                continue
            vehicle_coordinates = get_car_bbox(image)
            pickle_path = os.path.splitext(file)[0]+'.pkl'
            full_pickle_path = os.path.join(output_path, pickle_path)
            print("###################################")
            print("File:", file)
            print("Output path:", full_pickle_path)
            if vehicle_coordinates:
                with open(full_pickle_path, 'wb') as f:
                    pickle.dump(vehicle_coordinates, f)


if __name__ == "__main__":
    args = parse_args()
    main(args.data_folder, args.output_data_folder)
