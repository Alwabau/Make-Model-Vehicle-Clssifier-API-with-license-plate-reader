'''
This script loads a dataframe with filepaths of unprocessed files.
Then it will process a threshold amount of files per class.
It will run detectron and mobilenet and if its a car exterior image will crop 
it and save it in training data folder.
'''

import os
import cv2
import torch
import pandas as pd
from tqdm import tqdm
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from torchvision.ops import box_area
from tensorflow.keras.utils import img_to_array
from tensorflow.keras.preprocessing.image import smart_resize
from utils import utils
from mobilenet_model import mobilenetv3_model



CONFIG_YML = "/data_volume/alan/ay22-01-final-project-5/mobilenet_weights/w_002/config_mobilenet7.yml"
WEIGHTS = "/data_volume/alan/ay22-01-final-project-5/alan/ay22-01-final-project-5/mobilenet_weights/w_007/model.77-0.1488.h5"

CONFIG = utils.load_config(CONFIG_YML)
INPUT_SIZE=CONFIG["data"]["image_size"]
MODEL_CLASSES = utils.get_class_names(CONFIG)

DETECTRON_CAR_CODE = 2
DETECTRON_TRUCK_CODE = 7
DETECTRON_BUS_CODE = 5
AREA_TRESHHOLD = 0.2

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")

UNPROCESSED_FILES_DF = "/home/app/src/Notebooks/unprocessed_files.csv"

TRAIN_FOLDER = "/data_volume/dataset/sample_classifier_images/train"

THRESHOLD = 300



def get_car_bbox(img, model):
    '''
    Gets the car bounding box.
    '''
    outputs = model(img)
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

def is_exterior(image, model):
    """
    Returns True if the image is exterior.
    """
    array_image = img_to_array(image)
    resized_image = smart_resize(array_image, INPUT_SIZE)
    reshaped_image = resized_image.reshape([1, 224, 224, 3])
    all_predictions = model.predict(reshaped_image)
    prediction_argmax = all_predictions.argmax()

    return prediction_argmax == 0

def get_bboxes_df(df):
    '''
    Get a dataframe with the bounding boxes
    '''
    detectron_model = DefaultPredictor(cfg)
    cols = ["make", "model", "file_name", "original_path", "bbox"]
    out_df = pd.DataFrame(columns=cols)
    for index, row in tqdm(df.iterrows(), total=len(df)):
        image = cv2.imread(row["original_path"])
        if image is None:
            print('Corrupted image:', row["original_path"])
            continue
        bbox = get_car_bbox(image, detectron_model)
        if bbox is not None:
            out_df = pd.concat([out_df, pd.DataFrame([[row["make"], row["model"], row["file_name"], row["original_path"], bbox]], columns=cols)])
    return out_df

def get_car_exterior_df(df):
    '''
    Creates a car exterior dataframe with the "make", "model", "file_name", "original_path", 
    and "bbox".
    '''
    model = mobilenetv3_model.create_model(weights=WEIGHTS)
    cols = ["make", "model", "file_name", "original_path", "bbox"]
    out_df = pd.DataFrame(columns=cols)
    for index, row in tqdm(df.iterrows(), total=len(df)):
        image = cv2.imread(row["original_path"])
        if image is None:
            print('Corrupted image:', row["original_path"])
            continue
        if is_exterior(image, model):
            out_df = pd.concat([out_df, pd.DataFrame([[row["make"], row["model"], row["file_name"], row["original_path"], row['bbox']]], columns=cols)])
    return out_df

def cut_and_save_images(df):
    '''
    Uses the bounding boxes to cut the images and saves them.
    '''
    cols = ["make", "model", "original_path"]
    class_generated_images_df = pd.DataFrame(columns = cols)
    for index, row in df.iterrows():
        image = cv2.imread(row["original_path"])
        if image is None:
            print('Corrupted image:', row["original_path"])
            continue
        bbox = row["bbox"]
        x1 = bbox[0]
        y1 = bbox[1]
        x2 = bbox[2]
        y2 = bbox[3]
        cropped_image = image[y1:y2, x1:x2]
        class_name = row["make"] + "_" + row["model"]
        out_path = os.path.join(TRAIN_FOLDER, class_name, row["file_name"])
        os.makedirs(os.path.dirname(out_path), exist_ok=True,  mode=0o777)
        class_generated_images_df = pd.concat([class_generated_images_df, pd.DataFrame([[row["make"], row["model"],row["original_path"]]], columns=cols)])
        cv2.imwrite(out_path, cropped_image)
    return class_generated_images_df



unprocessed_files = pd.read_csv(UNPROCESSED_FILES_DF, index_col=0)

make_models_list = unprocessed_files.groupby(['make','model']).size().reset_index(name='amount').values.tolist()

cols = ["make", "model", "original_path"]
generated_images_df = pd.DataFrame(columns = cols)

# Generating a dataframe with the created images and classes
for item in make_models_list:
    make_model_df = unprocessed_files.loc[(unprocessed_files.make == item[0]) & (unprocessed_files.model == item[1])]
    
    if len(make_model_df) > THRESHOLD:
        make_model_df = make_model_df.sample(THRESHOLD)
    else:
        make_model_df = make_model_df.sample(len(make_model_df))  

    make_model_df = get_bboxes_df(make_model_df)
    make_model_df = get_car_exterior_df(make_model_df)
    class_generated_df = cut_and_save_images(make_model_df)
    generated_images_df = pd.concat([generated_images_df, class_generated_df])

# Saves new dataframe to a .csv file   
generated_images_df.to_csv('/home/app/src/data_analysis/added_images2.csv')

