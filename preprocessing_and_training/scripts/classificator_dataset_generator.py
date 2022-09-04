'''
Creates most balanced possible dataset with filtered and cropped images to train the densnet model
'''

import os 
import argparse
import cv2
import pandas as pd
from utils import utils
from tqdm import tqdm
from tensorflow.keras.utils import img_to_array
from tensorflow.keras.preprocessing.image import smart_resize
from mobilenet_model import mobilenetv3_model

CAR_CODE = 2
TRUCK_CODE = 7
BUS_CODE = 5
CODES = [CAR_CODE, TRUCK_CODE, BUS_CODE]

CONFIG_YML = "/data_volume/alan/ay22-01-final-project-5/mobilenet_weights/w_002/config_mobilenet7.yml"
WEIGHTS = "/data_volume/alan/ay22-01-final-project-5/alan/ay22-01-final-project-5/mobilenet_weights/w_007/model.77-0.1488.h5"

CONFIG = utils.load_config(CONFIG_YML)
INPUT_SIZE=CONFIG["data"]["image_size"]
MODEL_CLASSES = utils.get_class_names(CONFIG)

def parse_args():
    parser = argparse.ArgumentParser(description="Train your model.")
    parser.add_argument(
        "path_df",
        type=str,
        help=(
            "Full path to the dataframe having all the dataset images paths. Already "
            "organized with make and model columns."
            "E.g. `/data_volume/dataset/images`."
        ),
    )
    parser.add_argument(
        "frames_df",
        type=str,
        help=(
            "Full path to the directory in which we will store the resulting "
            "hardlinks. E.g. `/home/app/src/reduced_images_dataset/`."
        ),
    )
    parser.add_argument(
        "origin_folder",
        type=str,
        help=(
            "Amount of elements per class we´ll have in our reduced dataset."
            "E.g. 50."
        ),
    )
    parser.add_argument(
        "destiny_folder",
        type=str,
        help=(
            "Amount of elements per class we´ll have in our reduced dataset."
            "E.g. 50."
        ),
    )
    parser.add_argument(
        "threshold",
        type=float,
        help=(
            "Amount of elements per class we´ll have in our reduced dataset."
            "E.g. 50."
        ),
    )
    args = parser.parse_args()

    return args


def get_area(height, width):
    '''
    Calculates area of the picture
    '''
    return height * width

def get_area_ratio(original_area, frame_area):
    '''
    Calculates the bounding box ratio
    '''
    bbox_ratio = frame_area / original_area
    return bbox_ratio

def get_output_df(path_df, pickles_df, threshold):
    '''
    Get a dataframe with the images that are cars, with outside views, and that pass the 20% threshold.
    '''
    car_trucks_buses_mask = (pickles_df['class_number'] == 2) | (pickles_df['class_number'] == 7) | (pickles_df['class_number'] == 5)
    pickles_class_filtered_df = pickles_df[car_trucks_buses_mask]

    pickles_class_filtered_df['image_area'] = pickles_class_filtered_df['im_h'] * pickles_class_filtered_df['im_w']
    pickles_class_filtered_df['ratio'] = pickles_class_filtered_df['area'] / pickles_class_filtered_df['image_area']

    ratio_mask = pickles_class_filtered_df['ratio'] > threshold
    pickles_ratio_class_filtered_df = pickles_class_filtered_df.loc[ratio_mask]
    pickles_ratio_class_filtered_max_df = pickles_ratio_class_filtered_df.groupby(by='file_id').max()

    result = pd.merge(
    path_df,
    pickles_ratio_class_filtered_max_df,
    how="inner",
    right_on='file_id',
    left_index=True,
    right_index=False,
    sort=True,
    suffixes=("_x", "_y"),
    copy=True,
    indicator=False,
    validate=None,
    )

    return result

def is_exterior(model, image):
    """
    Returns True if the image is exterior.
    """
    array_image = img_to_array(image)
    resized_image = smart_resize(array_image, INPUT_SIZE)
    reshaped_image = resized_image.reshape([1, 224, 224, 3])
    all_predictions = model.predict(reshaped_image)
    prediction_argmax = all_predictions.argmax()
    predicted_class = MODEL_CLASSES[prediction_argmax]

    return predicted_class == 'ext'

def generate_categories(df, destiny_folder,model):
    '''
    Generates categories file structure.
    '''
    for index, row in tqdm(df.iterrows(), total=len(df)):
        file_name = os.path.basename(row['original_path'])
        make_model_str = row['make'] + '_' + row['model']
        out_path = os.path.join(destiny_folder, make_model_str, file_name)
        img = cv2.imread(row['original_path'])
        if is_exterior(model, img):
            cropped_image = img[row['y0']:row['y1'] , row['x0']:row['x1']]
            os.makedirs(os.path.dirname(out_path), exist_ok=True,  mode=0o777)
            cv2.imwrite(out_path, cropped_image)


def generate_reduced_imgs_dataset(path_df, pickles_df, origin_folder, destiny_folder, threshold, model):
    """
    Generates reduced images datasets with cropped image.
    """
    out_df = get_output_df(path_df, pickles_df, threshold)
    result_test = out_df.sample(frac=threshold, replace=False)
    result_train = pd.concat([out_df,result_test]).drop_duplicates(keep=False)

    generate_categories(result_train, os.path.join(destiny_folder, 'train'), model)
    generate_categories(result_test, os.path.join(destiny_folder, 'test'), model)



def main(path_df, frames_df, origin_folder, destiny_folder, threshold):
    """
    Parameters
    ----------
    data_folder : str
        Full path to Raw images Dataset folder.

    output_data_folder : str
        Full path to the directory in which we will store the resulting
        valid car/truck pickles.
    """

    paths_data_frame = pd.read_csv(path_df, index_col='index')
    frames_data_frame = pd.read_csv(frames_df, index_col=0)

    cnn_model = mobilenetv3_model.create_model(weights=WEIGHTS)

    generate_reduced_imgs_dataset(paths_data_frame, frames_data_frame, origin_folder, destiny_folder, threshold, cnn_model)



if __name__ == "__main__":
    args = parse_args()
    main(args.path_df, args.frames_df, args.origin_folder, args.destiny_folder ,args.threshold)