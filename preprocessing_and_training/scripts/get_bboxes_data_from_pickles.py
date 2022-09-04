"""
This script will be used to extract Detectron information from the pickle 
files, already generated in their respective folders, maintaing the make/model
structure, and generates a new dataframe with bounding boxes info.
"""

import argparse
import os
from tqdm import tqdm
import fileinput
import pandas as pd
import pickle
from collections.abc import Callable

def parse_args():
    parser = argparse.ArgumentParser(description="Train your model.")
    parser.add_argument(
        "source_df_path",
        type=str,
        help=(
            "Full path to the csv file containing all the file paths "
            "E.g. "
            "`/home/app/src/data/files.csv`."
        ),
    )
    parser.add_argument(
        "dest_df_path",
        type=str,
        help=(
            "Full path to the resulting csv file "
            "containing all the frames info. E.g. `/home/app/src/data/frames.csv`."
        ),
    )

    args = parser.parse_args()

    return args

def pickle_paths_to_frames(orig_files_csv_path: str, dest_frames_csv_path: str, process_row):
    """
    Parameters
    ----------
    orig_files_csv_path: str
        path to the csv file containing the picke files paths
    dest_frames_csv_path : str
        path to the csv file that will contain all the detection frame data. If None the file will not be created,
        instead a pandas.DataFrame will be returned
    process_row: Callable
        function that takes a picke file info row and index and returns the information contained in the pickle
        as a list of python dictionaries, each one containing the information of a single frame

    Returns
    -------
    dict_list : list
        List of dictionaries.
        Each dictionary represents the data of a detection frame.
    """
    # Check if the destination file does not exist and if it can be created
    if dest_frames_csv_path is not None:
        with open(dest_frames_csv_path, 'xt') as f:
            pass

    orig_files_csv_df = pd.read_csv(orig_files_csv_path, sep='\t', index_col='index')
    result_list = []
    for index, row in tqdm(orig_files_csv_df.iterrows(), total=orig_files_csv_df.shape[0]):
        single_file_frame_list = process_row(index, row)
        result_list.extend(single_file_frame_list)
    result_df = pd.DataFrame(result_list)
    if dest_frames_csv_path is not None:
        result_df.to_csv(dest_frames_csv_path)
    return result_df

def _process_row_pickle_path(index: int, pickle_file_row) -> list:
    with open(pickle_file_row['pickle_path'], 'rb') as f:
        detection_obj = pickle.load(f)
    frame_list = _extract_info_file_pickle(index, detection_obj)
    return frame_list

def _extract_info_file_pickle(index: int, detection_obj: dict):
    """
    Parameters
    ----------
    index: int
        index of the row that contains the pickle file path.
    detection_obj : dict
        dictionary returned by detectron2 inference.

    Returns
    -------
    dict_list : list
        List of dictionaries.
        Each dictionary represents the data of a detection frame.
    """
    dict_list = []
    im_h = detection_obj["instances"].image_size[0]
    im_w = detection_obj["instances"].image_size[1]
    class_number_list = detection_obj["instances"].pred_classes.tolist()
    frame_list = detection_obj["instances"].pred_boxes.tensor.tolist()
    score_list = detection_obj["instances"].scores.tolist()
    area_list = detection_obj["instances"].pred_boxes.area().tolist()

    for i in range(len(detection_obj["instances"])):
        frame_dict = {
            'file_id': index,
            'im_h': im_h,
            'im_w': im_w,
            'class_number': class_number_list[i],
            'score': score_list[i],
            'x0': int(frame_list[i][0]),
            'y0': int(frame_list[i][1]),
            'x1': int(frame_list[i][2]),
            'y1': int(frame_list[i][3]),
            'area': int(area_list[i]),
            }
        dict_list.append(frame_dict)
    return dict_list

def main(source_df_path: str, dest_df_path: str):
    """
    Parameters
    ----------
    source_df_path : str
        Full path to .csv file cointaining pickles info paths

    dest_df_path : str
        Full path to the csv file in which the resulting
        frames information will be stored.
    """
    pickle_paths_to_frames(source_df_path, dest_df_path, _process_row_pickle_path)

if __name__ == "__main__":
    args = parse_args()
    main(args.source_df_path, args.dest_df_path)
