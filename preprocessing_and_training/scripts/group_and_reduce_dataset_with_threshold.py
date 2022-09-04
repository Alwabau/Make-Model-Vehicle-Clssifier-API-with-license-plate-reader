"""
This script will be used to create a reduced and balanced dataset for training a classification model.
The system will first classify cars by make and model, without taking the year into account.
The samples are first grouped by make and model.
A threshold is used to limit the maximum samples per make and model.
If a (make,model) class has more than $threshold samples, limit the total samples to $threshold samples.
If a (make,model) class has less than $threshold samples, just keep the existing (make,model) class number of samples.
A new directory structure is created containing hard links the paths indicated in the input dataframe.
The newly created directory structure classifies the hard links by make and model.
For example: src_root_dir/{make}_{model}_{year}/{original_filename}.jpg -> dst_root_dir/{make}/{model}/{original_filename}.jpg
Example usage:
group_and_reduce_dataset_with_threshold.py /data_volume/dataset/full_path_info.csv /data_volume/dataset/sample_dataset_images 300
"""

import os 
import argparse
import random
import pandas as pd
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Balance dataset by (make,model) class.")
    parser.add_argument(
        "data_frame",
        type=str,
        help=(
            "Full path to the dataframe having all the dataset images paths. Already "
            "orgenized witrh brand and model columns."
            "E.g. `/data_volume/dataset/full_path_info.csv`."
        ),
    )
    parser.add_argument(
        "output_data_folder",
        type=str,
        help=(
            "Full path to the directory in which we will the resulting hardlinks "
            "will be stored. E.g. `/data_volume/dataset/reduced_images_dataset/`."
        ),
    )
    parser.add_argument(
        "threshold",
        type=int,
        help=(
            "Maximum sample count per class in the resulting reduced dataset."
            "E.g. 300."
        ),
    )
    args = parser.parse_args()

    return args



def generate_hardlinks(data_frame, output_data_folder, threshold):
    """
    Parameters
    ----------
    data_frame: str
        Full path to image paths csv.

    output_data_folder: str
        Full path to the directory in which we will build the resulting dir structure
        and store the resulting hardlinks.

    threshold: int
        Maximum sample count per (make,model) class of the resulting dataset
    """

    files_df = pd.read_csv(data_frame)
    unique_brand_models = pd.DataFrame(files_df.groupby(['brand','model']).size().reset_index(name='freq'))
    for _, row in tqdm(unique_brand_models.iterrows()):

        model_df = files_df.loc[(files_df.model == row.model) & (files_df.brand == row.brand)]
        if len(model_df) > threshold:
            model_df = model_df.sample(threshold)

        for file in model_df.full_path:          
            src_filename = os.path.basename(file)
            dst = os.path.join(row.brand,row.model)
            final_dst = os.path.join(output_data_folder, dst)
            os.makedirs(final_dst, exist_ok=True)
            final_dst = os.path.join(final_dst, src_filename)
            os.link(file, final_dst)

def main(data_frame, output_data_folder, threshold):
    """
    Parameters
    ----------
    data_frame: str
        Full path to image paths csv.

    output_data_folder: str
        Full path to the directory in which we will build the resulting dir structure
        and store the resulting hardlinks.

    threshold: int
        Maximum sample count per (make,model) class of the resulting dataset
    """
    generate_hardlinks(data_frame, output_data_folder, threshold)

if __name__ == "__main__":
    args = parse_args()
    main(args.data_frame, args.output_data_folder, args.threshold)
