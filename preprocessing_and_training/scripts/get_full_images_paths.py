"""
This script will be used to create a csv file with the information from files
classified by folder.
"""

import argparse
import os
import pandas as pd
from tqdm import tqdm

remove_characters_dict = {"\"": None, "'": None}

def parse_args():
    parser = argparse.ArgumentParser(description="Train your model.")
    parser.add_argument(
        "input_data_folder_path",
        type=str,
        help=(
            "Full path to the directory having all the cars images."
        ),
    )
    parser.add_argument(
        "output_file_path",
        type=str,
        help=(
            "Full path to the resulting csv file"
        ),
    )

    args = parser.parse_args()

    return args


def main(input_data_folder, output_csv):
    stats_df = get_folders_stats(input_data_folder, class_dirpath_info_extract_cars_underscore)
    stats_df.to_csv(output_csv)

def get_folders_stats(input_data_folder, class_dirpath_info_extract):
    """
    This function will create a csv file with the information from files
    classified by folder.
    The class description should be the first level in the tree.
    The individual files should be in the second level.
    In the case of the image files the file metadata follows a pattern:
    {make}_{model}_{year}/

    Parameters
    ----------
    input_data_folder : str
        Full path to the folder cointaining the whole dataset.

    class_dirpath_info_extract : str
        Function to extract the needed information from the path to the 
        directory that contains the files.
    """
    un_process_file = []
    dict_list = []
    count = 0
    count2 = 0
    for root, dirs, files in tqdm(os.walk(input_data_folder)):
        for file_path in files:
            file_frames = {}
            try:
                file_name = os.path.join(root, file_path)
                parsed_dict = class_dirpath_info_extract(root)
                file_frames['full_path'] = file_name
                file_frames['brand'] = parsed_dict['make']
                file_frames['model'] = parsed_dict['model']
                file_frames['year'] = parsed_dict['year']
                dict_list.append(file_frames)
            except Exception as e:
                print(e)
                un_process_file.append(file_path)
    if un_process_file:
        print('\nUnable To Process these files\n')
        for files in un_process_file:
            print(files)
    df_final = pd.DataFrame.from_dict(dict_list)
    return df_final

_remove_characters_dict = {"\"": None, "'": None}

def class_dirpath_info_extract_cars_underscore(dir_path):
    name = os.path.basename(dir_path)
    name = name.translate(_remove_characters_dict)
    split_name = name.split('_')
    make = split_name[0]
    model = split_name[1]
    year = int(split_name[-1])
    ret = {'path':dir_path, 'make':make, 'model':model, 'year':year}
    return ret

if __name__ == "__main__":
    args = parse_args()
    main(args.input_data_folder_path, args.output_file_path)
