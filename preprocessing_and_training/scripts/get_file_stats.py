"""
This script will be used gather information from files classified by folder.
The class description should be the first level in the tree.
The individual files should be in the second level.
In the case of the image files the file metadata follows a pattern:
{brand}_{model}_{year}/
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
    print("Min value: ", stats_df['count'].min())
    stats_df.to_csv(output_csv)

def get_folders_stats(input_data_folder, class_dirpath_info_extract):
    """
    Parameters
    ----------
    input_data_folder : str
        Full path to root folder.

    class_dirpath_info_extract : str
        Function to extract the needed information from the path to the directory that contains the files
    """
    dict_list = []
    for node in tqdm(os.listdir(input_data_folder)):
        abs_node_path = os.path.join(input_data_folder,node)
        if os.path.isdir(abs_node_path):
            parsed_dict = class_dirpath_info_extract(abs_node_path)
            dict_list.append(parsed_dict)
    df_final = pd.DataFrame.from_dict(dict_list)
    return df_final

_remove_characters_dict = {"\"": None, "'": None}
def class_dirpath_info_extract_cars_underscore(dir_path):
    count = len([name for name in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, name))])
    name = os.path.basename(dir_path)
    name = name.translate(_remove_characters_dict)
    split_name = name.split('_')
    make = split_name[0]
    model = split_name[1]
    year = int(split_name[-1])
    ret = {'path':dir_path, 'make':make, 'model':model, 'year':year, 'count':count}
    return ret

if __name__ == "__main__":
    args = parse_args()
    main(args.input_data_folder_path, args.output_file_path)



