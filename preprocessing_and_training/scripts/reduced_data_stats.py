"""
This script will be used gather information from files
classified by folder.
Fist level in the tree will be brand, second level will be model,
and sum of the files in the third level will be the count.
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
    stats_df = get_folders_stats(input_data_folder)
    stats_df.to_csv(output_csv)

def get_folders_stats(input_data_folder):
    """
    Parameters
    ----------
    input_data_folder : str
        Full path to root folder.

    Thi funcion will generate a DataFrame with the following columns:
    - Brand : str
        Car brand name.
    - Model : str
        Car model name.
    - count : int
        The number of files in the class.
    """
    out_df = pd.DataFrame(columns=['Brand', 'Model', 'Count'])
    for root, dirs, files in os.walk(input_data_folder):
        row = {}
        for file in files:
            file_full_path = os.path.join(root, file)
            row['Brand'] = file_full_path.split('/')[-3]
            row['Model'] = file_full_path.split('/')[-2]
            row['Count'] = len(files)
            break
        if row:
            out_df = out_df.append(row, ignore_index=True)
    return out_df
        
    
if __name__ == "__main__":
    args = parse_args()
    main(args.input_data_folder_path, args.output_file_path)
