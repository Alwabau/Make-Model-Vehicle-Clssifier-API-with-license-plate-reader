"""
This script will load a csv with manual tagged images and generate a train and test set
"""

import os 
import argparse
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser(description="Generate train test dataset.")
    parser.add_argument(
        "output_data_folder",
        type=str,
        help=(
            "Full path to the directory in which we will store the resulting "
            "hardlinks. E.g. `/data_volume/datest/mobilenet_training_dataset/`."
        ),
    )
    parser.add_argument(
        "csv_path",
        type=str,
        help=(
            "Path to the csv file containing the manual tagged images."
            "E.g. /data_volume/ferf8/FinalExtIntTrashBalanced.csv"
        ),
    )
    parser.add_argument(
        "percentage",
        type=int,
        help=(
            "Percentage of elements that will be used as test."
            "E.g. 10."
        ),
    )
    args = parser.parse_args()

    return args

def create_dirs(output_path, df):
    class_dict = {
        0 : 'ext',
        1 : 'int',
        2 : 'trash' 
    }
    for index, row in df.iterrows():
        destination = os.path.join(output_path, class_dict[row['class']], os.path.basename(row['path']))
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        print(destination)
        os.link(row['path'], destination)

def main(output_data_folder, csv_path, percentage):
    df = pd.read_csv(csv_path)
    df = df.iloc[:,1:]
    df = df.dropna(axis=0, how='any')
    df = df.sort_values(by=['class'])
    test_elements = int ((len(df.index) * percentage / 100) / 3) #3 = amount of classes
    test_df = df[df['class'] == 0].sample(test_elements)
    test_df = test_df.append(df[df['class'] == 1].sample(test_elements))
    test_df = test_df.append(df[df['class'] == 2].sample(test_elements))
    train_df = pd.concat([df, test_df]).drop_duplicates(keep=False)

    create_dirs(os.path.join(output_data_folder, 'train'), train_df)
    create_dirs(os.path.join(output_data_folder, 'test'), test_df)


if __name__ == "__main__":
    args = parse_args()
    main(args.output_data_folder, args.csv_path, args.percentage)

