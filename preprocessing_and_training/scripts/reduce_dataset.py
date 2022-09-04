import os
import argparse
import random
import pandas as pd
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Train your model.")
    parser.add_argument(
        "data_frame",
        type=str,
        help=(
            "Full path to the dataframe having all the dataset images paths. Already "
            "orgenized witrh brand and model columns."
            "E.g. `/data_volume/dataset/images`."
        ),
    )
    parser.add_argument(
        "output_data_folder",
        type=str,
        help=(
            "Full path to the directory in which we will store the resulting "
            "hardlinks. E.g. `/home/app/src/reduced_images_dataset/`."
        ),
    )
    parser.add_argument(
        "threshold",
        type=int,
        help=(
            "Amount of elements per class we´ll have in our reduced dataset."
            "E.g. 50."
        ),
    )
    args = parser.parse_args()

    return args



def generate_hardlinks(data_frame, output_data_folder, threshold):
    '''
    Generates the hardlinks of the reduced dataset.
    '''
    files_df = pd.read_csv(data_frame)
    unique_brand_models = pd.DataFrame(files_df.groupby(['brand','model']).size().reset_index(name='freq'))
    for _, row in tqdm(unique_brand_models.iterrows()):

        model_df = files_df.loc[(files_df.model == row.model) & (files_df.brand == row.brand)]
        if len(model_df) > threshold:
            model_df = model_df.sample(threshold)

        for file in model_df.full_path:          
            src_filename = os.path.basename(file)
            dst_base = os.path.basename(os.path.dirname(file))
            dst = os.path.join(row.brand,row.model)
            final_dst = os.path.join(output_data_folder, dst)
            os.makedirs(final_dst, exist_ok=True)
            final_dst = os.path.join(final_dst, src_filename)
            os.link(file, final_dst)

def main(data_frame, output_data_folder, threshold):
    """
    Parameters
    ----------
    data_folder : str
        Full path to Raw images Dataset folder.

    output_data_folder : str
        Full path to the directory in which we will store the resulting
        valid car/truck pickles.
    """
    generate_hardlinks(data_frame, output_data_folder, threshold)

if __name__ == "__main__":
    args = parse_args()
    main(args.data_frame, args.output_data_folder, args.threshold)
