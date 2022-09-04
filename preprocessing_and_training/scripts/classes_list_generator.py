'''
Simple script to generate a list of classes for a given dataset.
Said list will be used by the api to decode predictions.
'''

import os
import pandas as pd

TRAIN_DIRECTORY = "/data_volume/dataset/sample_classifier_images/train"
OUT_PATH = "/home/parac3lsus/ay22-01-final-project-5/data_analysis/classes_list.txt"

classes = []

for folder in os.listdir(TRAIN_DIRECTORY):
    classes.append(os.path.basename(folder))

with open (OUT_PATH, "w") as f:
    for c in classes:
        f.write(c + "\n")