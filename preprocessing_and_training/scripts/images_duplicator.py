'''
This script is used to duplicate images in train filder in order to avoid unbalanced data
Either run it with sudo or change permissions to the folder
'''

import os, random

DATA_FOLDER = "/data_volume/dataset/sample_classifier_images/train"
MIN_IMAGES_IN_FOLDER = 100



def main():
    for folder in os.listdir(DATA_FOLDER):
        amount_files = len(os.listdir(os.path.join(DATA_FOLDER, folder)))
        if amount_files < MIN_IMAGES_IN_FOLDER:
            print("Folder:", folder,  " Amount of files", amount_files)
        if amount_files < MIN_IMAGES_IN_FOLDER:
            print("Duplicating images")
            for i in range(MIN_IMAGES_IN_FOLDER - amount_files):

                original_file = random.choice(os.listdir(os.path.join(DATA_FOLDER, folder))) 
                file = "duplicated_" + str(i) + "_" + original_file
                os.makedirs(os.path.join(DATA_FOLDER, folder), exist_ok=True)
                os.system("cp " + os.path.join(DATA_FOLDER, folder, original_file) + " " + os.path.join(DATA_FOLDER, folder, file))
                print(os.path.join(DATA_FOLDER, folder,file))
            print("Done")

if __name__ == "__main__":
    main()