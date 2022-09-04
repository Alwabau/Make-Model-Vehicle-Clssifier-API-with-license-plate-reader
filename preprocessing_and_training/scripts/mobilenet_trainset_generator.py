import os
import random

ORIGIN_PATH = '/data_volume/dataset/images'
TARGET_PATH = '/data_volume/dataset/images_for_manual_tagging'
paths = []
for root, dirs, files in os.walk(ORIGIN_PATH):
    for file in files:
        if os.path.isfile(os.path.join(root, file)):
            paths.append(os.path.join(root, file))
paths = random.sample(paths, 5000)


for path in paths:
    _, extension = os.path.splitext(path) 
    out_path = os.path.join(TARGET_PATH, (str(n)+extension))
    print("###################################")
    print('Path:', path)
    print('Out Path:', out_path)
    os.link(path, out_path)
