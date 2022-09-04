# Intro

This project takes vehicle images and returns, on the one hand, the make and model of 836 European vehicle classes, and on the other, the European vehicle registration plate. 

Appropriate feedback is given if there is no image file, if the image is too small, if the image shows the inside of the car and if the license plate can't be read either because there is none or because the letters and numbers are too small to be read. The country ID letters should be readable for the model to take the whole plate as valid.

# Folder Structure

```
├── densenet_model
│   └── densenet121_model.py
├── mobilenet_model
│   └── mobilenetv3_model.py
├── notebooks
│   ├── Dataset_Increaser.ipynb
│   ├── Densenet_Model_Evaluation.ipynb
│   ├── EDA.ipynb
│   └── Mobilenet_Model_Evaluation.ipynb
├── scripts
│   ├── classes_list_generator.py
│   ├── classificator_dataset_generator.py
│   ├── dataset_increaser.py
│   ├── densenet_train.py
│   ├── generate_pickles_dataset.py
│   ├── get_bboxes_data_from_pickles.py
│   ├── get_classifier_images_path.py
│   ├── get_file_stats.py
│   ├── get_full_images_paths.py
│   ├── get_pickles_from_detectron.py
│   ├── group_and_reduce_dataset_with_threshold.py 
│   ├── images_duplicator.py
│   ├── mobilenetv3_train.py
│   ├── model_ext_dataset_gen.py
│   ├── reduce_dataset.py
│   ├── reduced_data_stats.py
├── utils
│   ├── data_aug.py
│   └── utils.py
├── Dockerfile
├── docker-compose-cpu.yml
├── trainset_generator.py
└── README.md
```

Let's take a quick overview on each module:

- densenet_model: Contains the DenseNet121 model that was trained.
    - 'densenet_model/densenet121_model.py': DenseNet121 model that was trained.

- mobilenet_model: Contains the MobileNetv3_model model that was trained.
    - 'mobilenet_model/mobilenetv3_model.py': MobileNetv3_model model that was trained.

- notebooks: Contains all jupyter notebooks used.
    - 'notebooks/Dataset_Increaser.ipynb': Generates a table with the info of all the files already present in the train and test datasets that were already separated.
    - 'notebooks/Densenet_Model_Evaluation.ipynb':File used to test the DenseNet Model.
    - 'notebooks/EDA.ipynb': EDA file of all the image information for this project.
    - 'notebooks/Mobilenet_Model_Evaluation.ipynb': File used to test the MobileNet Model.

- scripts: Contains all the scripts used for training and preprocessing
    - 'scripts/classes_list_generator.py': Generates a list of classes that will be used by the API.
    - 'scripts/classificator_dataset_generator.py': Creates most balanced possible dataset with filtered and cropped images to train the densnet model.
    - 'scripts/dataset_increaser.py': Used Dataset_Increaser.ipynb to add more images to train.
    - 'scripts/densenet_train.py': This script will be used for training our DenseNet CNN. This model is the last classification between all the 836 make/model classes.
    - 'scripts/generate_pickles_dataset.py': Generates pickles with the object info of each image like size and bounding boxes, among others.
    - 'scripts/get_bboxes_data_from_pickles.py': This script will be used to extract Detectron information from the pickle files, already generated in their respective folders, maintaing the make/model structure.
    - 'scripts/get_classifier_images_path.py': Creates a csv file with the information from files classified by folder without years.
    - 'scripts/get_file_stats.py': This script will be used gather information from files classified by folder. The class description should be the first level in the tree. The individual files should be in the second level. In the case of the image files the file metadata follows a pattern: {make}_{model}_{year}/ use for EDA.
    - 'scripts/get_full_images_paths.py': Creates a csv file with the information from files classified by folder. 
    - 'scripts/get_pickles_from_detectron.py': This script will process the input image with Detectron2 and generates a pickle file.
    - 'scripts/group_and_reduce_dataset_with_threshold.py': Creates a reduced and balanced dataset for training a classification model.
    - 'scripts/images_duplicator.py': Duplicates images of classes that have less than 100 images until they reach that amount.
    - 'scripts/mobilenetv3_train.py': This script will be used for training our MobileNet CNN. This model will classify between exterior and interior vahicules images.
    - 'scripts/mobilenet_trainset_generator.py': Generates the set of images that will be tagged later to distinguish between car interior and car exterior.
    - 'scripts/model_ext_dataset_gen.py': This script will load a csv with manual tagged images and generate a train and test set.
    - 'scripts/reduce_dataset.py': Takes the original dataset and generates a smaller dataset.
    - 'scripts/reduced_data_stats.py': This script will be used gather information from files classified by folder. Fist level in the tree will be brand, second level will be model, and sum of the files in the third level will be the count.
    - 'scripts/subclass_compress.py: This script merges similar vehicle classes into one.

- utils:
    - 'utils/data_aug.py': Creates the data augmentation layers.
    - 'utils/utils.py': Extra functions used internally by our api.

- 'Dockerfile': Dockerfile settings.

- 'README.md': Readme file with commands and considerations.
