# PREPROCESSING AND TRAINING


## Install and run

To run the services using compose with GPU:

```bash
$ sudo docker build -t final_project -f Dockerfile .
```

To run the services (with GPU):

```bash
$ sudo docker run --rm --net host --gpus all -it -v $(pwd):/home/app/src --workdir /home/app/src final_project bash
```

## EDA Notebook files

All the necessary files to run the EDA Notebook are in the following link:
https://drive.google.com/drive/u/0/folders/1sI6yPuysYPA4isjONA9sKJsrywKd10pt

## Training models

To run the trainings you need to type the following since it uses parsing:

```bash
sudo python3 (route to training, e.g. "preprocess_and_traning/densenet_model/desnenet121_model.py") (rout to .yml file you have created containing the training parameters, e.g. "preprocessing_and_training/densenet_weights/w_001/config_densenet.yml").
```