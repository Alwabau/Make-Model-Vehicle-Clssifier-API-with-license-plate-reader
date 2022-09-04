# Intro

This project takes vehicle images and returns, on the one hand, the make and model of 836 European vehicle classes, and on the other, the European vehicle registration plate. 

Appropriate feedback is given if there is no image file, if the image is too small, if the image shows the inside of the car and if the license plate can't be read either because there is none or because the letters and numbers are too small to be read. The country ID letters should be readable for the model to take the whole plate as valid.

# Project Structure

```
├── api
│   ├── Dockerfile
│   ├── app.py
│   ├── middleware.py
│   ├── views.py
│   ├── settings.py
│   ├── utils.py
│   ├── template
│   │   └── index.html
│   └── tests
│       ├── test_api.py
│       └── test_utils.py
├── densenet_model
│   ├── Dockerfile
│   ├── densenet_service.py
|   ├── classes_merger.py
│   ├── settings.py
│   ├── densenet_model.h5
│   ├── classes_list.txt
│   └── tests
│       └── test_densenet_service.py
├── detectron_model
│   ├── Dockerfile
│   ├── detectron_service.py
│   ├── settings.py
│   └── tests
│       └── test_detectron_service.py
├── mobilenet_model
│   ├── Dockerfile
│   ├── mobilenet_service.py
│   ├── settings.py
│   ├── mobilenet_model.h5
│   └── tests
│       └── test_mobilenet_service.py
├── ocr_model
│   ├── Dockerfile
│   ├── ocr_service.py
│   ├── settings.py
│   └── tests
│       └── test_ocr_service.py
├── template_files
├── tests
│   └── integration
│       └── test_integration.py
├── docker-compose-cpu.yml
├── docker-compose.yml
└── README.md
```

Let's take a quick overview on each module:

- api: It has all the needed code to implement the communication interface between the users and our service. It uses Flask and Redis to queue tasks to be processed by our machine learning model.
    - 'api/Dockerfile': Dockerfile settings.
    - 'api/app.py': Setup and launch the Flask api.
    - 'api/middleware.py': Creates an interaction between the API and the models.
    - 'api/views.py': Contains the API endpoints.
    - 'api/settings.py': It has all the API settings.
    - 'api/utils.py': Implements some extra functions used internally by our api.
    - 'api/templates': Here we put the index.html file used in the frontend.
    - 'api/tests': Test suite.

- densenet_model: Implements the logic to get jobs from Redis and process them with our machine Learning model. When we get the predicted value from our model, we must encole it on Redis again so it can be delivered to the user.
    - 'densenet_model/Dockerfile': Dockerfile settings.
    - 'densenet_model/densenet_service.py': Runs a thread in which it get jobs from Redis, process them with the model and returns the answers.
    - 'densenet_model/classes_merger.py': Function to merge vehicles subclasses into main class e.g.('abarth_500c': 'abarth_500').
    - 'densenet_model/settings.py': Settings for our DenseNet model.
    - 'densenet_model/densenet_model.h5': Densnet model weights used.
    - 'densenet_model/classes_list: List with all the vehicle classes.
    - 'densenet_model/tests': Test suite.

- detectron_model: Implements the logic to get jobs from Redis and process them with our machine Learning model. When we get the predicted value from our model, we must encole it on Redis again so it can be delivered to the user.
    - 'detectron_model/Dockerfile': Dockerfile settings.
    - 'detectron_model/detectron_service.py': Runs a thread in which it get jobs from Redis, process them with the model and returns the answers.
    - 'detectron_model/settings.py': Settings for our Detecron2 model.
    - 'detectron_model/tests': Test suite.

- mobilenet_model: Implements the logic to get jobs from Redis and process them with our machine Learning model. When we get the predicted value from our model, we must encole it on Redis again so it can be delivered to the user.
    - 'mobilenet_model/Dockerfile': Dockerfile settings.
    - 'mobilenet_model/mobilenet_service.py': Runs a thread in which it get jobs from Redis, process them with the model and returns the answers.
    - 'mobilenet_model/settings.py': Settings for our MobileNet model.
    - 'mobilenet_model/mobilenet_model.h5': MobileNet model weights used.
    - 'mobilenet_model/tests': Test suite.

- ocr_model: Implements the logic to get jobs from Redis and process them with our machine Learning model. When we get the predicted value from our model, we must encole it on Redis again so it can be delivered to the user.
    - 'ocr_model/Dockerfile': Dockerfile settings.
    - 'ocr_model/ocr_service.py': Runs a thread in which it get jobs from Redis, process them with the model and returns the answers.
    - 'ocr_model/settings.py': Settings for our OCR model.
    - 'ocr_model/tests': Test suite.

- template_files: Contains templates to renderize the web.

- tests: This module contains integration tests so we can properly check our system end-to-end behavior is the expected.

- 'docker-compose-cpu.yml': Dockerfile compose settings for CPU.

- 'docker-compose.yml': Dockerfile compose settings for GPU.

- 'README.md': Readme file with commands and considerations.

The communication among our services (api and models) will be done using Redis. Every time the api wants to process an image, it will store the image on disk and send the image name through Redis to the Detectron2 model service. If Detectron2 detects a vehicle image the MobileNet service will classify it as internal or external. If the image is external, then it will be sent parallelly to the DenseNet model to classify the vehicle, and to the OCR model to read the vehicle registration plate. The models already know in which folder images are being store, so it only has to use the file name to load it, get predictions and return the results back to the api.

