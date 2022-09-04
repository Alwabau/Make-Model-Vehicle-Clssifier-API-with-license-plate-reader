# MAKE-MODEL API

## Install and run

To run the services using compose with GPU:

```bash
$ sudo docker-compose -p make_model_api up --build
```

To run the services using compose with CPU:

```bash
$ sudo docker-compose -f docker-compose-cpu.yml -p make_model_api up --build
```

To stop the services:

```bash
$ sudo docker-compose -p make_model_api down
```

## Tests

All tests (unitary and integration) are run automatically from the Dockerfiles before launching the service, while creating the containers using the following lines that are included in the file:

FROM base as test
RUN ["pytest", "-v", "/src/tests"]

## Model Weights

To run the MAKE-MODEL API you will need to add 2 weight files for the models:
- densenet_model.h5: Used for the DenseNet model, it should be on the make-model-api/densenet_service folder.
- mobilenet_model.h5: Used for the MobileNet model, it should be on the make-model-api/mobilenet_service folder.

The weights are in the following link:
https://drive.google.com/drive/u/0/folders/1RvA6czMlKu1Sp4dRUqGFNB33q8tqJFVS

## Port

The port you need to use for the API to run is 0.0.0.0:5000
