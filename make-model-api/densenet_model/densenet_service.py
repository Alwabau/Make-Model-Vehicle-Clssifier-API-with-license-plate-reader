import time
import redis
import settings
import json
import cv2
import os
import classes_merger
import numpy as np
import tensorflow as tf
from tensorflow.keras import applications, models
from tensorflow.keras.utils import img_to_array
from tensorflow.keras.preprocessing.image import smart_resize

# Adjusts the percentage of GPU used by our model
config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.15
session = tf.compat.v1.InteractiveSession(config=config)
physical_devices = tf.config.list_physical_devices("GPU")

if physical_devices:
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    except:
        pass

# Redis connection
db = redis.Redis(
    host=settings.REDIS_IP, port=settings.REDIS_PORT, db=settings.REDIS_DB_ID
)

# Loads classes list from the model
CLASSES_FILE = "classes_list.txt"
txt_file = open(CLASSES_FILE, "r")
classes = txt_file.read().splitlines()

# Model loaded from the weights file
WEIGHTS = "densenet_model.h5"
INPUT_SIZE = [224, 224]
model = models.load_model(WEIGHTS)


def predict(image_name):
    """
    This funcions takes an image path to be processed.

    Returns a list with the top 3 predictions from our model.
    """
    image = cv2.imread(image_name)
    array_image = img_to_array(image)
    resized_image = smart_resize(array_image, INPUT_SIZE)
    reshaped_image = resized_image.reshape([1, 224, 224, 3])
    all_predictions = model.predict(reshaped_image)
    mergerd_classes, merged_predictions = classes_merger.get_merged_lists(classes, all_predictions[0])
    best_n = np.argsort(merged_predictions)[-3:]
    top3_list = [
        {
            "make_model": mergerd_classes[best_n[2]],
            "pred_score": str(merged_predictions[best_n[2]]),
        },
        {
            "make_model": mergerd_classes[best_n[1]],
            "pred_score": str(merged_predictions[best_n[1]]),
        },
        {
            "make_model": mergerd_classes[best_n[0]],
            "pred_score": str(merged_predictions[best_n[0]]),
        },
    ]
    return top3_list


def classify_process():
    """
    Loops indefinitely asking Redis for new jobs.
    When a new job arrives, takes it from the Redis queue, uses the loaded ML
    model to get predictions and stores the results back in Redis using
    the original job ID so other services can see it was processed and access
    the results.
    """
    while True:
        job_to_process = json.loads(db.brpop(settings.DENSENET_QUEUE)[1])
        if job_to_process:
            start_time = time.time()
            top_3 = predict(job_to_process["image_path"])
            response_time = time.time() - start_time
            print(f"Make/Model classifier response time: {response_time}", flush=True)
            result_dict = {
                "prediction_1": top_3[0]["make_model"],
                "score_1": top_3[0]["pred_score"],
                "prediction_2": top_3[1]["make_model"],
                "score_2": top_3[1]["pred_score"],
                "prediction_3": top_3[2]["make_model"],
                "score_3": top_3[2]["pred_score"],
                "plate_number": "N/A",
                "message": "Successful prediction",
            }
            db.set(job_to_process["id"], json.dumps(result_dict))
        time.sleep(settings.SERVER_SLEEP)


if __name__ == "__main__":
    # Now launch process
    print("Launching Make/Model service...")
    classify_process()
