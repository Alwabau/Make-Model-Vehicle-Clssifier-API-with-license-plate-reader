import time
import redis
import settings
import json
import cv2
import tensorflow as tf
from tensorflow.keras import layers, applications, Model, models
from tensorflow.keras.utils import img_to_array
from tensorflow.keras.preprocessing.image import smart_resize

# Adjusts the percentage of GPU used by our model
config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.10
session = tf.compat.v1.InteractiveSession(config=config)
physical_devices = tf.config.list_physical_devices("GPU")
if physical_devices:
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    except:
        pass

# Connection to Redis
db = redis.Redis(
    host=settings.REDIS_IP, port=settings.REDIS_PORT, db=settings.REDIS_DB_ID
)

# Loads ML model and assigns it to variable `model`
WEIGHTS = "mobilenet_model.h5"
INPUT_SIZE = [224, 224]
model = models.load_model(WEIGHTS)

# Internal car image detected message
CAR_INT_MSG = "Car Interior detected, please upload an exterior image in order to get make/model"


def predict(image_name):
    """
    Returns True if the image is exterior.
    """
    start_time = time.time()
    image = cv2.imread(image_name)
    array_image = img_to_array(image)
    resized_image = smart_resize(array_image, INPUT_SIZE)
    reshaped_image = resized_image.reshape([1, 224, 224, 3])
    all_predictions = model.predict(reshaped_image)
    prediction_argmax = all_predictions.argmax()
    response_time = time.time() - start_time
    print(f"Views detector response time: {response_time}", flush=True)
    return prediction_argmax == 0


def classify_process():
    """
    Loops indefinitely asking Redis for new jobs.
    When a new job arrives, takes it from the Redis queue, uses the loaded ML
    model to get predict if car exterior and stores the results back in Redis using
    the original job ID so the make/model service can see it was processed and make
    the prediction.
    """
    while True:
        job_to_process = json.loads(db.brpop(settings.MOBILENET_QUEUE)[1])
        if job_to_process:
            if predict(job_to_process["image_path"]):
                denseNet_job_data = {
                    "id": job_to_process["id"],
                    "image_path": job_to_process["image_path"],
                }
                db.lpush(
                    settings.DENSENET_QUEUE, json.dumps(denseNet_job_data)
                )
                db.lpush(settings.OCR_QUEUE, json.dumps(denseNet_job_data))
            else:
                result_dict = {
                    "prediction_1": "N/A",
                    "score_1": None,
                    "plate_number": "N/A",
                    "message": CAR_INT_MSG,
                }
                db.set(job_to_process["id"], json.dumps(result_dict))
        time.sleep(settings.SERVER_SLEEP)


if __name__ == "__main__":
    print("Launching View Detector service...")
    classify_process()
