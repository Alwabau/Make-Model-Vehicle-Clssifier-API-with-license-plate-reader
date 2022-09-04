import time
import redis
import settings
import uuid
import json

# Redis connection
db = redis.Redis(
    host=settings.REDIS_IP, port=settings.REDIS_PORT, db=settings.REDIS_DB_ID
)


def wait_for_licence_plate(job_id, out_data):
    """
    Waits for the next response from the licence plate service
    and returns the original dictionary adding the license plate.

    Parameters
    ----------
    job_id : str
        Id of the job.
    out_data : dict
        Dictionary with the results

    Returns
    -------
    output: dict
        A dictionary with the top 3 categories, prediction scores,
        plate number, message and filename
    """
    timeout = time.time() + 20  # 20-second timeout
    
    while True:
        id = "plate_number_" + job_id
        if db.exists(id):
            plate_data = json.loads(db.get(id))
            db.delete(id)
            out_data["plate_number"] = plate_data["plate_number"]
            return out_data
        if time.time() > timeout:
            return out_data
        time.sleep(settings.API_SLEEP)


def model_predict(image_name):
    """
    Receives an image name and queues the job into Redis.
    Will loop until getting the answer from our ML service.

    Parameters
    ----------
    image_name : str
        Name for the image uploaded by the user.

    Returns
    -------
    output: dict
        A dictionary with the top 3 categories, prediction scores,
        plate number, message and filename

    """
    # Assigns an unique ID for this job and adds it to the queue
    job_id = str(uuid.uuid4())

    job_data = {"id": job_id, "image_name": image_name}

    db.lpush(settings.REDIS_QUEUE, json.dumps(job_data))

    # Loops until we received the response from our ML model
    while True:

        output = None
        if db.exists(job_id):
            output = json.loads(db.get(job_id))
            db.delete(job_id)

            if output["prediction_1"] != "N/A":
                output = wait_for_licence_plate(job_id, output)

            print("Output Sent", flush=True)
            return output

        time.sleep(settings.API_SLEEP)
    return None
