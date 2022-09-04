import time
import redis
import settings
import json
import cv2
import os
import easyocr


# Redis connection
db = redis.Redis(
    host=settings.REDIS_IP, port=settings.REDIS_PORT, db=settings.REDIS_DB_ID
)

# Downloads English language to OCR model
reader = easyocr.Reader(["en"], gpu=True)

# European countries vehicle registration plate code list
plate_list = [
    "A", "B", "BG", "HR", "CY", "CZ", "DK",
    "EST", "FIN", "F", "D", "GR", "H", "IRL",
    "I", "LV", "LT", "L", "M", "NL", "PL", "GB",
    "P", "RO", "SK", "SLO", "E", "S", "UA",
    "AL", "AND", "BY", "BIH", "IS", "FL", "MD",
    "MC", "MNE", "MK", "NMK", "N", "RSM", "SRB",
    "CH", "UK", "SCV", "CV", "V"
]


def plate_reader(image):
    """
    This function receives an outside image of a car and returns the Eurpoean
    vehicle registration plate value if it exists and the charachters are readable.

    Parameters
    ----------
    image : str
        Path to the image you want to process.

    Returns
    -------
    ocr_result : str
        It will return the full license plate value (e.g. 'RO B396JOY') comprised of
        two elements:
            - country: Country code (e.g. 'RO' for Romania)
            - plate number: Plate letters and numbers that identify the car (e.g.
            'B396JOY')
    """

    def plate_ocr(result):
        for text in range(len(result)):
            if result[text][1].upper() in plate_list:
                country = result[text][1].upper()
                plate_number = result[text + 1][1].upper()
                plate_number = plate_number.replace("|", "I")
                plate_number = "".join(filter(str.isalnum, plate_number))
                output = f"{country} {plate_number}"
                return output
        output = "There is no European vehicle registration plate or the characters are unclear"
        return output

    def ocr_result(image):
        initial_time = time.time()
        result = reader.readtext(image)
        license_plate = plate_ocr(result)
        response_time = time.time() - initial_time
        print(f"Plate Detector Service response time: {response_time}", flush=True)
        return license_plate

    return ocr_result(image)


def classify_process():
    """
    Loosp indefinitely asking Redis for new jobs.
    When a new job arrives, takes it from the Redis queue, uses the loaded ML
    model to get predictions and stores the results back in Redis using
    the original job ID so other services can see it was processed and access
    the results.
    """
    while True:
        job_to_process = json.loads(db.brpop(settings.OCR_QUEUE)[1])
        if job_to_process:
            img_path = os.path.join(settings.UPLOAD_FOLDER, os.path.basename(job_to_process["image_path"]))
            plate_number = plate_reader(img_path)
            result_dict = {
                "prediction": "N/A",
                "score": "N/A",
                "plate_number": plate_number,
                "message": "N/A",
            }
            id = "plate_number_" + job_to_process["id"]
            db.set(id, json.dumps(result_dict))
        time.sleep(settings.SERVER_SLEEP)


if __name__ == "__main__":
    # Now launch process
    print("Launching Plate Detector service...")
    classify_process()
