import time
import redis
import settings
import json
import os
import cv2
import torch
from torchvision.ops import box_area
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg

# Redis configuration
db = redis.Redis(
    host=settings.REDIS_IP, port=settings.REDIS_PORT, db=settings.REDIS_DB_ID
)

# Defines the codes for filtering cars, trucks and buses
DETECTRON_CAR_CODE = 2
DETECTRON_TRUCK_CODE = 7
DETECTRON_BUS_CODE = 5

# Sets the minimum area threshold a vehicle should have in proportion to its image
AREA_TRESHHOLD = 0.2

# Folder with the cropped image
CROPPED_IMAGES_FOLDER = "cropped_images/"

# Error messages
LOW_AREA_MSG = (
    "Low vehicle area, please submit a closer picture of the car"
)
NOT_CAR_MSG = (
    "We could not find a car in the image, please submit a car picture"
)
SMALL_IMG_SIZE_MSG = "Small image size please try again with a bigger image"

# Uploads Detectron2 weights and checks if GPU is not available
cfg = get_cfg()
if not torch.cuda.is_available():
    print("Detectron Service Running on CPU", flush=True)
    cfg.MODEL.DEVICE = "cpu"
cfg.merge_from_file(
    model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")
)
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
    "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"
)
DET_MODEL = DefaultPredictor(cfg)


def get_box_info(boxes, full_pic_area):
    """
    This function calculates the ratio of the box area to the full picture area,
    then returns the box coordinates of the biggest box to be used in the cropped image
    """
    vehicle_areas = box_area(boxes.pred_boxes.tensor)
    max_arg_index = vehicle_areas.argmax().item()
    max_frame_area = vehicle_areas[max_arg_index].item()
    bbox_ratio = max_frame_area / full_pic_area
    box_coordinates = (
        boxes.pred_boxes.tensor[max_arg_index].to(torch.int).tolist()
    )
    return bbox_ratio, box_coordinates


def generate_cropped_image(path):
    """
    This function crops the image and saves it on the cropped images folder to be used later by
    the make/model service.
    In case of an image wtih no car or if the image is not big enough it returns the
    corresonding codes and messages.
    """
    img = cv2.imread(path)
    outputs = DET_MODEL(img)
    cars_trucks_buses = (
        outputs["instances"].pred_classes == DETECTRON_CAR_CODE
    ) | (
        outputs["instances"].pred_classes
        == DETECTRON_BUS_CODE
        | (outputs["instances"].pred_classes == DETECTRON_TRUCK_CODE)
    )
    car_trucks_boxes = outputs["instances"][cars_trucks_buses]

    if car_trucks_boxes.pred_classes.size()[0] != 0:
        full_pic_area = img.shape[0] * img.shape[1]
        bbox_ratio, box_coordinates = get_box_info(
            car_trucks_boxes, full_pic_area
        )

        if bbox_ratio > AREA_TRESHHOLD:
            cropped_image = img[
                box_coordinates[1] : box_coordinates[3],
                box_coordinates[0] : box_coordinates[2],
            ]
            h, w, _ = cropped_image.shape
            if (
                h > 224 and w > 224
            ):  # Checks if the cropped image is big enough to be used in the next model
                # Saves the cropped image to be use by the densenet model
                cropped_image_path = os.path.join(
                    settings.UPLOAD_FOLDER,
                    CROPPED_IMAGES_FOLDER,
                    os.path.basename(path),
                )
                cv2.imwrite(cropped_image_path, cropped_image)

                # Saves the original image with the car box highlighted to be shown in the template
                cv2.rectangle(
                    img,
                    (box_coordinates[0], box_coordinates[1]),
                    (box_coordinates[2], box_coordinates[3]),
                    (210, 58, 134),
                    3,
                )
                cv2.imwrite(
                    os.path.join(
                        settings.BOXED_IMAGES_FOLDER, os.path.basename(path)
                    ),
                    img,
                )
                return 0, cropped_image_path
            else:
                return 1, SMALL_IMG_SIZE_MSG
        else:
            return 1, LOW_AREA_MSG
    return 2, NOT_CAR_MSG


def send_response(code, msg, job_id):
    """
    Sends a response to the client when it is not able to find a vehicle or it is not
    big enough.
    Otherwise, it sends the job to the next model queue.
    """
    if code == 0:
        mobileNet_job_data = {"id": job_id, "image_path": msg}
        db.lpush(settings.MOBILENET_QUEUE, json.dumps(mobileNet_job_data))
    else:
        result_dict = {
            "prediction_1": "N/A",
            "score_1": None,
            "plate_number": "N/A",
            "message": msg,
        }
        db.set(job_id, json.dumps(result_dict))


def classify_process():
    """
    Loops indefinitely asking Redis for new jobs.
    When a new job arrives, takes it from the Redis queue, uses the loaded ML
    model to get predictions and stores the results back in Redis using
    the original job ID so other services can see it was processed and access
    the results.
    """
    while True:
        job_to_process = json.loads(db.brpop(settings.REDIS_QUEUE)[1])
        if job_to_process:
            img_path = os.path.join(
                settings.UPLOAD_FOLDER, job_to_process["image_name"]
            )
            start_time = time.time()
            code, message = generate_cropped_image(img_path)
            response_time = time.time() - start_time
            print(f"Car Detector response time: {response_time}", flush=True)            
            send_response(code, message, job_to_process["id"])
        time.sleep(settings.SERVER_SLEEP)


if __name__ == "__main__":
    # Now launch process
    print("Launching Car Detector service...")
    classify_process()
