import os

# Stores images uploaded by the user on this folder
UPLOAD_FOLDER = "uploads/"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# REDIS settings

# Queue names
MOBILENET_QUEUE = 'mobilenet_queue'
DENSENET_QUEUE = 'densenet_queue'
OCR_QUEUE = 'ocr_queue'

# Port
REDIS_PORT = 6379

# DB Id
REDIS_DB_ID = 0

# Host IP
REDIS_IP = 'redis'

# Sleep parameters which manages the interval between requests to our redis queue
SERVER_SLEEP = 0.05
