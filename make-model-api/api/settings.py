import os

# Runs API in Debug mode
API_DEBUG = True

# Stores images uploaded by the user on this folder
UPLOAD_FOLDER = "static/uploads/"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
BOXED_IMAGES_FOLDER = "static/uploads/boxed_images"
os.makedirs(BOXED_IMAGES_FOLDER, exist_ok=True)

# Stores user's feedback on this file
FEEDBACK_FILEPATH = "feedback/feedback"
os.makedirs(os.path.basename(FEEDBACK_FILEPATH), exist_ok=True)

# REDIS settings

# Queue name
REDIS_QUEUE = "app_queue"
MOBILENET_QUEUE = "mobilenet_queue"

# Port
REDIS_PORT = 6379

# DB Id
REDIS_DB_ID = 0

# Host IP
REDIS_IP = "redis"

# Sleep parameters which manages the interval between requests to our redis queue
API_SLEEP = 0.05
