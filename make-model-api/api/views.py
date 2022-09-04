import utils
import settings
import os
import redis
import re
from middleware import model_predict
from genericpath import exists

from flask import (
    Blueprint,
    flash,
    redirect,
    render_template,
    request,
    url_for,
    jsonify,
)
import json

router = Blueprint("app_router", __name__, template_folder="templates")


@router.route("/", methods=["GET"])
def index():
    """
    Index endpoint, renders our HTML code
    """
    return render_template("index.html")


def clean_class_output(class_name):
    """
    Cleans the output class name to make it readable
    """
    out_class = class_name
    out_class = re.sub("_", " ", out_class)
    out_class = re.sub("-", " ", out_class)
    out_class = out_class.title()
    return out_class


def transform_string_percent(value):
    """
    Transforms float string with predictions percentage to int
    """
    value = float(value)
    return int(value * 100)


@router.route("/", methods=["POST"])
def upload_image():
    """
    Function used in our frontend so we can upload and show an image.
    When it receives an image from the UI, it also calls our ML model to
    get and display the predictions
    """
    # No file received, show basic UI
    if "file" not in request.files:
        flash("No file part")
        return redirect(request.url)

    # File received but no filename is provided, show basic UI
    file = request.files["file"]

    if file.filename == "":
        flash("No image selected for uploading")
        return redirect(request.url)

    # File received and it's an image, we show it and get predictions
    if file and utils.allowed_file(file.filename):
        img_name = utils.get_file_hash(file)
        path = settings.UPLOAD_FOLDER + img_name

        if not os.path.exists(path):
            file.save(path, buffer_size=16384)
        file.close()
        response = model_predict(img_name)
        context = {}

        if response["prediction_1"] == "N/A":
            print(os.path.join(settings.UPLOAD_FOLDER, img_name), flush=True)
            context = {
                "top1_pred": response["prediction_1"],
                "top1_score": response["score_1"],
                "top2_pred": response["prediction_1"],
                "top2_score": response["score_1"],
                "top3_pred": response["prediction_1"],
                "top3_score": response["score_1"],
                "plate_number": response["plate_number"],
                "message": response["message"],
                "filename": os.path.join(settings.UPLOAD_FOLDER, img_name),
            }
        else:

            context = {
                "top1_pred": clean_class_output(response["prediction_1"]),
                "top1_score": transform_string_percent(response["score_1"]),
                "top2_pred": clean_class_output(response["prediction_2"]),
                "top2_score": transform_string_percent(response["score_2"]),
                "top3_pred": clean_class_output(response["prediction_3"]),
                "top3_score": transform_string_percent(response["score_3"]),
                "plate_number": response["plate_number"],
                "message": response["message"],
                "filename": os.path.join(
                    settings.BOXED_IMAGES_FOLDER, img_name
                ),
            }

        # Update `render_template()` parameters
        return render_template(
            "index.html",
            filename=context["filename"],
            context=context,
            anchor="about",
        )
    # File received but it isn't an image
    else:
        flash("Allowed image types are -> png, jpg, jpeg, gif")
        return redirect(request.url)


@router.route("/display/<filename>")
def display_image(filename):
    """
    Displays uploaded image in our UI
    """
    return redirect(
        # url_for("static", filename="uploads/" + filename), code=301
        url_for("static", filename=filename),
        code=301,
    )


@router.route("/predict", methods=["POST"])
def predict():
    """
    Endpoint used to get predictions without need to access the UI.

    Parameters
    ----------
    file : str
        Input image we want to get predictions from.

    Returns
    -------
    flask.Response
        JSON response from our API having the following format:
            {[
            {"make_model": str, "pred_score": int},
            {"make_model": str, "pred_score": int},
            {"make_model": str, "pred_score": int},
            ]}

        - "make_model" model predicted class as string.
        - "pred_score" model confidence score for the predicted class as integer.
    """
    rpse = [
        {"make_model": None, "pred_score": None},
        {"make_model": None, "pred_score": None},
        {"make_model": None, "pred_score": None},
    ]

    if "file" not in request.files:
        return jsonify(rpse), 400

    file = request.files["file"]  # Uses file from request.files
    if file.filename == "":
        return jsonify(rpse), 400

    hashed_file_name = utils.get_file_hash(file)
    if not utils.allowed_file(
        file.filename
    ):  # Makes sure the file extension is an image
        return jsonify(rpse), 400

    path = settings.UPLOAD_FOLDER + hashed_file_name
    if not os.path.exists(path):
        file.save(path, buffer_size=16384)
    file.close()

    class_pred = model_predict(hashed_file_name)

    class_pred_1 = class_pred[0]["make_model"]
    score_pred_1 = class_pred[0]["pred_score"]
    class_pred_2 = class_pred[1]["make_model"]
    score_pred_2 = class_pred[1]["pred_score"]
    class_pred_3 = class_pred[2]["make_model"]
    score_pred_3 = class_pred[2]["pred_score"]

    rpse = [
        {"make_model": class_pred_1, "pred_score": score_pred_1},
        {"make_model": class_pred_2, "pred_score": score_pred_2},
        {"make_model": class_pred_3, "pred_score": score_pred_3},
    ]

    return jsonify(rpse)


@router.route("/feedback", methods=["GET", "POST"])
def feedback():
    """
    Stores feedback from users about wrong predictions on a text file.

    Parameters
    ----------
    report : request.form
        Feedback given by the user with the following JSON format:
            {
                "filename": str,
                "prediction": str,
                "score": float,
                "plate_number": str,
                "message": str
            }

        - "filename" corresponds to the image used stored in the uploads
          folder.
        - "prediction" is the model predicted class as string reported as
          incorrect.
        - "score" model confidence score for the predicted class as float.
        - "plate_number": European plate number predicted.
        - "message": Prediction feedback to the user.
    """

    report = request.form.get("report")

    feedback_path = settings.FEEDBACK_FILEPATH
    open(feedback_path, "a").close()  # Creates the file if it does not exist
    feedback_file = open(feedback_path, "a")
    if report:
        feedback_file.write(report + "\n")
    feedback_file.close()

    return render_template("index.html")
