import os
import hashlib
from fileinput import filename


def allowed_file(filename):
    """
    Checks if the format for the file received is acceptable. For this
    particular case, we must accept only image files.

    Parameters
    ----------
    filename : str
        Filename from werkzeug.datastructures.FileStorage file.

    Returns
    -------
    bool
        True if the file is an image, False otherwise.
    """
    ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif"}
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


def get_file_hash(file):
    """
    Returns a new filename based on the file content using MD5 hashing.
    It uses hashlib.md5() function from Python standard library to get
    the hash.

    Parameters
    ----------
    file : werkzeug.datastructures.FileStorage
        File sent by user.

    Returns
    -------
    str
        New filename based in md5 file hash.
    """
    _, ext = os.path.splitext(file.filename)
    file_content = file.read()
    file_hash = hashlib.md5(file_content)
    hashed_file_name = file_hash.hexdigest() + ext
    file.seek(0)
    return hashed_file_name
