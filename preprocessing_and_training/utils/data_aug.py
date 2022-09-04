from tensorflow import keras
def create_data_aug_layer(data_aug_layer):
    """
    Use this function to parse the data augmentation methods for the
    experiment and create the corresponding layers.

    It will be mandatory to support at least the following three data
    augmentation methods (you can add more if you want):
        - `random_flip`: keras.layers.RandomFlip()
        - `random_rotation`: keras.layers.RandomRotation()
        - `random_zoom`: keras.layers.RandomZoom()

    See https://tensorflow.org/tutorials/images/data_augmentation.

    Parameters
    ----------
    data_aug_layer : dict
        Data augmentation settings coming from the experiment YAML config
        file.

    Returns
    -------
    data_augmentation : keras.Sequential
        Sequential model having the data augmentation layers inside.
    """
    data_aug_layers = []
    if data_aug_layer:
        if("random_flip" in data_aug_layer):
            data_aug_layers.append(keras.layers.RandomFlip(**data_aug_layer["random_flip"]))
        if("random_rotation" in data_aug_layer):
            data_aug_layers.append(keras.layers.RandomRotation(**data_aug_layer["random_rotation"]))
        if("random_zoom" in data_aug_layer):
            data_aug_layers.append(keras.layers.RandomZoom(**data_aug_layer["random_zoom"]))
        if("random_brightness" in data_aug_layer):
            data_aug_layers.append(keras.layers.RandomBrightness(**data_aug_layer["random_brightness"]))             
    data_augmentation = keras.Sequential(data_aug_layers)

    return data_augmentation
