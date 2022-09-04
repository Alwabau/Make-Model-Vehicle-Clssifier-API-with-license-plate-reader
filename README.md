# MAKE-MODEL API

This project takes vehicle images and returns, on the one hand, the make and model of 611 (originally 836; some were merged, e.g. 'abarth_500c' and 'abarth_500' into 'abarth_500') European vehicle classes, and on the other, the European vehicle registration plate. 

Appropriate feedback is given if there is no image file, if the image is too small, if the image shows the inside of the car and if the license plate can't be read either because there is none or because the letters and numbers are too small to be read. The country ID letters should be readable for the model to take the whole plate as valid.

## Structure

The structure of the project consists of two separate parts, the MAKE-MODEL API and the Preprocessing and Training files. Each of these folders have their own README files and their p1_* files that indicate the structure of both and what each folder and file does.

