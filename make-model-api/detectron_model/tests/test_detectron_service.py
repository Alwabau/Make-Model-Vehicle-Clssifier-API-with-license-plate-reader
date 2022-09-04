import unittest
import detectron_service


class TestMLService(unittest.TestCase):
    """
    Tests if Detectron2 model predicts correctly
    """

    def test_predict(self):

        # Predict a car image
        prediction = detectron_service.generate_cropped_image(
            "/src/tests/ford_ranger.jpg"
        )
        self.assertEqual(
            prediction, (0, "uploads/cropped_images/ford_ranger.jpg")
        )

        # Predict a small size image
        prediction_2 = detectron_service.generate_cropped_image(
            "/src/tests/smallcar.jpg"
        )
        self.assertEqual(
            prediction_2,
            (1, "Small image size please try again with a bigger image"),
        )

        # Predict a small area car(less than 20% of the image surface)
        prediction_3 = detectron_service.generate_cropped_image(
            "/src/tests/far.jpg"
        )
        self.assertEqual(
            prediction_3,
            (
                1,
                "Low vehicle area, please submit a closer picture of the car",
            ),
        )

        # Predict there is no car
        prediction_4 = detectron_service.generate_cropped_image(
            "/src/tests/motorbike.jpg"
        )
        self.assertEqual(
            prediction_4,
            (
                2,
                "We could not find a car in the image, please submit a car picture",
            ),
        )


if __name__ == "__main__":
    unittest.main()
