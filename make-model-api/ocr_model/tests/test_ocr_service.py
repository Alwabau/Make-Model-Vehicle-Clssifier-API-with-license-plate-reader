import unittest

import ocr_service


class TestMLService(unittest.TestCase):
    """
    Tests all different possible feedbacks from the vehicle registration plate
    model.
    """

    def test_plate_reader(self):

        # Predict normal European plate
        car_1 = ocr_service.plate_reader("/src/tests/cartest1.jpg")
        self.assertEqual(car_1, "RO B396JOY")

        # Predict European plate with text under the plate and with e-mail
        car_2 = ocr_service.plate_reader("/src/tests/cartest2.jpg")
        self.assertEqual(car_2, "BG CA7552PP")

        # Predict European plate with text above and under the plate
        car_3 = ocr_service.plate_reader("/src/tests/cartest3.jpg")
        self.assertEqual(car_3, "UA AA6888KI")

        # Predict European plate with an angle and separated
        car_4 = ocr_service.plate_reader("/src/tests/cartest4.jpg")
        self.assertEqual(car_4, "NL BT4732")

        # Predict European plate with unclear characters
        car_5 = ocr_service.plate_reader("/src/tests/cartest5.jpg")
        self.assertEqual(
            car_5,
            "There is no European vehicle registration plate or the characters are unclear",
        )

        # Predict no European plate
        car_6 = ocr_service.plate_reader("/src/tests/cartest6.jpg")
        self.assertEqual(
            car_6,
            "There is no European vehicle registration plate or the characters are unclear",
        )


if __name__ == "__main__":
    unittest.main()
