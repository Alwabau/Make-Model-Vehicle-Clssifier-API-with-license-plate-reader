import unittest

import mobilenet_service


class TestMLService(unittest.TestCase):
    """
    Tests if the vehicle view is external or internal
    """

    def test_predict(self):

        car_external = mobilenet_service.predict("/src/tests/car_external.jpg")
        self.assertEqual(car_external, True)

        car_internal = mobilenet_service.predict("/src/tests/car_internal.jpg")
        self.assertEqual(car_internal, False)


if __name__ == "__main__":
    unittest.main()
