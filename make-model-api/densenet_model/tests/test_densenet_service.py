import unittest
import densenet_service


class TestIntegration(unittest.TestCase):
    '''
    Tests if the predicted results are correct
    '''

    def test_predict(self):
        class_pred = densenet_service.predict("/src/tests/ford_ranger.jpg")

        self.assertEqual(class_pred[0]['make_model'], "ford_ranger")
        self.assertEqual(class_pred[0]['pred_score'], '0.9894895')

        self.assertEqual(class_pred[1]['make_model'], "ford_f-150")
        self.assertEqual(class_pred[1]['pred_score'], '0.009471423')

        self.assertEqual(class_pred[2]['make_model'], "dodge_ram")
        self.assertEqual(class_pred[2]['pred_score'], '0.0005521297')

if __name__ == "__main__":
    unittest.main()
