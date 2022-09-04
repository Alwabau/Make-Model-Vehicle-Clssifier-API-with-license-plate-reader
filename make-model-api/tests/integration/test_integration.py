import unittest
import requests


class TestIntegration(unittest.TestCase):
    '''
    Performs an integration test to see if the whole system works
    '''
    def test_index(self):
        '''
        Tests the index to see if there are correct get and post answers 
        '''
        response = requests.request("GET", "http://0.0.0.0/",)
        self.assertEqual(response.status_code, 200)

        response = requests.request("POST", "http://0.0.0.0/",)
        self.assertEqual(response.status_code, 200)

    def test_predict(self):
        '''
        Tests the system through the endpoint used to get predictions without need to access the UI
        '''
        files = [
            ("file", ("ford_ranger.jpg", open("tests/integration/ford_ranger.jpg", "rb"), "image/jpg"))
        ]
        headers = {}
        payload = {}
        response = requests.request(
            "POST",
            "http://0.0.0.0/predict",
            headers=headers,
            data=payload,
            files=files,
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data[0]['make_model'], "ford_ranger")
        self.assertEqual(data[0]['pred_score'], '0.9894895')

        self.assertEqual(data[1]['make_model'], "ford_f-150")
        self.assertEqual(data[1]['pred_score'], '0.009471423')

        self.assertEqual(data[2]['make_model'], "dodge_ram")
        self.assertEqual(data[2]['pred_score'], '0.0005521297')


if __name__ == "__main__":
    unittest.main()
