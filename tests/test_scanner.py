import unittest
from preprocess import load_and_preprocess_image
from contour_detection import find_document_contour, warp_perspective
import cv2

class TestDocumentScanner(unittest.TestCase):

    def setUp(self):
        self.image_path = 'images/sample.jpg'
        self.image, self.edged = load_and_preprocess_image(self.image_path)

    def test_find_document_contour(self):
        contour = find_document_contour(self.edged)
        self.assertIsNotNone(contour, "Document contour should be found")

    def test_warp_perspective(self):
        contour = find_document_contour(self.edged)
        if contour is not None:
            warped = warp_perspective(self.image, contour)
            self.assertEqual(warped.shape[0] > 0, True, "Warped image should have positive height")
            self.assertEqual(warped.shape[1] > 0, True, "Warped image should have positive width")

if __name__ == '__main__':
    unittest.main()
