import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch
import cv2
import numpy as np
from anpr.pipeline import localize, recognize, read_image


class PipelineTests(unittest.TestCase):
    def test_blank_image_uses_full_image(self):
        image = np.zeros((80, 160, 3), dtype=np.uint8)
        crop, origin, contour, _ = localize(image)
        self.assertIsNone(contour)
        self.assertEqual(origin, (0, 0))
        self.assertEqual(crop.shape, image.shape[:2])

    def test_rectangular_candidate_is_cropped(self):
        image = np.zeros((160, 300, 3), dtype=np.uint8)
        cv2.rectangle(image, (60, 50), (240, 110), (255, 255, 255), -1)
        crop, (x, y), contour, _ = localize(image)
        self.assertIsNotNone(contour)
        self.assertLess(crop.shape[0], image.shape[0])
        self.assertGreater(x, 0)
        self.assertGreater(y, 0)

    def test_ocr_boxes_are_translated_and_text_combined(self):
        class Reader:
            def readtext(self, image):
                return [([[1, 2], [8, 2], [8, 7], [1, 7]], 'ab12', .9),
                        ([[9, 2], [18, 2], [18, 7], [9, 7]], 'cd', .8)]
        image = np.zeros((50, 70, 3), dtype=np.uint8)
        contour = np.array([[[10, 20]], [[30, 20]], [[30, 40]], [[10, 40]]])
        with patch('anpr.pipeline.localize', return_value=(image[:20,:20], (10,20), contour, {})):
            result, _ = recognize(image, Reader())
        self.assertEqual(result['detections'][0]['box'][0], [11.,22.])
        self.assertEqual(result['text'], 'AB12 CD')
        self.assertFalse(result['fallback'])

    def test_empty_ocr_result(self):
        class Reader:
            def readtext(self, image): return []
        result, _ = recognize(np.zeros((20,30,3),dtype=np.uint8), Reader())
        self.assertEqual(result['text'], '')
        self.assertTrue(result['fallback'])

    def test_unreadable_file(self):
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory) / 'invalid.jpg'
            file.write_text('invalid')
            with self.assertRaises(ValueError): read_image(file)


if __name__ == '__main__': unittest.main()
