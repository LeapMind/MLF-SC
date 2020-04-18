import unittest

import numpy as np

import preprocessor


class TestResize(unittest.TestCase):
    size = (256, 256)  # (width, height)
    resize = preprocessor.Resize(size)

    def test_init(self):
        self.assertEqual(self.resize.size, self.size)

    def test_call(self):
        channel = 3
        image = np.zeros([500, 600, channel])
        resized_image = self.resize(image)

        self.assertEqual(resized_image.shape,
                         (self.size[1], self.size[0], channel))


class TestHWC2CHW(unittest.TestCase):

    hwc2chw = preprocessor.HWC2CHW()

    def test_call(self):
        image = np.zeros([1, 2, 3])
        ret_image = self.hwc2chw(image)
        self.assertEqual(ret_image.shape, (3, 1, 2))
