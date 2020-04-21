import unittest

import numpy as np

import preprocessor


class TestBatchSplitImg(unittest.TestCase):
    patch_size = 4
    stride = 2
    batchsplitimg = preprocessor.BatchSplitImg(patch_size, stride)
    batch_img = np.zeros((10, 3, 6, 8))

    def test_init(self):
        self.assertEqual(self.batchsplitimg.patch_size, self.patch_size)
        self.assertEqual(self.batchsplitimg.stride, self.stride)

    def test_call(self):
        ret = self.batchsplitimg(self.batch_img)
        self.assertEqual(ret.shape, (10, 6, 3, 4, 4))


class TestHWC2CHWandToTensor(unittest.TestCase):
    hwc2chw = preprocessor.HWC2CHW()
    totensor = preprocessor.ToTensor()

    def test_call(self):
        image = np.zeros([1, 2, 3])
        ret_image = self.hwc2chw(image)
        self.assertEqual(ret_image.shape, (3, 1, 2))

        tensor = self.totensor(ret_image)
        self.assertEqual(tensor.shape, (3, 1, 2))


class TestGray2RGB(unittest.TestCase):
    gray2rgb = preprocessor.Gray2RGB()

    def test_call(self):
        gray_img = np.zeros([1, 2, 1])
        color_img = np.zeros([1, 2, 3])
        ret_gray_img = self.gray2rgb(gray_img)
        ret_color_img = self.gray2rgb(color_img)
        self.assertEqual(ret_gray_img.shape, (1, 2, 3))
        self.assertEqual(ret_color_img.shape, (1, 2, 3))


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


class TestTransformForTorchModel(unittest.TestCase):
    trans = preprocessor.TransformForTorchModel()

    def test_call(self):
        image = np.ones([3, 1, 1])
        ret_image = self.trans(image)
        self.assertAlmostEqual(ret_image[0][0][0], 2.2489083)
        self.assertAlmostEqual(ret_image[1][0][0], 2.42857143)
        self.assertAlmostEqual(ret_image[2][0][0], 2.64)


class DivideBy255(unittest.TestCase):
    divide = preprocessor.DivideBy255()

    def test_call(self):
        image = np.ones([3, 1, 1])
        ret_image = self.divide(image)
        self.assertAlmostEqual(ret_image[0][0][0], 1 / 255)


class VGG16ScaledFeatures(unittest.TestCase):
    cutoff_edge_width = 2
    vgg16 = preprocessor.VGG16ScaledFeatures(
        cutoff_edge_width=cutoff_edge_width)

    def test_init(self):
        self.assertEqual(self.vgg16.cutoff_edge_width, 2)

    def test_call(self):
        image = np.ones([1, 3, 64, 64])
        totensor = preprocessor.ToTensor()
        tensor = totensor(image)
        ret = self.vgg16(tensor)
        self.assertEqual(ret.shape, (1, 896, 4, 4))
