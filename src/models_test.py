import os
import shutil
import unittest

import numpy as np
import yaml

import models


class TestCalculateScore(unittest.TestCase):

    with open("./cfg/sample_config.yml") as f:
        config = yaml.load(f, yaml.SafeLoader)

    np.random.seed(config["seed"])
    model_params = config["model_params"]

    model = models.SparseCodingWithMultiDict([], model_params)

    def test_output_image(self):
        is_positive = True
        batch_name = "image.png"
        ch_err = np.ones([896])
        output_img = np.zeros([10, 10, 3])
        self.model.output_image(is_positive, batch_name, ch_err, output_img)
        self.assertTrue(os.path.exists("visualized_results/pos/image-896.png"))
        shutil.rmtree("visualized_results")

    def test_calclate_ssim(self):
        arr1 = np.zeros([1, 3, 512])
        arr2 = np.ones([1, 3, 512])
        dim = (1, 3, 16, 32)  # P, C, H, W
        self.assertAlmostEqual(
            self.model.calculate_ssim(arr1, arr2, dim)[0], 0.0, delta=0.01)
        self.assertAlmostEqual(
            self.model.calculate_ssim(arr1, arr1, dim)[0], -1.0)

    def test_calculate_score(self):
        dn = [1, 4]
        dp1 = [5, 6]
        dp2 = [2, 3]

        ap1, roc1 = self.model.calculate_score(dn, dp1)
        self.assertAlmostEqual(ap1, 1.0)
        self.assertAlmostEqual(roc1, 1.0)

        ap2, roc2 = self.model.calculate_score(dn, dp2)
        self.assertAlmostEqual(ap2, 0.5833333)
        self.assertAlmostEqual(roc2, 0.5)

    def test_reconst_from_array(self):
        arr = np.ones([121, 64])
        self.assertEqual(self.model.reconst_from_array(arr).shape, (1, 28, 28))
