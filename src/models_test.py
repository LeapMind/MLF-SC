import unittest

import numpy
import yaml

import models


class TestCalculateScore(unittest.TestCase):

    with open("./cfg/sample_config.yml") as f:
        config = yaml.load(f, yaml.SafeLoader)

    numpy.random.seed(config["seed"])
    model_params = config["model_params"]

    model = models.SparseCodingWithMultiDict([], model_params)

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
