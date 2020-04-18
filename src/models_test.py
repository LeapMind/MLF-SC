import unittest

import models


class TestCalculateScore(unittest.TestCase):

    model = models.SparseCodingWithMultiDict([], 0, 0, None, 0, 0, 0, 0)

    def test_calculate_score_1(self):
        dn = [1, 2]
        dp = [3, 4]
        ap, roc = self.model.calculate_score(dn, dp)
        self.assertAlmostEqual(ap, 1.0)
        self.assertAlmostEqual(roc, 1.0)

    def test_calculate_score_2(self):
        dn = [1, 4]
        dp = [2, 3]
        ap, roc = self.model.calculate_score(dn, dp)
        self.assertAlmostEqual(ap, 0.5833333)
        self.assertAlmostEqual(roc, 0.5)
