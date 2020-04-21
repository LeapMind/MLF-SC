import unittest

import models


class TestCalculateScore(unittest.TestCase):

    model = models.SparseCodingWithMultiDict([], 0, 0, None, 0, 0, 0, 0)

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
