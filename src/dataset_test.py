import unittest

import dataset


class TestMVTecDataset(unittest.TestCase):

    mvtec_dataset = dataset.MVTecDataset("./sample_data/", ".png", "train", neg_dir="train/good")

    def test_dataset(self):
        self.assertEqual(len(self.mvtec_dataset), 10)
