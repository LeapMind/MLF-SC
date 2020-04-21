import unittest

import dataset


class TestMVTecDataset(unittest.TestCase):

    dir_env = dict()
    dir_env["ext"] = ".png"
    dir_env["root"] = "./sample_data/"
    dir_env["train_good_dir"] = "train/good"
    dir_env["test_good_dir"] = None
    dir_env["test_bad_dir"] = None

    mvtec_dataset = dataset.MVTecDataset(is_train=True, dir_env=dir_env)

    def test_dataset(self):
        self.assertEqual(len(self.mvtec_dataset), 10)
