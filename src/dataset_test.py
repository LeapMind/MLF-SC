import unittest

import dataset


class TestDatasetandDataLoader(unittest.TestCase):

    dir_env = dict()
    dir_env["ext"] = ".png"
    dir_env["root"] = "./sample_data/"
    dir_env["train_good_dir"] = "train/good"
    dir_env["test_good_dir"] = "test/good"
    dir_env["test_bad_dir"] = None

    mvtec_dataset_train = dataset.MVTecDataset(is_train=True, dir_env=dir_env)
    mvtec_dataset_test = dataset.MVTecDataset(is_train=False, dir_env=dir_env)
    dataloader_test = dataset.DataLoader(
        mvtec_dataset_test,
        batch_size=2,
        shuffle=True,
        drop_last=False,
    )

    def test_dataset(self):
        self.assertEqual(len(self.mvtec_dataset_train), 10)
        self.assertEqual(len(self.mvtec_dataset_test), 2)
        self.assertEqual(len(self.mvtec_dataset_test[0]), 3)

    def test_dataloader(self):
        self.assertEqual(len(self.dataloader_test), 2)
        ret = 0
        for _ in self.dataloader_test:
            ret += 1
        self.assertEqual(ret, 2 // 2)
