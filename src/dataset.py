import os

import cv2
import numpy
from tqdm import tqdm


class MVTecDataset(object):
    def __init__(
            self, is_train, dir_env, is_positive=True, preprocessor=None,
    ):

        ext = dir_env["ext"]
        root = dir_env["root"]
        train_good_dir = dir_env["train_good_dir"]
        test_good_dir = dir_env["test_good_dir"]
        test_bad_dir = dir_env["test_bad_dir"]
        self.preprocessor = preprocessor

        if ext[0] != ".":
            ext = "." + ext

        if is_train:
            dir_name = train_good_dir
        elif not is_positive:
            dir_name = test_good_dir
        elif test_bad_dir:
            dir_name = test_bad_dir
        else:
            dir_name = None
            excp_name = test_good_dir

        if dir_name:
            dir_path = os.path.abspath(os.path.join(root, dir_name))
            dir_parent_path = os.path.dirname(dir_path)
            dir_name = os.path.basename(dir_path)
            self.dataset = self.load_dataset(dir_parent_path, dir_name, ext)
        else:
            dir_path = os.path.abspath(os.path.join(root, excp_name))
            dir_parent_path = os.path.dirname(dir_path)
            dirs = [
                d
                for d in os.listdir(dir_parent_path)
                if d not in excp_name
            ]
            self.dataset = []
            for dir_name in dirs:
                self.dataset.extend(self.load_dataset(
                    dir_parent_path, dir_name, ext))

    def load_dataset(self, dir_parent_path, dir_name, ext):

        return [
            (dir_name, f, cv2.imread(os.path.join(
                dir_parent_path, dir_name, f))
             [:, :, [2, 1, 0]])
            for f in tqdm(
                os.listdir(os.path.join(dir_parent_path, dir_name)),
                desc="loading images"
            )
            if ext in f
        ]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx][2]

        if self.preprocessor:
            for p in self.preprocessor:
                sample = p(sample)

        return (self.dataset[idx][0], self.dataset[idx][1], sample)


class DataLoader(object):
    def __init__(self, dataset, batch_size, shuffle, drop_last):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.counter = 0
        self.idxs = numpy.arange(len(self.dataset))

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        return self

    def __next__(self):
        if self.counter == 0 and self.shuffle:
            numpy.random.shuffle(self.idxs)

        if (
            self.counter + self.batch_size > len(self.dataset)
            and self.drop_last
        ):
            self.counter = 0
            raise StopIteration()

        if self.counter >= len(self.dataset):
            self.counter = 0
            raise StopIteration()

        batch = []
        for idx in self.idxs[self.counter: self.counter + self.batch_size]:
            batch.append(self.dataset[idx][2])

        self.counter += self.batch_size

        return self.dataset[idx][0], self.dataset[idx][1], numpy.stack(batch)
