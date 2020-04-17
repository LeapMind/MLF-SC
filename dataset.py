import os
import skimage.io
import numpy
from tqdm import tqdm


class MVTecDataset(object):
    def __init__(
        self, root, ext, train, mode=None, neg_dir=None, pos_dir=None, preprocessor=None
    ):
        if train:
            if not neg_dir:
                raise ValueError("The argument 'neg_dir' must be set.")

        else:
            if not (mode == "neg" or mode == "pos"):
                raise ValueError("The argument 'mode' must be set to 'neg' or 'pos'.")

            if mode == "neg" and not neg_dir:
                raise ValueError("The argument 'neg_dir' must be set.")

            if mode == "pos" and (not pos_dir and not neg_dir):
                raise ValueError(
                    "The argument 'neg_dir' must be set in order to consider data "
                    "contained in other directories as positive samples"
                )

        self.preprocessor = preprocessor
        if ext[0] != ".":
            ext = "." + ext

        if train:
            dir_path = os.path.abspath(os.path.join(root, neg_dir))
            self.dataset = [
                (f, skimage.io.imread(os.path.join(dir_path, f)))
                for f in tqdm(os.listdir(dir_path), desc="loading dataset for training")
                if ext in f
            ]
        else:
            if mode == "neg":
                dir_path = os.path.abspath(os.path.join(root, neg_dir))
                self.dataset = [
                    (f, skimage.io.imread(os.path.join(dir_path, f)))
                    for f in tqdm(
                        os.listdir(dir_path),
                        desc="loading negative dataset for testing",
                    )
                    if ext in f
                ]
            else:
                if pos_dir:
                    dir_path = os.path.abspath(os.path.join(root, pos_dir))
                    self.dataset = [
                        (f, skimage.io.imread(os.path.join(dir_path, f)))
                        for f in tqdm(
                            os.listdir(dir_path),
                            desc="loading positive dataset for testing",
                        )
                        if ext in f
                    ]
                else:
                    dir_path = os.path.abspath(os.path.join(root, neg_dir))
                    dir_parent_path = os.path.dirname(dir_path)

                    dir_paths = [
                        os.path.join(dir_parent_path, d)
                        for d in os.listdir(dir_parent_path)
                        if d not in neg_dir
                    ]
                    self.dataset = [
                        (f, skimage.io.imread(os.path.join(d, f)))
                        for d in tqdm(
                            dir_paths, desc="loading positive dataset for testing"
                        )
                        for f in os.listdir(d)
                        if ext in f
                    ]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx][1]

        if self.preprocessor:
            for p in self.preprocessor:
                sample = p(sample)

        return (self.dataset[idx][0], sample)


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
            batch.append(self.dataset[idx][1])

        self.counter += self.batch_size

        return self.dataset[idx][0], numpy.stack(batch)
