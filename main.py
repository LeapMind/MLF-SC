import argparse
import os

import numpy
import yaml

import src.dataset as dataset
import src.models as models
import src.preprocessor as preprocessor


def ini_file(d):
    try:
        assert os.path.isfile(d)
        return d
    except Exception:
        raise argparse.ArgumentTypeError(
            "ini file {} cannot be located.".format(d))


def main():
    parser = argparse.ArgumentParser(description="Anomaly Detection")
    parser.add_argument("split", nargs="?", choices=["train", "test"])
    parser.add_argument("ini", nargs="?", type=ini_file, help="inifile name")

    args = parser.parse_args()

    with open(args.ini) as f:
        config = yaml.load(f, yaml.SafeLoader)

    numpy.random.seed(config["seed"])
    paths = config["paths"]
    model_params = config["model_params"]

    preprocesses = []
    preprocesses.append(preprocessor.Resize((512, 512)))
    preprocesses.append(preprocessor.Gray2RGB())
    preprocesses.append(preprocessor.HWC2CHW())
    preprocesses.append(preprocessor.DivideBy255())
    preprocesses.append(preprocessor.TransformForTorchModel())

    model_preprocesses = []
    model_preprocesses.append(preprocessor.ToTensor())
    model_preprocesses.append(
        preprocessor.ResNet50ScaledFeatures(
            last_layer=50, cutoff_edge_width=model_params["cutoff_edge_width"]
        )
    )
    model_preprocesses.append(
        preprocessor.BatchSplitImg(
            patch_size=model_params["patch_size"],
            stride=model_params["stride"],
        )
    )

    if args.split == "train":
        train_dataset = dataset.MVTecDataset(
            is_train=True,
            dir_env=paths,
            preprocessor=preprocesses,
        )
        train_loader = dataset.DataLoader(
            train_dataset,
            batch_size=model_params["batch_size"],
            shuffle=True,
            drop_last=False,
        )

        model = models.SparseCodingWithMultiDict(
            preprocesses=model_preprocesses,
            model_env=model_params,
            train_loader=train_loader,
        )
        model.train()
        model.save_dict(paths["dict_file"])

    elif args.split == "test":
        test_neg_dataset = dataset.MVTecDataset(
            is_train=False,
            dir_env=paths,
            is_positive=False,
            preprocessor=preprocesses,
        )
        test_pos_dataset = dataset.MVTecDataset(
            is_train=False,
            dir_env=paths,
            is_positive=True,
            preprocessor=preprocesses,
        )

        test_neg_loader = dataset.DataLoader(
            test_neg_dataset, batch_size=1, shuffle=False, drop_last=False
        )
        test_pos_loader = dataset.DataLoader(
            test_pos_dataset, batch_size=1, shuffle=False, drop_last=False
        )

        model = models.SparseCodingWithMultiDict(
            preprocesses=model_preprocesses,
            model_env=model_params,
            test_neg_loader=test_neg_loader,
            test_pos_loader=test_pos_loader,
        )
        model.load_dict(paths["dict_file"])
        model.test()


if __name__ == "__main__":
    main()
