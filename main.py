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
        raise argparse.ArgumentTypeError("ini file {} cannot be located.".format(d))


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
    preprocesses.append(preprocessor.Resize((256, 256)))
    preprocesses.append(preprocessor.Gray2RGB())
    preprocesses.append(preprocessor.HWC2CHW())
    preprocesses.append(preprocessor.DivideBy255())
    preprocesses.append(preprocessor.TransformForTorchModel())

    model_preprocesses = []
    model_preprocesses.append(preprocessor.ToTensor())
    model_preprocesses.append(
        preprocessor.VGG16ScaledFeatures(
            last_layer=22, cutoff_edge_width=model_params["cutoff_edge_width"]
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
            root=paths["root"],
            ext=paths["ext"],
            train=True,
            neg_dir=paths["train_good_dir"],
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
            num_of_basis=model_params["num_of_basis"],
            alpha=model_params["alpha"],
            transform_algorithm=model_params["transform_algorithm"],
            transform_alpha=model_params["transform_alpha"],
            fit_algorithm=model_params["fit_algorithm"],
            n_iter=model_params["n_iter"],
            num_of_nonzero=model_params["num_of_nonzero"],
            train_loader=train_loader,
        )
        model.train()
        model.save_dict(paths["dict_file"])

    elif args.split == "test":
        test_neg_dataset = dataset.MVTecDataset(
            root=paths["root"],
            ext=paths["ext"],
            train=False,
            mode="neg",
            neg_dir=paths["test_good_dir"],
            preprocessor=preprocesses,
        )
        if paths["test_bad_dir"] is None:
            test_pos_dataset = dataset.MVTecDataset(
                root=paths["root"],
                ext=paths["ext"],
                train=False,
                mode="pos",
                neg_dir=paths["test_good_dir"],
                preprocessor=preprocesses,
            )
        else:
            test_pos_dataset = dataset.MVTecDataset(
                root=paths["root"],
                ext=paths["ext"],
                train=False,
                mode="pos",
                pos_dir=paths["test_bad_dir"],
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
            num_of_basis=model_params["num_of_basis"],
            alpha=model_params["alpha"],
            transform_algorithm=model_params["transform_algorithm"],
            transform_alpha=model_params["transform_alpha"],
            fit_algorithm=model_params["fit_algorithm"],
            n_iter=model_params["n_iter"],
            num_of_nonzero=model_params["num_of_nonzero"],
            test_neg_loader=test_neg_loader,
            test_pos_loader=test_pos_loader,
        )
        model.load_dict(paths["dict_file"])
        model.test(
            org_H=int(256 / 8.0) - model_params["cutoff_edge_width"] * 2,
            org_W=int(256 / 8.0) - model_params["cutoff_edge_width"] * 2,
            patch_size=model_params["patch_size"],
            stride=model_params["stride"],
        )


if __name__ == "__main__":
    main()
