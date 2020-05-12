import numpy
from tqdm import tqdm
from sklearn.decomposition import MiniBatchDictionaryLearning, SparseCoder
from skimage.metrics import structural_similarity
from sklearn.metrics import average_precision_score, roc_auc_score
import pickle
import os
import matplotlib.pyplot as plt
import cv2
import random


class SparseCodingWithMultiDict(object):
    def __init__(
        self,
        preprocesses,
        model_env,
        train_loader=None,
        test_neg_loader=None,
        test_pos_loader=None,
    ):

        self.preprocesses = preprocesses

        self.num_of_basis = model_env["num_of_basis"]
        self.alpha = model_env["alpha"]
        self.transform_algorithm = model_env["transform_algorithm"]
        self.transform_alpha = model_env["transform_alpha"]
        self.fit_algorithm = model_env["fit_algorithm"]
        self.n_iter = model_env["n_iter"]
        self.num_of_nonzero = model_env["num_of_nonzero"]
        self.use_ssim = model_env["use_ssim"]

        self.cutoff_edge_width = model_env["cutoff_edge_width"]
        self.patch_size = model_env["patch_size"]
        self.stride = model_env["stride"]
        self.num_of_ch = model_env["num_of_ch"]
        self.output_npy = False

        self.org_l = int(256 / 8.0) - self.cutoff_edge_width * 2

        self.train_loader = train_loader
        self.test_neg_loader = test_neg_loader
        self.test_pos_loader = test_pos_loader

        self.dictionaries = None

    def train(self):
        arrs = []
        cnt = 0
        for batch_data in self.train_loader:
            print(cnt)
            cnt += 1
            batch_img = batch_data[2]
            for p in self.preprocesses:
                batch_img = p(batch_img)
            N, P, C, H, W = batch_img.shape
            batch_arr = batch_img.reshape(N * P, C, H * W)
            arrs.append(batch_arr)

        train_arr = numpy.concatenate(arrs, axis=0)

        self.dictionaries = [
            MiniBatchDictionaryLearning(
                n_components=self.num_of_basis,
                alpha=self.alpha,
                transform_algorithm=self.transform_algorithm,
                transform_alpha=self.transform_alpha,
                fit_algorithm=self.fit_algorithm,
                n_iter=self.n_iter,
            )
            .fit(train_arr[:, i, :])
            .components_
            for i in tqdm(range(C), desc="learning dictionary")
        ]
        print("learned.")

    def save_dict(self, file_path):
        with open(file_path, "wb") as f:
            pickle.dump(self.dictionaries, f)

    def load_dict(self, file_path):
        with open(file_path, "rb") as f:
            self.dictionaries = pickle.load(f)

    def test(self):
        C = len(self.dictionaries)
        coders = [
            SparseCoder(
                dictionary=self.dictionaries[i],
                transform_algorithm=self.transform_algorithm,
                transform_n_nonzero_coefs=self.num_of_nonzero,
            )
            for i in range(C)
        ]

        neg_err = self.calculate_error(coders=coders, is_positive=False)
        pos_err = self.calculate_error(coders=coders, is_positive=True)

        ap, auc = self.calculate_score(neg_err, pos_err)
        print("\nTest set: AP: {:.4f}, AUC: {:.4f}\n".format(ap, auc))

    def calculate_error(self, coders, is_positive):
        if is_positive:
            loader = self.test_pos_loader
        else:
            loader = self.test_neg_loader

        errs = []
        top_5 = numpy.zeros(len(self.dictionaries))

        random.seed(0)
        dict_order = list(range(960))
        random.shuffle(dict_order)

        for batch_data in tqdm(loader, desc="testing"):

            batch_path, batch_name, batch_img = batch_data
            p_batch_img = batch_img
            for p in self.preprocesses:
                p_batch_img = p(p_batch_img)

            for p_img, org_img in zip(p_batch_img, batch_img):

                P, C, H, W = p_img.shape
                img_arr = p_img.reshape(P, C, H * W)
                f_diff = numpy.zeros((1, self.org_l, self.org_l))

                ch_err = []
                for num in range(self.num_of_ch):
                    i = dict_order[num]
                    target_arr = img_arr[:, i]
                    coefs = coders[i].transform(target_arr)
                    rcn_arr = coefs.dot(self.dictionaries[i])

                    f_img_org = self.reconst_from_array(target_arr)
                    f_img_rcn = self.reconst_from_array(rcn_arr)
                    f_diff += numpy.square((f_img_org - f_img_rcn) / 1.5)

                    if not self.use_ssim:
                        err = numpy.sum((target_arr - rcn_arr) ** 2, axis=1)
                    else:
                        err = self.calc_ssim(img_arr, rcn_arr, (P, C, H, W))
                    sorted_err = numpy.sort(err)[::-1]
                    total_err = numpy.sum(sorted_err[:5])
                    ch_err.append(total_err)

                top_5[numpy.argsort(ch_err)[::-1][:5]] += 1
                errs.append(numpy.sum(ch_err))
                f_diff /= self.num_of_ch
                if self.output_npy:
                    self.output_np_array(batch_path, batch_name, f_diff)
                else:
                    visualized_out = self.visualize(org_img, f_diff)
                    self.output_image(batch_path, batch_name,
                                      ch_err, visualized_out)
        return errs

    def output_np_array(self, batch_path, batch_name, f_diff):
        output_path = os.path.join("visualized_results", batch_path)
        os.makedirs(output_path, exist_ok=True)
        numpy.save(os.path.join(
            output_path, batch_name.split(".")[0] + ".npy"), f_diff)

    def output_image(self, batch_path, batch_name, ch_err, visualized_out):
        output_path = os.path.join("visualized_results", batch_path)
        os.makedirs(output_path, exist_ok=True)

        cv2.imwrite(
            os.path.join(
                output_path,
                batch_name.split(".")[0] + "-" +
                str(int(numpy.sum(ch_err))) + ".png",
            ),
            visualized_out,
        )

    def calculate_ssim(self, img_arr, rcn_arr, dim):
        P, C, H, W = dim
        return [
            -1
            * structural_similarity(
                img_arr[p, c].reshape(H, W),
                rcn_arr[p, c].reshape(H, W),
                win_size=11,
                data_range=1.0,
                gaussian_weights=True,
            )
            for p in range(P)
            for c in range(C)
        ]

    def visualize(self, org_img, f_diff):
        color_map = plt.get_cmap("viridis")
        heatmap = numpy.uint8(color_map(f_diff[0])[:, :, :3] * 255)
        transposed = org_img.transpose(1, 2, 0)[:, :, [2, 1, 0]]
        resized = cv2.resize(
            heatmap, (transposed.shape[0], transposed.shape[1])
        )
        blended = cv2.addWeighted(
            transposed, 1.0, resized, 0.01, 2.2, dtype=cv2.CV_32F
        )
        blended_normed = (
            255 * (blended - blended.min()) /
            (blended.max() - blended.min())
        )
        blended_out = numpy.array(blended_normed, numpy.int)
        return blended_out

    def calculate_score(self, dn, dp):
        N = len(dn)
        y_score = numpy.concatenate([dn, dp])
        y_true = numpy.zeros(len(y_score), dtype=numpy.int32)
        y_true[N:] = 1
        return average_precision_score(y_true, y_score),\
            roc_auc_score(y_true, y_score)

    def reconst_from_array(self, arrs):
        rcn = numpy.zeros((1, self.org_l, self.org_l))
        arr_iter = iter(arrs)
        for ty in range(0, self.org_l - self.patch_size + 1, self.stride):
            for tx in range(0, self.org_l - self.patch_size + 1, self.stride):
                arr = next(arr_iter)
                rcn[:, ty: ty + self.patch_size, tx: tx + self.patch_size] =\
                    arr.reshape(
                    1, self.patch_size, self.patch_size
                )
        return rcn
