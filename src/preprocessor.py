import cv2
import numpy
import torch
import torch.nn.functional as F
from torchvision import models


class BatchSplitImg(object):
    def __init__(self, patch_size, stride):
        self.patch_size = patch_size
        self.stride = stride

    def __call__(self, batch_img):
        N, C, H, W = batch_img.shape
        batch = []
        for img in batch_img:
            split_images = []
            for ty in range(0, H - self.patch_size + 1, self.stride):
                for tx in range(0, W - self.patch_size + 1, self.stride):
                    split_images.append(
                        img[:, ty: ty + self.patch_size,
                            tx: tx + self.patch_size]
                    )
            batch.append(numpy.stack(split_images))
        return numpy.stack(batch)


class HWC2CHW(object):
    def __call__(self, img):
        shape = img.shape
        if len(shape) == 3:  # HWC
            return img.transpose(2, 0, 1)
        else:
            raise ValueError("The shape of 'img' must be 3D.")


class ToTensor(object):
    def __call__(self, img):
        return torch.from_numpy(img).float()


class Gray2RGB(object):
    def __call__(self, img):
        if len(img.shape) == 2:
            img = img[:, :, None]

        # If input is RGB, do nothing
        if img.shape[2] == 3:
            return img

        return numpy.tile(img, (1, 1, 3))


class Resize(object):
    def __init__(self, size):
        if len(size) != 2:
            raise ValueError("The argument 'size' must be a list or tuple.")
        self.size = size

    def __call__(self, img):
        return cv2.resize(img, dsize=self.size)


class TransformForTorchModel(object):
    def __call__(self, img):
        if len(img.shape) != 3:
            raise ValueError("The shape of 'img' must be 3D.")
        elif img.shape[0] != 3:
            raise ValueError("'img' must be RGB image.")

        img -= numpy.array([0.485, 0.456, 0.406])[:, None, None]
        img /= numpy.array([0.229, 0.224, 0.225])[:, None, None]

        return img


class DivideBy255(object):
    def __call__(self, img):
        img = img / 255.0
        return img


class VGG16ScaledFeatures(object):
    def __init__(self, last_layer=22, cutoff_edge_width=0):
        self.vgg16_features = torch.nn.ModuleList(
            list(models.vgg16(pretrained=True).features)[:last_layer]
        ).eval()
        self.cutoff_edge_width = cutoff_edge_width

    def __call__(self, org):
        x_ = torch.tensor([])
        with torch.no_grad():
            for s in range(3):
                x = F.max_pool2d(org, (2 ** s, 2 ** s))
                for i, f in enumerate(self.vgg16_features):
                    x = f(x)
                    if (
                        (s == 0 and i == 21)
                        or (s == 1 and i == 14)
                        or (s == 2 and i == 7)
                    ):
                        print(x.shape)
                        x_ = torch.cat([x_, x], dim=1)
                        break

        if self.cutoff_edge_width > 0:
            x_ = x_[
                :,
                :,
                self.cutoff_edge_width: -self.cutoff_edge_width,
                self.cutoff_edge_width: -self.cutoff_edge_width,
            ]
        x_ = (x_ - x_.mean(dim=(2, 3), keepdim=True)) / \
            x_.std(dim=(2, 3), keepdim=True)

        return x_


class ResNet50ScaledFeatures(object):
    def __init__(self, last_layer=50, cutoff_edge_width=0):
        self.resnet50 = models.resnet50(pretrained=True)
        self.resnet50.eval()
        self.cutoff_edge_width = cutoff_edge_width

    def __call__(self, org):
        x_ = torch.tensor([])
        with torch.no_grad():
            x1 = F.max_pool2d(org, (8, 8))
            x1 = self.resnet50.conv1(x1)
            x1 = self.resnet50.bn1(x1)
            x1 = self.resnet50.relu(x1)
            x1 = self.resnet50.maxpool(x1)
            x1, _ = self.forward(self.resnet50.layer1[0], x1)
            x1, _ = self.forward(self.resnet50.layer1[1], x1)
            _, out1 = self.forward(self.resnet50.layer1[2], x1)
            x_ = torch.cat([x_, out1], dim=1)

            x2 = F.max_pool2d(org, (4, 4))
            x2 = self.resnet50.conv1(x2)
            x2 = self.resnet50.bn1(x2)
            x2 = self.resnet50.relu(x2)
            x2 = self.resnet50.maxpool(x2)
            x2, _ = self.forward(self.resnet50.layer1[0], x2)
            x2, _ = self.forward(self.resnet50.layer1[1], x2)
            x2, _ = self.forward(self.resnet50.layer1[2], x2)
            x2, _ = self.forward(self.resnet50.layer2[0], x2)
            x2, _ = self.forward(self.resnet50.layer2[1], x2)
            x2, _ = self.forward(self.resnet50.layer2[2], x2)
            _, out2 = self.forward(self.resnet50.layer2[3], x2)
            x_ = torch.cat([x_, out2], dim=1)

            x3 = F.max_pool2d(org, (2, 2))
            x3 = self.resnet50.conv1(x3)
            x3 = self.resnet50.bn1(x3)
            x3 = self.resnet50.relu(x3)
            x3 = self.resnet50.maxpool(x3)
            x3, _ = self.forward(self.resnet50.layer1[0], x3)
            x3, _ = self.forward(self.resnet50.layer1[1], x3)
            x3, _ = self.forward(self.resnet50.layer1[2], x3)
            x3, _ = self.forward(self.resnet50.layer2[0], x3)
            x3, _ = self.forward(self.resnet50.layer2[1], x3)
            x3, _ = self.forward(self.resnet50.layer2[2], x3)
            x3, _ = self.forward(self.resnet50.layer2[3], x3)
            x3, _ = self.forward(self.resnet50.layer3[0], x3)
            x3, _ = self.forward(self.resnet50.layer3[1], x3)
            x3, _ = self.forward(self.resnet50.layer3[2], x3)
            x3, _ = self.forward(self.resnet50.layer3[3], x3)
            x3, _ = self.forward(self.resnet50.layer3[4], x3)
            _, out3 = self.forward(self.resnet50.layer3[5], x3)
            x_ = torch.cat([x_, out3], dim=1)

            x4 = F.max_pool2d(org, (1, 1))
            x4 = self.resnet50.conv1(x4)
            x4 = self.resnet50.bn1(x4)
            x4 = self.resnet50.relu(x4)
            x4 = self.resnet50.maxpool(x4)
            x4, _ = self.forward(self.resnet50.layer1[0], x4)
            x4, _ = self.forward(self.resnet50.layer1[1], x4)
            x4, _ = self.forward(self.resnet50.layer1[2], x4)
            x4, _ = self.forward(self.resnet50.layer2[0], x4)
            x4, _ = self.forward(self.resnet50.layer2[1], x4)
            x4, _ = self.forward(self.resnet50.layer2[2], x4)
            x4, _ = self.forward(self.resnet50.layer2[3], x4)
            x4, _ = self.forward(self.resnet50.layer3[0], x4)
            x4, _ = self.forward(self.resnet50.layer3[1], x4)
            x4, _ = self.forward(self.resnet50.layer3[2], x4)
            x4, _ = self.forward(self.resnet50.layer3[3], x4)
            x4, _ = self.forward(self.resnet50.layer3[4], x4)
            x4, _ = self.forward(self.resnet50.layer3[5], x4)
            x4, _ = self.forward(self.resnet50.layer4[0], x4)
            x4, _ = self.forward(self.resnet50.layer4[1], x4)
            _, out4 = self.forward(self.resnet50.layer4[2], x4)
            x_ = torch.cat([x_, out4], dim=1)

        if self.cutoff_edge_width > 0:
            x_ = x_[
                :,
                :,
                self.cutoff_edge_width: -self.cutoff_edge_width,
                self.cutoff_edge_width: -self.cutoff_edge_width,
            ]
        x_ = (x_ - x_.mean(dim=(2, 3), keepdim=True)) / \
            x_.std(dim=(2, 3), keepdim=True)

        return x_

    def forward(self, f, x):
        identity = x

        out1 = f.conv1(x)
        out2 = f.bn1(out1)
        out3 = f.relu(out2)

        out4 = f.conv2(out3)
        out5 = f.bn2(out4)
        out6 = f.relu(out5)

        out7 = f.conv3(out6)
        out8 = f.bn3(out7)

        if f.downsample is not None:
            identity = f.downsample(x)

        out9 = out8 + identity
        out10 = f.relu(out9)

        return out10, out4
