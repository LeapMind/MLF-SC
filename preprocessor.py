import numpy
import torch
from PIL import Image

class SplitImg(object):
    def __init__(self, patch_size, stride, data_format='HWC'):
        if not (data_format == 'HWC' or data_format == 'CHW'):
            raise ValueError("The argument 'data_format' must be set to 'HWC' or 'CHW'.")

        self.patch_size=patch_size
        self.stride=stride
        self.data_format = data_format

    def __call__(self, img):
        shape = img.shape
        if self.data_format == 'HWC':
            H, W, C = img.shape
        else:
            C, H, W = img.shape

        split_images = []
        for ty in range(0, H-self.patch_size+1, self.stride):
            for tx in range(0, W-self.patch_size+1, self.stride):
                if self.data_format=='HWC':
                    split_images.append(img[ty:ty+self.patch_size, tx:tx+self.patch_size, :])
                else:
                    split_images.append(img[:, ty:ty+self.patch_size, tx:tx+self.patch_size])
        return numpy.stack(split_images)


class BatchSplitImg(object):
    def __init__(self, patch_size, stride, data_format='HWC'):
        if not (data_format == 'HWC' or data_format == 'CHW'):
            raise ValueError("The argument 'data_format' must be set to 'HWC' or 'CHW'.")

        self.patch_size=patch_size
        self.stride=stride
        self.data_format = data_format

    def __call__(self, batch_img):
        shape = batch_img.shape
        if self.data_format == 'HWC':
            N, H, W, C = batch_img.shape
        else:
            N, C, H, W = batch_img.shape

        batch = []
        for img in batch_img:
            split_images = []
            for ty in range(0, H-self.patch_size+1, self.stride):
                for tx in range(0, W-self.patch_size+1, self.stride):
                    if self.data_format=='HWC':
                        split_images.append(img[ty:ty+self.patch_size, tx:tx+self.patch_size, :])
                    else:
                        split_images.append(img[:, ty:ty+self.patch_size, tx:tx+self.patch_size])
            batch.append(numpy.stack(split_images))
        return numpy.stack(batch)


class HWC2CHW(object):
    def __call__(self, img):
        shape = img.shape
        if len(shape) == 3:  #HWC
            return img.transpose(2,0,1)
        else:
            raise ValueError("The shape of 'img' must be 3D.")


class ToTensor(object):
    def __call__(self, img):
        return torch.from_numpy(img).float()


class Gray2RGB(object):
    def __call__(self, img):
        if len(img.shape)==2:
            img = img[:,:,None]
        return numpy.tile(img, (1,1,3))


class RGB2Gray(object):
    def __init__(self, num_output_channels=1):
        self.num_output_channels = num_output_channels

    def __call__(self, img):
        converted = numpy.asarray(Image.fromarray(img).convert('L'))[:,:,None]
        return numpy.tile(converted, (1,1,self.num_output_channels))


class Resize(object):
    def __init__(self, size):
        if len(size) != 2:
            raise ValueError("The argument 'size' must be a list or tuple.")
        self.size = size

    def __call__(self, img):
        return numpy.array(Image.fromarray(img).resize(self.size, Image.BILINEAR))


class Normalize(object):
    def __call__(self, img):
        img = img.astype(numpy.float32)
        mean = img.mean()
        stddev = img.std()
        adjusted_stddev = max(stddev, 1.0/numpy.sqrt(img.size))

        img -= mean
        img /= adjusted_stddev

        return img        


class TransformForTorchModel(object):
    def __call__(self, img):
        if len(img.shape) != 3:
            raise ValueError("The shape of 'img' must be 3D.")
        elif img.shape[0] != 3:
            raise ValueError("'img' must be RGB image.")

        img-=numpy.array([0.485, 0.456, 0.406])[:,None,None]
        img/=numpy.array([0.229, 0.224, 0.225])[:,None,None]

        return img


class DivideBy255(object):
    def __call__(self, img):
        img = img/255.0
        return img


#----using pytorch library----
import torch
import torch.nn.functional as F
from torchvision import models

class VGG16Features(object):
    def __init__(self, last_layer=22, cutoff_edge_width=0):
        self.vgg16_features = torch.nn.ModuleList(list(models.vgg16(pretrained=True).features)[:last_layer]).eval()
        self.cutoff_edge_width = cutoff_edge_width

    def __call__(self, x):
        with torch.no_grad():
            for f in self.vgg16_features:
                x = f(x)

        if self.cutoff_edge_width > 0:
            x = x[:, :, self.cutoff_edge_width:-self.cutoff_edge_width, self.cutoff_edge_width:-self.cutoff_edge_width]

        return x.detach().numpy()

class VGG16ScaledFeatures(object):
    def __init__(self, last_layer=22, cutoff_edge_width=0):
        self.vgg16_features = torch.nn.ModuleList(list(models.vgg16(pretrained=True).features)[:last_layer]).eval()
        self.cutoff_edge_width=cutoff_edge_width

    def __call__(self, org):
        x_ = torch.tensor([])
        with torch.no_grad():
            for s in range(3):
                x = F.max_pool2d(org, (2**s,2**s))
                #x = F.interpolate(org, scale_factor=1/(2**s), mode='nearest')
                for i, f in enumerate(self.vgg16_features):
                    x = f(x)
                    if (s == 0 and i == 21) or (s == 1 and i == 14) or (s == 2 and i == 7):
                        #x = F.max_pool2d(x, (2**s,2**s))
                        x_ = torch.cat([x_, x], dim=1)
                        break

        if self.cutoff_edge_width > 0:
            x_ = x_[:, :, self.cutoff_edge_width:-self.cutoff_edge_width, self.cutoff_edge_width:-self.cutoff_edge_width]
        x_ = (x_-x_.mean(dim=(2,3), keepdim=True))/x_.std(dim=(2,3), keepdim=True)

        return x_

"""class VGG16ScaledFeatures(object):
    def __init__(self, last_layer=22, cutoff_edge_width=0):
        self.vgg16_features = torch.nn.ModuleList(list(models.vgg16(pretrained=True).features)[:last_layer]).eval()
        self.cutoff_edge_width=cutoff_edge_width

    def __call__(self, x):
        x_ = torch.tensor([])
        with torch.no_grad():
            for i, f in enumerate(self.vgg16_features):
                x = f(x)
                if i == 7:
                    x_ = torch.cat([x_, F.interpolate(x, scale_factor=1/4, mode='bilinear')], dim=1)
                    #x_ = torch.cat([x_, F.max_pool2d(x, (4,4))], dim=1)
                elif i == 14:
                    x_ = torch.cat([x_, F.interpolate(x, scale_factor=1/2, mode='bilinear')], dim=1)
                    #x_ = torch.cat([x_, F.max_pool2d(x, (2,2))], dim=1)
                elif i == 21:
                    x_ = torch.cat([x_, x], dim=1)

        if self.cutoff_edge_width > 0:
            x_ = x_[:, :, self.cutoff_edge_width:-self.cutoff_edge_width, self.cutoff_edge_width:-self.cutoff_edge_width]

        return x_"""
