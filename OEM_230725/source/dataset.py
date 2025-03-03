import os
import numpy as np
import rasterio
from torch.utils.data import Dataset as BaseDataset
from . import transforms as T
import cv2
import numpy as np


def load_multiband(path):
    """Loads an image, handling both PNG and TIFF formats."""
    if path.endswith('.tif') or path.endswith('.tiff'):
        src = rasterio.open(path, "r")
        return (np.moveaxis(src.read(), 0, -1)).astype(np.uint8)
    elif path.endswith('.png') or path.endswith('.jpg'):
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"Failed to load image at {path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        raise ValueError(f"Unsupported file format: {path}")



def load_grayscale(path):
    """Loads a grayscale mask image, supporting both PNG and TIFF formats."""
    if path.endswith('.tif') or path.endswith('.tiff'):
        src = rasterio.open(path, "r")
        return (src.read(1)).astype(np.uint8)
    elif path.endswith('.png') or path.endswith('.jpg'):
        return cv2.imread(path, cv2.IMREAD_GRAYSCALE).astype(np.uint8)
    else:
        raise ValueError(f"Unsupported file format: {path}")


def get_crs(path):
    src = rasterio.open(path, "r")
    return src.crs, src.transform

def save_img(path,img,crs,transform):
    with rasterio.open(
        path,
        'w',
        driver='GTiff',
        height=img.shape[1],
        width=img.shape[2],
        count=img.shape[0],
        dtype=img.dtype,
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(img)
        dst.close()


class Dataset(BaseDataset):
    def __init__(self, label_list, classes=None, size=128, train=False):
        self.fns = [x.replace(".tif", ".png") if x.endswith(".tif") else x for x in label_list]  # Ensure correct file extension
        self.augm = T.train_augm3 if train else T.valid_augm
        self.size = size
        self.train = train
        self.to_tensor = T.ToTensor(classes=classes)
        self.load_multiband = load_multiband
        self.load_grayscale = load_grayscale

        # FIX: Use self.fns instead of self.img_paths
        self.msk_paths = [x.replace("images", "labels") for x in self.fns]

    def __getitem__(self, idx):
        img = self.load_multiband(self.fns[idx].replace("labels", "images"))
        msk = self.load_grayscale(self.msk_paths[idx])  # Ensure using correct mask path

        if self.train:
            data = self.augm({"image": img, "mask": msk}, self.size)
        else:
            data = self.augm({"image": img, "mask": msk}, 1024)
        data = self.to_tensor(data)

        return {"x": data["image"], "y": data["mask"], "fn": self.fns[idx]}

    def __len__(self):
        return len(self.fns)




class Dataset2(BaseDataset):
    def __init__(self, root, label_list, classes=None, size=128, train=False):
        self.fns = [os.path.join(root, "labels", x) for x in label_list]
        self.augm = T.train_augm2 if train else T.valid_augm2
        self.size = size
        self.train = train
        self.to_tensor = T.ToTensor(classes=classes)
        self.load_multiband = load_multiband
        self.load_grayscale = load_grayscale

        # FIX: Use self.fns instead of self.img_paths
        self.msk_paths = [x.replace("images", "labels").replace(".tif", ".png") for x in self.fns]

    def __getitem__(self, idx):
        img = self.load_multiband(self.fns[idx].replace("labels", "images"))
        msk = self.load_grayscale(self.msk_paths[idx])  # Use correct mask path
        osm = self.load_multiband(self.fns[idx].replace("labels", "osm"))

        if self.train:
            data = self.augm({"image": img, "mask": msk, "osm": osm}, self.size)
        else:
            data = self.augm({"image": img, "mask": msk, "osm": osm}, 1024)
        data = self.to_tensor(data)

        return {"x": data["image"], "y": data["mask"], "z": data["osm"], "fn": self.fns[idx]}

    def __len__(self):
        return len(self.fns)



class Dataset3(BaseDataset):
    def __init__(self, root, label_list, classes=None, size=128, train=False):
        self.fns = label_list
        self.augm = T.train_augm if train else T.valid_augm
        self.size = size
        self.train = train
        self.to_tensor = T.ToTensor(classes=classes)
        self.load_multiband = load_multiband
        self.load_grayscale = load_grayscale

        # FIX: Use self.fns instead of self.img_paths
        self.msk_paths = [x.replace("images", "labels").replace(".tif", ".png") for x in self.fns]

    def __getitem__(self, idx):
        img = self.load_multiband(self.fns[idx].replace("labels", "images"))
        msk = self.load_grayscale(self.msk_paths[idx])  # Use correct mask path

        if self.train:
            data = self.augm({"image": img, "mask": msk}, self.size)
        else:
            data = self.augm({"image": img, "mask": msk}, 1024)
        data = self.to_tensor(data)

        return {"x": data["image"], "y": data["mask"], "fn": self.fns[idx]}

    def __len__(self):
        return len(self.fns)
