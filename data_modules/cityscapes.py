import os
import numpy as np
from PIL import Image
from tqdm.auto import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision import transforms as T
import argparse
from torchvision.datasets import Cityscapes
import sys
import cityscapesscripts.download.downloader as cityscapes_downloader
import zipfile

# local imports
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, '..'))
if root_dir not in sys.path:
    sys.path.append(root_dir)

from utils import albumentation_transform


NUM_CLASSES = 20  # number of segmentation classes 
NUM_TRAIN = 2975  # number of training images
NUM_VAL = 500     # number of validation images
NUM_TEST = 1525   # number of test images

# inverse normalization with ImageNet coefficients
INV_NORMALIZE = T.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
                            std=[1/0.229, 1/0.224, 1/0.225])

COLORS = [
    [0, 0, 0],
    [128, 64, 128],
    [244, 35, 232],
    [70, 70, 70],
    [102, 102, 156],
    [190, 153, 153],
    [153, 153, 153],
    [250, 170, 30],
    [220, 220, 0],
    [107, 142, 35],
    [152, 251, 152],
    [0, 130, 180],
    [220, 20, 60],
    [255, 0, 0],
    [0, 0, 142],
    [0, 0, 70],
    [0, 60, 100],
    [0, 80, 100],
    [0, 0, 230],
    [119, 11, 32],
    ]

CLASS_NAMES = ['unlabelled', 'road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic_light',
               'traffic_sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car', 'truck', 'bus',
               'train', 'motorcycle', 'bicycle']

LABEL_COLORS = dict(zip(range(NUM_CLASSES), COLORS))


class CityscapesDataset(Dataset):
    def __init__(self, path, transforms, split='train', resized=True,
                 downscaling=4):
        """torch Dataset for Cityscapes

        Args:
            path (str): path to the raw Cityscapes dataset
            transforms (obj): albumentation transforms
            split (str, optional): train, val, test, or predict. Defaults to 'train'.
            resized (bool, optional): whether to load the resized dataset
                or the original one. Defaults to True.
            downscaling (int, optional): if resized=True, the factor by
                which the dataset is downscaled. Defaults to 4.
        """
        self.path = path
        self.transforms = transforms
        self.split = split
        self.return_segmap = True
        if 'predict' in split:
            self.split = 'val'
            self.return_segmap = False
        self.resized = resized
        if downscaling==1:
            self.resized = False
        if self.resized:
            size_str = str(1024//int(downscaling))
            if not os.path.exists(os.path.join(self.path, size_str)):
                self.path = os.path.join(self.path+'_resized', size_str)
        else:
            self.cityscapes = Cityscapes(path, split, target_type='semantic') #transforms=transforms

        self._IGNORE_INDEX = 255
        self._VOID_CLASSES = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
        self._VALID_CLASSES = [self._IGNORE_INDEX, 7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]
        self._class_map = dict(zip(self._VALID_CLASSES, range(len(self._VALID_CLASSES))))

    def _encode_segmap(self, mask):
        """Substitute unwanted classes in segmentation map by the 'unlabelled class'
        """
        for void_class in self._VOID_CLASSES:
            mask[mask == void_class] = self._IGNORE_INDEX
        for valid_class in self._VALID_CLASSES:
            mask[mask == valid_class] = self._class_map[valid_class]
        return mask

    def __getitem__(self, index):
        if self.resized:
            # loading resized data in npy format
            x = np.load(os.path.join(self.path, self.split, f'rgb_{index:04}.npy'))
            y = np.load(os.path.join(self.path, self.split, f'ss_{index:04}.npy'))

            if self.transforms is not None:
                x, y = albumentation_transform(self.transforms, x, y)

            if self.return_segmap:
                return x, y
            else:
                return x

        else:
            img = Image.open(self.cityscapes.images[index]).convert('RGB')
            segmap = Image.open(self.cityscapes.targets[index][0])

            if self.transforms is not None:
                transformed = self.transforms(image=np.array(img), mask=np.array(segmap))
                img = transformed['image']
                segmap = transformed['mask']
            segmap = self._encode_segmap(segmap)
            return img, segmap
    
    def __len__(self):
        if self.resized:
            if self.split=='train':
                return NUM_TRAIN
            elif self.split=='val' or self.split=='predict':
                return NUM_VAL
            elif self.split=='test':
                return NUM_TEST
            else:
                raise ValueError(f'split must be one of {"train", "val", "test"}, but {self.split} is given')
        else:
            return len(self.cityscapes)
        

class CityscapesDataModule(LightningDataModule):
    def __init__(self, data_dir='./data/cityscapes', batch_size=16, downscaling=2,
                 train_transforms='default', val_transforms='default'):
        """LightningDataModule for downscaled Cityscapes dataset.

        Args:
            data_dir (str, optional): path to the data. The _resized suffix
                is appended when necessary. Defaults to './data/cityscapes'.
            batch_size (int, optional): batch size. Defaults to 16.
            downscaling (int, optional): downscaling factor. Defaults to 2.
            train_transforms (obj, optional): albumentation transforms for the
                training data or 'default', in which case HorizontalFlip,
                ColorJitter, Normalize and ToTensorV2 are applied.
                Defaults to 'default'.
            val_transforms (obj, optional): albumentation transforms for the
                validation data or 'default', in which case Normalize and
                ToTensorV2 are applied. Defaults to 'default'.
        """
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.downscaling = downscaling
        if train_transforms=='default':
            self.train_transforms = A.Compose([
                A.HorizontalFlip(),
                A.ColorJitter(hue=0),
                A.Normalize(),
                ToTensorV2(),
                ])
        else:
            self.train_transforms = train_transforms

        if val_transforms=='default':
            self.val_transforms = A.Compose([
                A.Normalize(),
                ToTensorV2(),
                ])
        else:
            self.val_transforms = val_transforms

    def setup(self, stage: str):
        if 'fit' in stage:
            self.train_dataset = CityscapesDataset(self.data_dir, self.train_transforms,
                                                   'train', downscaling=self.downscaling)
        if 'fit' in stage or 'val' in stage:
            self.val_dataset = CityscapesDataset(self.data_dir, self.val_transforms,
                                                 'val', downscaling=self.downscaling)
        if 'test' in stage:
            self.test_dataset = CityscapesDataset(self.data_dir, self.val_transforms,
                                                  'test', downscaling=self.downscaling)
        if 'predict' in stage:
            self.predict_dataset = CityscapesDataset(self.data_dir, self.val_transforms,
                                                     'predict', downscaling=self.downscaling)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)
    
    def predict_dataloader(self):
        return DataLoader(self.predict_dataset, batch_size=self.batch_size, shuffle=False)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)
    

class CityscapesDownloader:
    def __init__(self, data_dir='data'):
        """Helper class to download and downsample Cityscapes dataset images and segmentation maps

        Args:
            data_dir (str, optional): Directory where cityscapes directory appears. Defaults to 'data'.
        """
        self.data_dir = data_dir
        self.path = os.path.join(data_dir, 'cityscapes')
        self.path_resized = os.path.join(data_dir, 'cityscapes_resized')

    def download(self, resume=False):
        """
        Download RGB images and fine segmentation maps from https://www.cityscapes-dataset.com/.
        You will be prompted for a username and a password.
        Register at https://www.cityscapes-dataset.com/ to be able to download the dataset.
        """
        session = cityscapes_downloader.login()

        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

        if os.path.exists(self.path) and not resume:
            print(f'{self.path} already exists, aborting download.')
            return
        
        os.makedirs(self.path, exist_ok=True)

        packages = ['gtFine', 'leftImg8bit'] if not resume else ['leftImg8bit']
        zips = [p+'_trainvaltest.zip' for p in packages]

        cityscapes_downloader.download_packages(session=session, package_names=zips,
                                                destination_path=self.path, resume=resume)
        print('Download complete')    
        print('Unzipping...')
        for z in zips:
            with zipfile.ZipFile(os.path.join(self.path, z), 'r') as zip_ref:
                zip_ref.extractall(path=self.path)

            for f in ['README', 'license.txt', z]:
                os.remove(os.path.join(self.path, f))
        print('Done')

    def downscale(self, factor):
        """Downscale the Cityscapes dataset by the provided factor

        Args:
            factor (int): downscaling factor. Good choices are multiples of 2
        """
        raw_size = np.array((1024, 2048))
        new_size = raw_size//int(factor)
        datasets = {}
        for key in ['train', 'val', 'test']:
            datasets[key] = CityscapesDataset(self.path, transforms=A.Resize(*new_size),
                                              split=key, resized=False)

        os.makedirs(self.path_resized, exist_ok=True)

        output_path = os.path.join(self.path_resized, str(new_size[0]))
        if os.path.exists(output_path):
            print(output_path, 'already exists.')
            return
        
        os.makedirs(output_path)

        for key, dataset in datasets.items():
            print(key+' dataset...')
            full_output_path = os.path.join(output_path, key)
            os.makedirs(full_output_path)
            for i, (x, y) in tqdm(enumerate(dataset), total=len(dataset)-1):
                np.save(os.path.join(full_output_path, f'rgb_{i:04}.npy'), x)
                np.save(os.path.join(full_output_path, f'ss_{i:04}.npy'), y)
                if i == len(dataset)-1:
                    break
                
    def get_datamodule(self, batch_size=16, downscaling=2,
                       train_transforms='default', val_transforms='default'):
        """Returns a CityscapesDataModule

        Args:
            batch_size (int, optional): batch size. Defaults to 16.
            downscaling (int, optional): downscaling factor. Defaults to 2.
            train_transforms (obj, optional): albumentation transforms for the
                training data or 'default', in which case HorizontalFlip,
                ColorJitter, Normalize and ToTensorV2 are applied.
                Defaults to 'default'.
            val_transforms (obj, optional): albumentation transforms for the
                validation data or 'default', in which case Normalize and
                ToTensorV2 are applied. Defaults to 'default'.

        Returns:
            CityscapesDataModule: a Cityscapes LightningDataModule
        """
        return CityscapesDataModule(self.path, batch_size, downscaling,
                                    train_transforms, val_transforms)


def colorize_segmap(segmap, for_pil=False):
    """Convert segmentation map with class indices into an RGB image,
    represented with a tensor or np.ndarray 

    Args:
        mask (torch.Tensor): segmentation mask
        for_pil (bool, optional): If True, returns np.ndarray of uint8
            with shape (H, W, C), ready to be saved as image with PIL.
            If False, returns a torch.Tensor (C, H, W)

    Returns:
        np.ndarray or torch.Tensor: RGB image of the segmentation map
    """
    segmap = segmap.cpu().numpy()
    r = segmap.copy()
    g = segmap.copy()
    b = segmap.copy()
    for l in range(0, NUM_CLASSES):
        r[segmap == l] = LABEL_COLORS[l][0]
        g[segmap == l] = LABEL_COLORS[l][1]
        b[segmap == l] = LABEL_COLORS[l][2]
    
    rgb = np.zeros(list(segmap.shape)+[3])
    rgb[..., 0] = r
    rgb[..., 1] = g
    rgb[..., 2] = b

    if for_pil:
        return rgb.astype(np.uint8)
    else:
        rgb = rgb / 255.
        if len(rgb.shape)==4:
            rgb = np.transpose(rgb, (0, 3, 1, 2))
        elif len(rgb.shape)==3:
            rgb = np.transpose(rgb, (2, 0, 1))
        else:
            raise ValueError(f'mask must have shape either (H,W) or (N,H,W), but {segmap.size()} is provided')
        
        return torch.from_numpy(rgb)        


def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(current_dir, os.pardir))

    parser = argparse.ArgumentParser(description='CLI for downloading and downsampling the Cityscapes dataset.')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    download_parser = subparsers.add_parser('download', help='Download the dataset')
    download_parser.add_argument('--data_dir', type=str, default=os.path.join(root_dir,  'data'),
                                 help='Output path. (Default: %(default)s)')
    download_parser.add_argument('--resume', action=argparse.BooleanOptionalAction)

    downscale_parser = subparsers.add_parser('downscale', help='Downscales the images and maps in the dataset by the provided factor.')
    downscale_parser.add_argument('--data_dir', type=str, default=os.path.join(root_dir, 'data'),
                                 help='Path to the directory containing the cityscapes folder. (Default: %(default)s)')
    downscale_parser.add_argument('--factors', nargs='+', default=[2, 4],
                                 help='Downscaling factors. (Default: 2 4)')

    # Parse the arguments
    args = parser.parse_args()

    # Handle each command separately
    if args.command == 'download':
        cs = CityscapesDownloader(args.data_dir)
        cs.download(args.resume)
    elif args.command == 'downscale':
        cs = CityscapesDownloader(args.data_dir)
        for factor in args.factors:
            print(f'Downscaling Cityscapes by a factor of {factor}...')
            cs.downscale(factor)
    else:
        parser.print_help()
    

if __name__ == '__main__':
    main()
