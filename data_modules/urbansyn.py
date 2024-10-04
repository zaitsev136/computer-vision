import os
import numpy as np
from PIL import Image
from tqdm.auto import tqdm
from huggingface_hub import snapshot_download
import torch
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision import transforms as T
import argparse
import sys

# local imports
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, '..'))
if root_dir not in sys.path:
    sys.path.append(root_dir)

from utils import albumentation_transform


NUM_CLASSES = 20  # number of segmentation classes 
NUM_TRAIN = 6016  # number of training images
NUM_VAL = 7539 - 6016  # number of validation images

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


class UrbanSynDataset(Dataset):
    def __init__(self, path, transforms, split='train', resized=True,
                 downscaling=4):
        """torch Dataset for UrbanSyn

        Args:
            path (str): path to the raw UrbanSyn data
            transforms (obj): albumentation transforms
            split (str, optional): train, val or all. Defaults to 'train'.
            resized (bool, optional): whether to load the resized dataset
                or the original one. Defaults to True.
            downscaling (int, optional): if resized=True, the factor by
                which the dataset is downscaled. Defaults to 4.
        """
        self.path = path
        self.transforms = transforms
        self.split = split
        self.resized = resized
        if self.resized:
            size_str = str(1024//downscaling)
            if not os.path.exists(os.path.join(self.path, size_str)):
                self.path = os.path.join(self.path+'_resized', size_str)

    def __getitem__(self, index):
        if self.split=='val' or self.split=='predict':
            i = index + NUM_TRAIN + 1
        else:
            i = index + 1
        if self.resized:
            # loading resized data in npy format
            x = np.load(os.path.join(self.path, 'rgb', f'rgb_{i:04}.npy'))
            y = np.load(os.path.join(self.path, 'ss', f'ss_{i:04}.npy'))
        else:
            # loading raw data and converting it to np.ndarray
            rgb_path = os.path.join(self.path, 'rgb', f'rgb_{i:04}.png')
            x = np.array(Image.open(rgb_path).convert('RGB'))
            ss_path = os.path.join(self.path, 'ss', f'ss_{i:04}.png')
            y = np.array(Image.open(ss_path).convert('L'))

        x,y = albumentation_transform(self.transforms, x, y)
        # if using the raw dataset, shifting the class indices by 1,
        # so the background has index 0
        if self.split=='predict':
            return x
        if not self.resized:
            y = _shift_class_indices(y)
        return x, y
    
    def __len__(self):
        if self.split=='train':
            return NUM_TRAIN
        elif self.split=='val' or self.split=='predict':
            return NUM_VAL
        else:
            return NUM_TRAIN + NUM_VAL


class UrbanSynDataModule(LightningDataModule):
    def __init__(self, data_dir='./data/urbansyn', batch_size=16, downscaling=2,
                 train_transforms='default', val_transforms='default'):
        """LightningDataModule for downscaled UrbanSyn dataset.

        Args:
            data_dir (str, optional): path to the data. The _resized suffix
                is appended when necessary. Defaults to './data/urbansyn'.
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
            self.train_dataset = UrbanSynDataset(self.data_dir,
                                        self.train_transforms,
                                        'train',
                                        downscaling=self.downscaling)
        if 'fit' in stage or 'val' in stage:
            self.val_dataset = UrbanSynDataset(self.data_dir,
                                               self.val_transforms,
                                               'val',
                                               downscaling=self.downscaling)
        if 'predict' in stage:
            self.predict_dataset = UrbanSynDataset(self.data_dir,
                                                   self.val_transforms,
                                                   'predict',
                                                   downscaling=self.downscaling)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)
    
    def predict_dataloader(self):
        return DataLoader(self.predict_dataset, batch_size=self.batch_size, shuffle=False)


class UrbanSynDownloader:
    def __init__(self, data_dir='data'):
        self.data_dir = data_dir
        self.path = os.path.join(data_dir, 'urbansyn')
        self.path_resized = os.path.join(data_dir, 'urbansyn_resized')

    def download(self, data_types=('rgb', 'ss')):
        """Downloads the UrbanSyn dataset (https://www.urbansyn.org/) from
        huggingface_hub. The entire dataset is >70 GB, so it takes a while
        to download it. Without the depth maps it is "only" 21 GB.

        Args:
            data_types (tuple, optional): types of data to download.
                'rgb' for images (21 GB), 'ss' for semantic segmentation maps
                (385 MB), 'depth' for depth maps (59 GB!). Defaults to
                ('rgb', 'ss').
        """
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

        if os.path.exists(self.path):
            print(f'{self.path} already exists.')
        else:
            for dt in data_types:
                print('downloading ' + dt + '...')
                snapshot_download(repo_id="UrbanSyn/UrbanSyn", repo_type="dataset",
                                  local_dir=self.path, allow_patterns=f'{dt}/*')
                print('done')

    def downscale(self, factor):
        """Downscale the entire UrbanSyn dataset by the provided factor

        Args:
            factor (int): downscaling factor. Good choices are multiples of 2
        """
        raw_size = np.array((1024, 2048))
        new_size = raw_size//int(factor)
        dataset = UrbanSynDataset(self.path, transforms=A.Resize(*new_size),
                                  split='all', resized=False)

        os.makedirs(self.path_resized, exist_ok=True)

        full_output_path = os.path.join(self.path_resized, str(new_size[0]))
        if os.path.exists(full_output_path):
            print(full_output_path, 'already exists.')
            return
        
        os.makedirs(full_output_path)
        rgb_path = os.path.join(full_output_path, 'rgb')
        os.makedirs(rgb_path)
        ss_path = os.path.join(full_output_path, 'ss')
        os.makedirs(ss_path)

        for i, (x, y) in tqdm(enumerate(dataset), total=len(dataset)):
            np.save(os.path.join(rgb_path, f'rgb_{i+1:04}.npy'), x)
            np.save(os.path.join(ss_path, f'ss_{i+1:04}.npy'), y)
            if i == len(dataset)-1:
                break

    def get_datamodule(self, batch_size=16, downscaling=2,
                       train_transforms='default', val_transforms='default'):
        """Returns an UrbanSynDataModule

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
            UrbanSynDataModule: UrbanSyn LightningDataModule
        """
        return UrbanSynDataModule(self.path, batch_size, downscaling,
                                  train_transforms, val_transforms)


def _shift_class_indices(segmap):
    y = segmap + 1
    y[y==NUM_CLASSES] = 0
    return y


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

    parser = argparse.ArgumentParser(description='CLI for downloading and downsampling the UrbanSyn dataset.')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    download_parser = subparsers.add_parser('download', help='Download the dataset')
    download_parser.add_argument('--data_dir', type=str, default=os.path.join(root_dir,  'data'),
                                 help='Output path. (Default: %(default)s)')
    download_parser.add_argument('--data_types', nargs='+', default=['rgb', 'ss'],
                                 help='Which data to download: "rgb" for images, "ss" for segmentation masks. (Default: rgb ss)')

    downscale_parser = subparsers.add_parser('downscale', help='Downscales the images and maps in the dataset by the provided factor.')
    downscale_parser.add_argument('--data_dir', type=str, default=os.path.join(root_dir, 'data'),
                                 help='Path to the directory containing the urbansyn folder. (Default: %(default)s)')
    downscale_parser.add_argument('--factors', nargs='+', default=[2, 4],
                                 help='Downscaling factors. (Default: 2 4)')

    # Parse the arguments
    args = parser.parse_args()

    # Handle each command separately
    if args.command == 'download':
        us = UrbanSynDownloader(args.data_dir)
        us.download(args.data_types)
    elif args.command == 'downscale':
        us = UrbanSynDownloader(args.data_dir)
        for factor in args.factors:
            us.downscale(factor)
    else:
        parser.print_help()
    

if __name__ == '__main__':
    main()
