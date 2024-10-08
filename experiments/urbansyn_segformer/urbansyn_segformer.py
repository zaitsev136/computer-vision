import numpy as np
from PIL import Image
import os
import sys

import torch
import torch.nn.functional as F

import albumentations as A
from albumentations.pytorch import ToTensorV2

from transformers import SegformerForSemanticSegmentation

from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.cli import LightningCLI

# local imports
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, '..', '..'))
if root_dir not in sys.path:
    sys.path.append(root_dir)

from data_modules import urbansyn
import utils


def _add_background(logits):
    """Adds -inf logits for the background class

    Args:
        logits (torch.Tensor): logits for semantic segmentation prediction
            for all classes except the background. The shape is (B, C-1, H, W)

    Returns:
        torch.Tensor: logits with the added background class, (B, C, H, W)
    """
    size = logits.size()
    infs = (torch.zeros(size[0], 1, *size[-2:]) - torch.inf).to(logits)
    return torch.concat([infs, logits], dim=1)


class UrbanSynSegFormer(LightningModule):
    def __init__(self, learning_rate=1e-3, lr_gamma=0.7):
        """Lightning module for training and using a SegFormer on UrbanSyn
        dataset. Uses a B0 SegFormer from NVIDIA huggingface, pretrained on
        CityScapes.

        Args:
            learning_rate (float, optional): Learning rate. Defaults to 1e-3.
            lr_gamma (float, optional): Learning rate decay. Defaults to 0.7.
        """
        super().__init__()
        self.model = SegformerForSemanticSegmentation.from_pretrained('nvidia/segformer-b0-finetuned-cityscapes-1024-1024')
        self.lr = learning_rate
        self.lr_gamma = lr_gamma
        self.save_hyperparameters()
        self.loss_fn = utils.DiceLoss()
        self.validation_batch = None

    def setup(self, stage):
        super().setup(stage)
        self.val_intersections = torch.zeros(urbansyn.NUM_CLASSES-1, device=self.device)
        self.val_unions = torch.zeros(urbansyn.NUM_CLASSES-1, device=self.device)

    def forward(self, x):
        output = _add_background(self.model(x).logits)
        return F.interpolate(output, scale_factor=4)

    def training_step(self, batch, batch_idx):
        x, y = batch
        predictions = self.forward(x)
        loss = self.loss_fn(predictions, y)
        self.log('train_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        predictions = self.forward(x)
        loss = self.loss_fn(predictions, y)
        self.log('val_loss', loss)

        # for the IoU metrics
        intersections, unions = utils.get_intersections_and_unions(predictions, y)
        self.val_intersections += intersections
        self.val_unions += unions

        # classification accuracy
        accuracy = utils.get_accuracy(predictions, y)
        self.log('accuracy', accuracy, on_step=False, on_epoch=True)

        # saving the first validation batch for logging predicted segmaps
        if self.validation_batch is None:
            self.validation_batch = x
            self.decoded_targets = urbansyn.colorize_segmap(y).to(self.validation_batch)
            self.validation_images = torch.concat(list(urbansyn.INV_NORMALIZE(self.validation_batch)), dim=1)
            self.validation_truth = torch.concat(list(self.decoded_targets), dim=1)

    def on_validation_epoch_end(self):
        # predicted segmaps for the first validation batch
        prediction = self.forward(self.validation_batch)
        decoded_predictions = urbansyn.colorize_segmap(torch.argmax(prediction, 1)).to(self.validation_batch)
        validation_predictions = torch.concat(list(decoded_predictions), dim=1)
        image = torch.concat([self.validation_images, self.validation_truth, validation_predictions], dim=2)
        self.logger.experiment.add_image('image', image, self.current_epoch)
        # IoU
        self.val_unions[self.val_unions==0] = torch.nan
        ious = self.val_intersections / self.val_unions
        miou = ious.nanmean()
        self.log('_MeanIoU_', miou, on_epoch=True, on_step=False)
        for i, v in enumerate(ious):
            self.log(f'{urbansyn.CLASS_NAMES[i+1]}_IoU', v, on_epoch=True, on_step=False)
        self.val_intersections = 0
        self.val_unions = 0

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, self.lr_gamma)
        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}
    

class CustomLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        # Add custom arguments to the parser for prediction
        parser.add_argument('--input_path', type=str, help='Path to the input image for prediction')
        parser.add_argument('--output_path', type=str, help='Filename for the output. By default, appends _segmap to the input_path', default=None)
        parser.set_defaults({'data.data_dir': os.path.join(root_dir, 'data', 'urbansyn'),
                             'trainer.max_epochs': 500,
                             'trainer.precision': 'bf16-mixed'})

    def instantiate_classes(self):
        # for validate and predict, set best.ckpt as the default checkpoint
        if self.config[self.config['subcommand']]['ckpt_path'] is None and 'fit' not in self.config['subcommand']:
            self.config[self.config['subcommand']]['ckpt_path'] = os.path.join(current_dir, 'best.ckpt')

        super().instantiate_classes()
        
        # Check if in prediction mode
        if self.config['subcommand']=='predict':
            input_path = self.config['predict']['input_path'] if 'input_path' in self.config['predict'] else None
            if input_path:
                self.run_prediction(input_path, self.config['predict']['output_path'])
            else:
                raise ValueError('Input data path must be provided for prediction mode. Use the --input_path argument.')
    
    def run_prediction(self, input_path, output_path):
        input = np.array(Image.open(input_path).convert('RGB'))
        transform = A.Compose([A.Normalize(), ToTensorV2()])
        x = transform(image=input)['image'].unsqueeze(0).unsqueeze(0)

        # ignoring the CLI trainer, running a new one
        model = UrbanSynSegFormer.load_from_checkpoint(self.config[self.config['subcommand']]['ckpt_path'])
        output = Trainer().predict(model, x)[0].detach().cpu()
        output = torch.argmax(output, 1)[0]
        segmap = urbansyn.colorize_segmap(output, for_pil=True)
        
        if output_path is None:
            directory, filename = os.path.split(input_path)
            name, ext = os.path.splitext(filename)
            output_filename = f"{name}_segmap{ext}"
            output_path = os.path.join(directory, output_filename)

        segmap = Image.fromarray(segmap, mode='RGB')
        segmap.save(output_path)
        print('The output is saved to', output_path)
        exit()

    @staticmethod
    def subcommands():
        """Defines the list of available subcommands and the arguments to skip."""
        return {
            "fit": {"model", "train_dataloaders", "val_dataloaders", "datamodule"},
            "validate": {"model", "dataloaders", "datamodule"},
            "predict": {"model", "dataloaders", "datamodule"},
        }


def cli_main():
    cli = CustomLightningCLI(UrbanSynSegFormer, urbansyn.UrbanSynDataModule)


if __name__ == '__main__':
    cli_main()
