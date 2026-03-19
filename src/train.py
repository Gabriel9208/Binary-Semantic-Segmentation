import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

from logwb import LogWB
from .oxford_pet import (
    PetDataset, safe_load_df, AlbumentationsAdapter, 
    IMAGENET_MEAN, IMAGENET_STD
)

SEED = 42

hyper_params = {
    'lr': 0,
    'batch_size': 64,
    'test_batch_size': 1,
    'num_epochs': 0,
    'device': torch.device('cuda') if torch.cuda.is_available() else 'cpu',
    'weight_decay': 0,
    'num_workers': 0
}

train_df = safe_load_df(
    r'dataset\train.txt', 
    r'dataset\oxford-iiit-pet\images',
    r'dataset\oxford-iiit-pet\annotations\trimaps'
)
val_df = safe_load_df(
    r'dataset\val.txt', 
    r'dataset\oxford-iiit-pet\images',
    r'dataset\oxford-iiit-pet\annotations\trimaps'
)
test_unet_df = safe_load_df(
    r'dataset\test_unet.txt', 
    r'dataset\oxford-iiit-pet\images',
    r'dataset\oxford-iiit-pet\annotations\trimaps'
)
test_res_unet_df = safe_load_df(
    r'dataset\test_res_unet.txt', 
    r'dataset\oxford-iiit-pet\images',
    r'dataset\oxford-iiit-pet\annotations\trimaps'
)

train_unet_transform = AlbumentationsAdapter(
    A.Compose([
        A.LongestMaxSize(572),
        A.PadIfNeeded(min_height=572, min_width=572, border_mode=0, fill=0, fill_mask=255),
        A.RandomCrop(width=572, height=572),  
        A.OneOf([
            A.HorizontalFlip(p=1),
            A.VerticalFlip(p=1)
        ], p=0.5),
        A.GridDropout(p=0.5, ratio=0.3),
        A.OneOf([
            A.ChannelDropout((1,1), p=1),
            A.ToGray(p=1, num_output_channels=3)
        ], p=0.35),
        A.Affine(
            scale=(0.8, 1.2),
            translate_percent=(-0.2, 0.2),
            rotate=(-45, 45),
            shear=(-10, 10),
            fill=0,
            fill_mask=255
        ),
        A.Perspective(
            scale=(0.05, 0.1),
            keep_size=True,
            fit_output=True,
            fill=0,
            fill_mask=255
        ),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2()
    ])
)

val_unet_transform = AlbumentationsAdapter(A.Compose([
        A.LongestMaxSize(572),
        A.PadIfNeeded(min_height=572, min_width=572, border_mode=0, fill=0, fill_mask=255),        
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2()
    ])
)

test_transform = AlbumentationsAdapter(A.Compose([
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2()
    ])
)

train_unet_dataset = PetDataset(train_df, transforms=train_unet_transform)
val_unet_dataset = PetDataset(val_df, transforms=val_unet_transform)
test_unet_dataset = PetDataset(test_unet_df, transforms=test_transform)
test_res_unet_dataset = PetDataset(test_res_unet_df, transforms=test_transform)

class Trainer:
    def __init__(
        self, 
        model, 
        optimizer, 
        scheduler, 
        loss_fn, 
        train_loader, 
        val_loader,
        test_loader,
        hyper_params,
        enable_log,
        log_name
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.hyper_params = hyper_params
        self.enable_log = enable_log
        if enable_log:
            self.log = LogWB(log_name, self.hyper_params)
    
    def _train(self, e):
        loss_val = 0.0
        for data, target in tqdm(self.train_loader, desc=f"Epoch {e}/{self.hyper_params['num_epochs']} - Training"):
                data = data.to(self.hyper_params['device'])
                target = target.to(self.hyper_params['device'])
                
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.loss_fn(output, target)
                loss.backward()
                self.optimizer.step()
                
                loss_val += loss.item()
            
        loss_val /= len(self.train_loader)
        
        return loss_val
                
    @torch.no_grad()
    def _validate(self, e):
        loss_val = 0.0
        for data, target in tqdm(self.val_loader, desc=f"Epoch {e}/{self.hyper_params['num_epochs']} - Validation"):
                data = data.to(self.hyper_params['device'])
                target = target.to(self.hyper_params['device'])
                
                output = self.model(data)
                loss = self.loss_fn(output, target)
                
                loss_val += loss.item()
            
        loss_val /= len(self.train_loader)
        
        return loss_val
    
    @torch.no_grad()
    def _test(self):
        loss_val = 0.0
        for data, target in tqdm(self.test_loader):
                data = data.to(self.hyper_params['device'])
                target = target.to(self.hyper_params['device'])
                
                output = self.model(data)
                loss = self.loss_fn(output, target)
                
                loss_val += loss.item()
            
        loss_val /= len(self.train_loader)
        
        return loss_val
                
    
    def train(self):
        for e in range(self.hyper_params['num_epochs']):
            print(f"Epoch {e}/{self.hyper_params['num_epochs']}")
            self.model.train()
            train_loss = self._train(e)
            self.model.eval()
            val_loss = self._validate(e)
            
            
            if self.scheduler is not None:
                self.scheduler.step()
            
            print(f"Epoch {e}/{self.hyper_params['num_epochs']}: Train loss: {train_loss} Val loss: {val_loss}")

            if self.enable_log:
                self.log.log_data(e, train_loss, val_loss, self.optimizer.param_groups[0]['lr'])
                
                
    def test(self):
        self.model.eval()
        self._test()
                
