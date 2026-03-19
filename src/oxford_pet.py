import pandas as pd
import cv2
import os
import numpy as np

import albumentations as A
from albumentations.pytorch import ToTensorV2

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def load_df(file_path, image_dir, trimap_dir):
    dataframe_base = {
        'image_path': [],
        'trimap_path': []
    }

    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return pd.DataFrame(dataframe_base)
    
    for line in lines:
        line = line.strip()
        
        if not line:
            continue
        
        dataframe_base['image_path'].append(f"{image_dir}/{line}")
        dataframe_base['trimap_path'].append(f"{trimap_dir}/._{line}")
        
    return pd.DataFrame(dataframe_base)
        
def remove_missing_img(df):
    for r in df.iterrows():
        if not (os.path.exists(r['image_path']) and os.path.exists(r['trimap_path'])):
            df.drop(r.index, inplace=True)
    return df

def safe_load_df(file_path, image_dir, trimap_dir):
    df = load_df(file_path, image_dir, trimap_dir)
    return remove_missing_img(df)

class PetDataset:
    def __init__(self, df, transforms):
        self.df = df
        self.transforms = transforms
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = row['image_path']
        trimap_path = row['trimap_path']
        
        img = cv2.imread(image_path)
        if img is None:
            raise Exception(f"Image not found: {image_path}")
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        trimap = cv2.imread(trimap_path, cv2.IMREAD_GRAYSCALE) 
        
        if trimap is None:
            raise Exception(f"Trimap not found: {trimap_path}")
        
        binary_mask = np.zeros_like(trimap)      
        binary_mask[(trimap==2) | (trimap==3)] = 1        
        
        return img, binary_mask
        
class AlbumentationsAdapter:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, img):
        img_np = np.array(img)
        augmented = self.transform(image=img_np)
        return augmented['image']
