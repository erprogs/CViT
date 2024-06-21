import os
import torch
from torchvision import transforms, datasets
from helpers.augmentation import Aug

mean = [0.485, 0.456, 0.406] 
std = [0.229, 0.224, 0.225]

data_transforms = {
    'train': transforms.Compose([
        Aug(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
    'validation': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
    'test': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
}

def normalize_data():
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    return {
        "train": transforms.Compose(
            [Aug(), transforms.ToTensor(), transforms.Normalize(mean, std)]
        ),
        "valid": transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean, std)]
        ),
        "test": transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean, std)]
        ),
        "vid": transforms.Compose([transforms.Normalize(mean, std)]),
    }

def load_data(data_dir = 'sample/', batch_size=32):
    batch_size=batch_size
    data_dir = data_dir
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'validation', 'test']}
    
    dataloaders, dataset_sizes = load(image_datasets, batch_size, data_dir)
    return batch_size, dataloaders, dataset_sizes

def load(image_datasets, batch_size, data_dir):
    
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size,
                                                 shuffle=True, num_workers=1, pin_memory=True)
                   for x in ['train', 'validation', 'test']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'validation', 'test']}
    
    return dataloaders, dataset_sizes
