import os

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import (
    Compose,
    Resize,
    Grayscale,
    ToTensor
)

def num_classes(dataset: str) -> int:
    # Define the directory for training set
    root_path = dataset
    train_dir = os.path.join(root_path, 'Training')

    # Remove hidden folders in the training set directory
    all_files = os.listdir(train_dir)
    hidden_files = [f for f in all_files if f.startswith('.')]
    for hidden_file in hidden_files:
        hidden_file_path = os.path.join(train_dir, hidden_file)
        if os.path.isdir(hidden_file_path):
            os.rmdir(hidden_file_path) # Remove directory if it's hidden
        else:
            os.remove(hidden_file_path) # Remove file if it's hidden

    # determine the number of classes by counting the number of folders
    num_classes = len([name for name in os.listdir(train_dir) if 
        os.path.isdir(os.path.join(train_dir, name))])
    return num_classes

def clear_folder(folder_name : str) -> None:
    layer = os.listdir(folder_name)
    hidden_files = [f for f in layer if f.startswith('.')]
    for hidden_file in hidden_files:
        hidden_file_path = os.path.join(folder_name, hidden_file)
        if os.path.isdir(hidden_file_path):
            os.rmdir(hidden_file_path) # Remove directory if it's hidden
        else:
            os.remove(hidden_file_path) # Remove file if it's hidden

def clear_dataset(dataset_folder_name : str) -> None:
    root_path = dataset_folder_name
    clear_folder(root_path)
    
    train_dir = os.path.join(root_path, 'Training')
    clear_folder(train_dir)
    val_dir = os.path.join(root_path, 'Validation')
    clear_folder(val_dir)
    test_dir = os.path.join(root_path, 'Test')
    clear_folder(test_dir)
    
    dirs = [os.path.join(train_dir, 'L'),
                  os.path.join(train_dir, 'O'),
                  os.path.join(train_dir, 'S'),
                  os.path.join(train_dir, 'T'),
                  os.path.join(val_dir, 'L'),
                  os.path.join(val_dir, 'O'),
                  os.path.join(val_dir, 'S'),
                  os.path.join(val_dir, 'T'),
                  os.path.join(test_dir, 'L'),
                  os.path.join(test_dir, 'O'),
                  os.path.join(test_dir, 'S'),
                  os.path.join(test_dir, 'T')]
    for dir in dirs:
        clear_folder(dir)

def load_dataset(
            folder_name: str,
            batch_size: int,
            drop_last: bool=True) -> tuple:
    """
    This function returns  tuple containing the data loaders for train,
    test and validation
    """

    # Define the directories for train, validation and test
    root_path = folder_name
    clear_dataset(root_path)
    
    train_dir = os.path.join(root_path, 'Training')
    validation_dir = os.path.join(root_path, 'Validation')
    test_dir = os.path.join(root_path, 'Test')

    # Load datasets
    n_classes = num_classes(dataset=folder_name)

    transform = Compose([
            Resize(3),
            Grayscale(num_output_channels=1),
            ToTensor()
        ]
    )
    target_transform = Compose([
            lambda x: torch.tensor(x),
            lambda x: torch.eye(
                n=n_classes)[x].to(torch.float64)
        ]
    )

    train_dataset = ImageFolder(
        root=train_dir,
        transform=transform,
        target_transform=target_transform
    )
    validation_dataset = ImageFolder(
        root=validation_dir,
        transform=transform,
        target_transform=target_transform
    )
    test_dataset = ImageFolder(
        root=test_dir,
        transform=transform,
        target_transform=target_transform
    )

    # Create data loaders for train, validation, and test datasets
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=drop_last
    )                           
    validation_loader = DataLoader(
        dataset=validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=drop_last
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=drop_last
    )
    return train_loader, validation_loader, test_loader