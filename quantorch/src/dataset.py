import os
import torch
import medmnist
from typing import Tuple
from medmnist import INFO
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Resize, Grayscale, ToTensor, Normalize

def num_classes(dataset: str,
                local: bool) -> int:
    if dataset == 'breastMNIST':
        info = INFO['breastmnist']
        num_classes = len(info['label'])
    else:
        # Define the directory for training set
        root_path : str
        if local == True:
            root_path = r'C:\Users\giuseppe.dambruoso\OneDrive - LUTECH SPA\Desktop\Progetto\Dataset' +r'\\' + dataset
        elif local == False:
            root_path = dataset
        train_dir = os.path.join(root_path, 'Training')

        # Remove hidden folders in the training set directory
        all_files = os.listdir(train_dir)
        # Filter out hidden files and directories (starting with ".")
        hidden_files = [f for f in all_files if f.startswith('.')]
        # Delete hidden files and directories
        for hidden_file in hidden_files:
            hidden_file_path = os.path.join(train_dir, hidden_file)
            if os.path.isdir(hidden_file_path):
                # Remove directory if it's hidden
                os.rmdir(hidden_file_path)
            else:
                # Remove file if it's hidden
                os.remove(hidden_file_path)

        # determine the number of classes by counting the number of folders
        num_classes = len([name for name in os.listdir(train_dir) if 
            os.path.isdir(os.path.join(train_dir, name))])
    return num_classes

def load_data_from_folder(
            folder_name: str,
            batch_size: int,
            local : bool,
            drop_last: bool=True) -> tuple:
    """This function returns  tuple containing the data loaders for train,
    test and validation"""

    # Define the directories for train, validation and test
    if local == True:
        root_path = r'C:\Users\giuseppe.dambruoso\OneDrive - LUTECH SPA\Desktop\Progetto\Dataset' + r'\\' + folder_name
    elif local == False:
        root_path = folder_name
    train_dir = os.path.join(root_path, 'Training')
    validation_dir = os.path.join(root_path, 'Validation')
    test_dir = os.path.join(root_path, 'Test')

    # Load datasets
    n_classes = num_classes(dataset=folder_name, local=local)

    transform = Compose([
            Resize(3),
            Grayscale(num_output_channels=1),
            ToTensor()])
    target_transform = Compose([
            lambda x: torch.tensor(x),
            lambda x: torch.eye(
                n=n_classes)[x].to(torch.float64)])

    train_dataset = ImageFolder(
        root=train_dir,
        transform=transform,
        target_transform=target_transform)
    validation_dataset = ImageFolder(
        root=validation_dir,
        transform=transform,
        target_transform=target_transform)
    test_dataset = ImageFolder(
        root=test_dir,
        transform=transform,
        target_transform=target_transform)

    # Create data loaders for train, validation, and test datasets
    train_loader = DataLoader(dataset=train_dataset,
                                batch_size=batch_size,
                                shuffle=True,
                                drop_last=drop_last,
                                pin_memory=True,
                                num_workers=os.cpu_count(),
                                prefetch_factor=2)
    validation_loader = DataLoader(dataset=validation_dataset,
                                batch_size=batch_size,
                                shuffle=False,
                                drop_last=drop_last,
                                pin_memory=True,
                                num_workers=os.cpu_count(),
                                prefetch_factor=2)
    test_loader = DataLoader(dataset=test_dataset,
                                batch_size=batch_size,
                                shuffle=False,
                                drop_last=drop_last,
                                num_workers=os.cpu_count(),
                                pin_memory=True,
                                prefetch_factor=2)
    return train_loader, validation_loader

def load_breastMNIST(batch_size: int, local: bool=False) -> Tuple:
    # preprocessing
    info = INFO['breastmnist']
    n_classes = num_classes('breastMNIST', local)
    data_transform = Compose([
        ToTensor(),
        Normalize(mean=[.5], std=[.5])
    ])
    target_transform = Compose([
                lambda x: torch.tensor(x).squeeze(),
                lambda x: torch.eye(
                    n=n_classes)[x].to(torch.float64)
            ])

    # load the data
    DataClass = getattr(medmnist, info['python_class'])
    train_dataset = DataClass(split='train', transform=data_transform, target_transform=target_transform, download=True, root='breastMNIST')
    validation_dataset = DataClass(split='test', transform=data_transform, target_transform=target_transform, download=True, root='breastMNIST')

    # encapsulate data into dataloader form
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(dataset=validation_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, validation_loader

def load_dataset(
    dataset: str,
    batch_size: int,
    local : bool,
    drop_last: bool=True) -> Tuple:
    if dataset == 'breastMNIST':
        output = load_breastMNIST(
            batch_size=batch_size,
            local=local)
    else: 
        output = load_data_from_folder(
                folder_name=dataset,
                local=local,
                batch_size=batch_size,
                drop_last=drop_last)
    return output