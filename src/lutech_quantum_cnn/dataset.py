import os

import torch
from torch import manual_seed
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Resize, Grayscale, ToTensor

manual_seed(42)

def clear_folder(folder_path: str) -> None:
    """Remove all hidden files and/or hidden subfolder
    from a given folder.

    Parameters
    ----------
    folder_path : str
        The name of the folder to be cleared.

    Returns
    -------
    None
        This function simply modifies the input folder
    """
    content = os.listdir(folder_path)
    hidden_files = [f for f in content if f.startswith(".")]
    for hidden_file in hidden_files:
        hidden_file_path = os.path.join(folder_path, hidden_file)
        if os.path.isdir(hidden_file_path):
            os.rmdir(hidden_file_path)  # Remove directory if it's hidden
        else:
            os.remove(hidden_file_path)  # Remove file if it's hidden


def num_classes(dataset_folder_path: str) -> int:
    """Return the number of classes the data of a given dataset are
    categorized in by assuming that all the data are contained within
    a folder with the following structure:
    dataset_folder
        ├── Training
            ├── Class 1
            ├── Class 2
            ├── Class 3
            ...
        ├── Test
            ├── Class 1
            ├── Class 2
            ├── Class 3
            ...
        ├── Validation
            ├── Class 1
            ├── Class 2
            ├── Class 3
            ...
    To determine it, this function counts the number of subsubfolders
    contained within the Training subfolder after removing all hidden
    files and/or hidden subsubfolders.

    Parameters
    ----------
    dataset : str
        The name of the dataset folder.

    Returns
    -------
    int
        The number of classes the data are categorized in.
    """
    # Define the directory for training set
    train_dir = os.path.join(dataset_folder_path, "Training")

    # Remove hidden folders in the training set directory
    clear_folder(folder_path=train_dir)

    # determine the number of classes by counting the number of folders
    num_classes = len(
        [
            name
            for name in os.listdir(train_dir)
            if os.path.isdir(os.path.join(train_dir, name))
        ]
    )
    return num_classes


def clear_dataset(dataset_folder_path: str) -> None:
    """Given a dataset folder with the following structure:
    dataset_folder_path
        ├── Training
            ├── Class 1
            ├── Class 2
            ├── Class 3
            ...
        ├── Test
            ├── Class 1
            ├── Class 2
            ├── Class 3
            ...
        ├── Validation
            ├── Class 1
            ├── Class 2
            ├── Class 3
            ...
    remove all hidden files and/or folders from each folder.

    Parameters
    ----------
    dataset : str
        The name of the dataset folder.

    Returns
    -------
    None
        This function simply modifies the input folder.
    """
    clear_folder(dataset_folder_path)

    train_dir = os.path.join(dataset_folder_path, "Training")
    clear_folder(train_dir)
    val_dir = os.path.join(dataset_folder_path, "Validation")
    clear_folder(val_dir)
    test_dir = os.path.join(dataset_folder_path, "Test")
    clear_folder(test_dir)

    dirs = [
        os.path.join(train_dir, "L"),
        os.path.join(train_dir, "O"),
        os.path.join(train_dir, "S"),
        os.path.join(train_dir, "T"),
        os.path.join(val_dir, "L"),
        os.path.join(val_dir, "O"),
        os.path.join(val_dir, "S"),
        os.path.join(val_dir, "T"),
        os.path.join(test_dir, "L"),
        os.path.join(test_dir, "O"),
        os.path.join(test_dir, "S"),
        os.path.join(test_dir, "T"),
    ]

    for dir in dirs:
        clear_folder(dir)


def load_dataset(
    dataset_folder_path: str,
    batch_size: int,
    drop_last: bool = True,
) -> tuple:
    """Given a dataset folder with the following structure:
    dataset_folder
        ├── Training
            ├── Class 1
            ├── Class 2
            ├── Class 3
            ...
        ├── Test
            ├── Class 1
            ├── Class 2
            ├── Class 3
            ...
        ├── Validation
            ├── Class 1
            ├── Class 2
            ├── Class 3
            ...
    return the training, validation and test sets. Differently from the last
    two, the first is divided into batches.

    Parameters
    ----------
    dataset_folder_path : str
        The path of the folder containing the dataset.
    batch_size : int
        The size of the batch into which the training dataset is
        divided during the training phase.
    drop_last : bool
        Discard the last of the training dataset batch if incomplete

    Returns
    -------
    tuple
        A tuple containing the training, validation and test sets.
    """

    # Define the directories for train, validation and test
    clear_dataset(dataset_folder_path)

    train_dir = os.path.join(dataset_folder_path, "Training")
    validation_dir = os.path.join(dataset_folder_path, "Validation")
    test_dir = os.path.join(dataset_folder_path, "Test")

    # Load datasets
    n_classes = num_classes(dataset_folder_path=dataset_folder_path)

    # Pre-processing operations for data and labels
    transform = Compose([Resize(3), Grayscale(num_output_channels=1), ToTensor()])
    target_transform = Compose(
        [
            lambda x: torch.tensor(x),
            lambda x: torch.eye(n=n_classes)[x].to(torch.float64),  # ohe
        ]
    )

    train_dataset = ImageFolder(
        root=train_dir, transform=transform, target_transform=target_transform
    )
    validation_dataset = ImageFolder(
        root=validation_dir, transform=transform, target_transform=target_transform
    )
    test_dataset = ImageFolder(
        root=test_dir, transform=transform, target_transform=target_transform
    )

#    # Limit the datasets to the first 10 images
#    train_dataset = Subset(train_dataset, range(min(10, len(train_dataset))))
#    validation_dataset = Subset(validation_dataset, range(min(10, len(validation_dataset))))
#    test_dataset = Subset(test_dataset, range(min(10, len(test_dataset))))

    # Create data loaders for train, validation, and test datasets
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=drop_last
    )
    validation_loader = DataLoader(
        dataset=validation_dataset,
        batch_size=len(validation_dataset),
        shuffle=False
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=len(test_dataset),
        shuffle=False
    )
    return train_loader, validation_loader, test_loader