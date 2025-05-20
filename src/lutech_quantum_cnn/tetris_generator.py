import os
import numpy as np
from PIL import Image
import random

# Set the random seed for reproducibility
random.seed(42)
np.random.seed(42)

def generate_tetris_brick(brick_type):
    """Generates a 3x3 grayscale image of a Tetris brick with random orientation and pixel values."""
    orientations = {
        'S': [
            np.array([[0, 1, 1], [1, 1, 0], [0, 0, 0]]),
            np.array([[1, 0, 0], [1, 1, 0], [0, 1, 1]]),
            np.array([[0, 0, 0], [0, 1, 1], [1, 1, 0]]),
            np.array([[1, 1, 0], [0, 1, 1], [0, 0, 0]]),
        ],
        'L': [
            np.array([[0, 0, 1], [1, 1, 1], [0, 0, 0]]),
            np.array([[1, 0, 0], [1, 0, 0], [1, 1, 0]]),
            np.array([[0, 0, 0], [1, 1, 1], [1, 0, 0]]),
            np.array([[0, 1, 1], [0, 1, 0], [0, 1, 0]]),
        ],
        'O': [
            np.array([[1, 1, 0], [1, 1, 0], [0, 0, 0]]),
            np.array([[0, 1, 1], [0, 1, 1], [0, 0, 0]]),
            np.array([[1, 1, 0], [1, 1, 0], [0, 0, 0]]), # Included again for balance
            np.array([[0, 1, 1], [0, 1, 1], [0, 0, 0]]), # Included again for balance
        ],
        'T': [
            np.array([[0, 1, 0], [1, 1, 1], [0, 0, 0]]),
            np.array([[0, 1, 0], [0, 1, 1], [0, 1, 0]]),
            np.array([[0, 0, 0], [1, 1, 1], [0, 1, 0]]),
            np.array([[1, 0, 0], [1, 1, 0], [1, 0, 0]]),
        ],
    }

    brick_mask = random.choice(orientations[brick_type])
    image_array = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            if brick_mask[i, j] == 1:
                image_array[i, j] = random.uniform(0.7, 1.0)
            else:
                image_array[i, j] = random.uniform(0.0, 0.3)

    # Scale to 0-255 for image representation
    image_array = (image_array * 255).astype(np.uint8)
    return Image.fromarray(image_array), brick_type

def create_dataset(num_samples, output_dir):
    """Creates a balanced dataset of Tetris brick images, organized by class."""
    os.makedirs(output_dir, exist_ok=True)
    samples_per_class = num_samples // 4
    classes = ['S', 'L', 'O', 'T']

    for brick_type in classes:
        class_dir = os.path.join(output_dir, brick_type)
        os.makedirs(class_dir, exist_ok=True)  # Create class-specific directory
        for i in range(samples_per_class):
            image, label = generate_tetris_brick(brick_type)
            image.save(os.path.join(class_dir, f"{label}_{i}.png"))
    print(f"Generated {num_samples} images in '{output_dir}'")

# Define dataset sizes and output directory
total_samples = 1000
train_size = 640
validation_size = 160
test_size = 200

output_base_dir = "Tetris"
os.makedirs(output_base_dir, exist_ok=True)

# Create the datasets
create_dataset(train_size, os.path.join(output_base_dir, "Training"))
create_dataset(validation_size, os.path.join(output_base_dir, "Validation"))
create_dataset(test_size, os.path.join(output_base_dir, "Test"))

print("Dataset creation complete.")