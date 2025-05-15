# Noisy hybrid quantum-classical convolutional neural networks for multi-class image classification

## Key features
This library allows to investigate the performance of a noisy hybrid quantum-classical convolutional neural networks and compare it to its classical counterpart.

The library allows to create, train and validate CNNs composed of:
- a single quantum convolutional layer;
- a flattening operation;
- a single fully-connected layer;
- a softmax layer.

As in the classical case, the quantum convolutional layer acts on the input image by extracting sliding blocks from it and performing an operation - called filtering - on each of these blocks. However, unlike the ordinary case, the filtering operation relies upon the execution of a (variational) quantum circuit. More specifically, the $N$ pixel values of each sliding block are mapped into a $N$-qubit variational quantum circuit (VQC) by means of a particular arrangement of non-trainable parametric quantum gates, which compose the so-called "feature map". The remaining part of the VQC, usually referred to as "ansatz", features trainable parametric quantum gates. Finally, the VQC is executed a number of times and measurements in the computational basis on the output quantum state are performed, so to give an estimate for its $2^N$ probability coefficients. The obtained $2^N$ real values are fed into the $2^N$ output images, which are then created by means of a single filter. The number of trainable parameters in the ansatz can be chosen arbitrarily, so that one could in principle explore the chance to obtain better performance with fewer parameters.

The parameters update of the whole net is performed by means of the mini-batch gradient descent algorithm. In order to differentiate the quantum filter's output, the parameter-shift rule is applied.

The whole model is inspired to that proposed by Junhua Liu [1].

Users have to possibility to decide whether to execution the VQC contained within the filter with or without quantum noise. Moreover, they can choose to introduce one or more noise models.

## Usage
To use the library, follow the steps below.
1. Set the quantum filter architecture (feature map, ansatz, noise etc.), the model hyperparameters (epochs, learning rate etc.), the path of the dataset (dataset_folder_path) and the path of the csv file for saving the evolution of the metrics along the training (csv_path) in the configuration file. The configuration file location is `conf\config.yaml`.
2. Execute `python main.py` in the terminal to start training and validation. The evolution of the metrics along the training procedure are saved in `csv_path`, indicated in the configuration file.

A default 4-class synthetic dataset of 3x3 images, called 'Tetris', is placed in the root directory of the library. This dataset is composed of 640 training images, 160 validation images and 200 test images. Each image represent a tetris brick. The grey value of the brick pixels is randomly chosen between 0.7 and 1, whereas that of the background pixels between 0 and 0.3.

## Author
Giuseppe D'Ambruoso (Lutech S.p.A., University of Bari Aldo Moro).

## Acknowledgements
This library was realized with the indinspensable help of dr. Dario del Sorbo (Lutech S.p.A.), dr. Claudio Basilio Caporusso (Lutech S.p.A.), dr. Ivan Palmisano (Lutech S.p.A.), dr. Giuseppe Lamanna (Lutech S.p.A.) and prof. dr. Giovanni Gramegna (University of Bari Aldo Moro). We acknowledge also the support from the whole Lutech R&D 'MILE' pole, directed by dr. Giuseppe Ieva (Lutech S.p.A.).

## References
[1] Liu et al. (2021). _Hybrid quantum-classical convolutional neural networks_. Science China Physics, Mechanics &amp; Astronomy. DOI: 10.1007/s11433-021-1734-3. url: http://dx.doi.org/10.1007/s11433-021-1734-3.
