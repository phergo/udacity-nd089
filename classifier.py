#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# PROGRAMMER: Pablo E. Hernandez
# DATE CREATED: 2019-05-02
# REVISED DATE:
# PURPOSE: Build a Neural Network to classify flower images.

import time
import torch
from os import path
from torch import nn
from torch import optim
from torch import utils
from torchvision import datasets
from torchvision import models
from torchvision import transforms


class Classifier(nn.Module):
    """Custom Neural Network for Flower Image Classifier.
    Udacity - AI Programming with Python Nanodegree Program (ND089) Final Project
    """

    def __init__(self, data_dir=None, output_units=102, architecture='vgg16', epochs=1, hidden_units=512,
                 learning_rate=0.001, dropout=0.2, use_gpu=True, category_names_file=None):
        """Builds a pre-trained Neural Network based on the architecture model specified.

        :param data_dir: Path to root directory containing images. It is assumed 'test', 'train', and 'valid'
            sub-directories exist to be used in the different stages of the network preparation. If not specified
            the network can't be trained; however, it could be used for classification.
        :param output_units: The number of output units required for the neural network.
        :param architecture: Base architecture name. Valid options are from the torchvision.models set, which
            have been pre-trained with ImageNet data set. Sample options: resnet18, alexnet, vgg16, densenet, etc.
            Default value is vgg16
        :param epochs: The number of epochs to use during classifier training.
        :param hidden_units: The number of hidden units to be used for classifier training.
        :param learning_rate: The learning rate to be used during classifier training.
        :param dropout: The model's dropout level required to prevent neural network from overfitting.
        :param use_gpu: Indicates whether GPU should be used, if at all available.
        :param category_names_file: text filename containing class-to-name mapping in JSON format.
        """
        super(Classifier, self).__init__()

        # Keep all model's run-time settings in a dictionary in order to make it simpler to save/load model
        # checkpoints. The config attribute provides a single-point-of-entry for all Network configuration settings
        # that need to be persisted in a checkpoint.
        self.config = {
            'architecture': architecture,
            'dropout': float(dropout),
            'epochs': int(epochs),
            'hidden_units': int(hidden_units),
            'learning_rate': float(learning_rate),
            'input_units': 1,  # To be set during network initialization based on architecture.
            'output_units': int(output_units),
        }

        # Instance agnostic elements, no need to save-to/load-from checkpoint.
        self.criterion = nn.NLLLoss()
        self.data_dirs = self._get_data_directories(data_dir)
        self.data_transforms = self._get_data_transforms()
        self.data_sets = self._get_image_datasets()
        self.data_loaders = self._get_data_loaders()
        self.device = self._get_processing_device(use_gpu)

        # Class-level attributes declaration and initialization...
        # The initialization method is encapsulated so that it can be called after checkpoint reload.
        self.model = None
        self.optimizer = None
        self._initialize_network()

    @staticmethod
    def _get_data_directories(data_dir):
        """Sets the paths to test, training and validation directories based on the (root) data_dir specified.

        :param data_dir: Path to root directory containing images. It is assumed 'test', 'train', and 'valid'
            sub-directories exist to be used in the different stages of the network preparation.
        :return: A dictionary containing the path to the test, training and validation image directories.
        """
        data_dirs = {
            'test': path.join(data_dir, 'test'),
            'train': path.join(data_dir, 'train'),
            'valid': path.join(data_dir, 'valid'),
        }

        # Check the existence of all required sub-directories.
        for ds in data_dirs:
            if not path.isdir(data_dirs[ds]):
                raise NotADirectoryError(f'Directory "{data_dirs[ds]}" does not exist for the "{ds}" data set.')
        return data_dirs

    def _get_image_datasets(self):
        """Loads the datasets using ImageFolder method.

        :return: the image datasets loaded with ImageFolder method.
        """
        return {
            'test':  datasets.ImageFolder(self.data_dirs['test'],  transform=self.data_transforms['test']),
            'train': datasets.ImageFolder(self.data_dirs['train'], transform=self.data_transforms['train']),
            'valid': datasets.ImageFolder(self.data_dirs['valid'], transform=self.data_transforms['valid']),
        }

    @staticmethod
    def _get_input_units(model):
        """Return the input unit tensor size based on the current model selected.

        :param model: The model instance object for which the input size must be selected.
        :return: The input features size for the architecture specified.
        """
        if type(model).__name__ == 'AlexNet':
            return model.classifier[1].in_features
        elif type(model).__name__ == 'DenseNet':
            return model.classifier.in_features
        elif type(model).__name__ == 'Inception':
            return model.fc.in_features
        elif type(model).__name__ == 'ResNet':
            return model.fc.in_features
        elif type(model).__name__ == 'SqueezeNet':
            return model.classifier[1].in_channels
        elif type(model).__name__ == 'VGG':
            return model.classifier[0].in_features
        else:
            return 0

    def _get_data_loaders(self):
        """Defines the dataloaders using the defined datasets.

        :return: The dataloaders used to retrieve the images for training, validation and testing.
        """
        return {
            'test':  utils.data.DataLoader(self.data_sets['test'],  batch_size=64),
            'train': utils.data.DataLoader(self.data_sets['train'], batch_size=64, shuffle=True),
            'valid': utils.data.DataLoader(self.data_sets['valid'], batch_size=64),
        }

    @staticmethod
    def _get_data_transforms():
        """Defines the transforms for the training, validation, and testing sets.

        The pre-trained networks were trained on the ImageNet dataset where each color channel was normalized
        separately. All three sets, the means, and standard deviations of the images are normalized to what the network
        expects. For the means, it's [0.485, 0.456, 0.406] and for the standard deviations [0.229, 0.224, 0.225],
        calculated from the ImageNet images. These values will shift each color channel to be centered at 0 and range
        from -1 to 1.

        :return: the transforms dictionary for the training, validation, and testing sets.
        """
        return {
            'train': transforms.Compose([transforms.RandomRotation(30),
                                         transforms.RandomResizedCrop(224),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.ToTensor(),
                                         ]),

            'valid': transforms.Compose([transforms.Resize(255),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                                         ]),

            'test': transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                                        ])
        }

    def _get_model(self):
        """Builds the actual model to be used in the classifier of our Neural Network. The feature parameters will
        be frozen and a custom classifier will be added for training.

        :return: The pre-trained model with frozen features and custom classifier.
        """
        try:
            model_factory = getattr(models, self.config['architecture'])
        except AttributeError:
            raise ValueError(f'Unknown model architecture {self.config["architecture"]}.')

        # Instantiate the required model.
        model = model_factory(pretrained=True)
        self.config['input_units'] = self._get_input_units(model)

        # Freeze the features parameters of the pre-trained network, we will only be training the classifier.
        for param in model.parameters():
            param.requires_grad = False

        # Define the custom classifier for training.
        clr = nn.Sequential(
            nn.Linear(self.config['input_units'], self.config['hidden_units']),
            nn.ReLU(),
            nn.Dropout(p=self.config['dropout']),
            nn.Linear(self.config['hidden_units'], self.config['output_units']),
            nn.LogSoftmax(dim=1))

        # Attach the custom classifier to the model.
        # All known models use either 'fc' or 'classifier' attribute.
        if hasattr(model, 'fc'):
            model.fc = clr
        else:
            model.classifier = clr

        # Return the created model.
        return model

    @staticmethod
    def _get_processing_device(use_gpu):
        if not use_gpu:
            return torch.device('cpu')
        else:
            if torch.cuda.is_available():
                device = torch.device('cuda')
            else:
                device = torch.device('cpu')
                print('WARNING: a GPU is not available, using the CPU instead.')
        return device

    def _initialize_network(self):
        """Initializes the Neural Network with the current run-time attributes.

        This method is intended to be called during object construction or alternatively after a checkpoint has been
        reloaded and potentially run-time attributes changed as per checkpoint.
        """
        self.model = self._get_model()
        if hasattr(self.model, 'fc'):
            self.optimizer = optim.Adam(self.model.fc.parameters(), lr=self.config['learning_rate'])
        else:
            self.optimizer = optim.Adam(self.model.classifier.parameters(), lr=self.config['learning_rate'])
        self.model.to(self.device)

    def forward(self, features):
        """Performs a forward pass on the Neural Network.
        """
        return self.model.forward(features)

    def train(self, print_every=5):
        """Perform the training of the model's custom classifier.

        :param print_every: Print statistics every 'print_every' steps
        """
        if self.data_dirs is None:
            print('No input data folders specified, unable to train the network.')
            return None

        trainloader = self.data_loaders['train']

        step = 0
        start_time = time.time()
        for epoch in range(1, self.config['epochs']+1):
            running_loss = 0
            for images, labels in trainloader:
                step += 1
                images, labels = images.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()  # Don't forget to zero-out the gradients!!!
                output = self.forward(images)
                loss = self.criterion(output, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()

                if step % print_every == 0:
                    print('Validating...')
                    running_loss = 0
        elapsed_time = time.time() - start_time
        print(elapsed_time)
