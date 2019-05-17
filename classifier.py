#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# PROGRAMMER: Pablo E. Hernandez
# DATE CREATED: 2019-05-02
# REVISED DATE:
# PURPOSE: Build a Neural Network to classify flower images.

import os
import json
import time
import torch

from PIL import Image
from torch import nn
from torch import optim
from torch import utils
from torch.autograd import Variable
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
            'category_names': self._read_category_names_json(category_names_file),
            'category_to_idx': None,  # Class-To-IDX mapping from training dataset.
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
    def _enforce_checkpoint_extension(file_name):
        root, ext = os.path.splitext(file_name)
        return (root + '.pth') if ext == '' else file_name

    @staticmethod
    def _get_accuracy(logarithmic_probabilities, labels):
        """Calculates the accuracy of the model.

        :param logarithmic_probabilities: The LogSoftmax output from the model.
        :param labels: The known labels to compare against the predicted ones.
        :return: The calculated accuracy of the model.
        """
        ps = torch.exp(logarithmic_probabilities)
        top_ps, top_class = ps.topk(1, dim=1)
        equality = top_class == labels.view(*top_class.shape)
        accuracy = equality.type_as(torch.FloatTensor()).mean()
        return accuracy

    @staticmethod
    def _get_data_directories(data_dir):
        """Sets the paths to test, training and validation directories based on the (root) data_dir specified.

        :param data_dir: Path to root directory containing images. It is assumed 'test', 'train', and 'valid'
            sub-directories exist to be used in the different stages of the network preparation.
        :return: A dictionary containing the path to the test, training and validation image directories.
        """
        data_dirs = {
            'test': os.path.join(data_dir, 'test'),
            'train': os.path.join(data_dir, 'train'),
            'valid': os.path.join(data_dir, 'valid'),
        }

        # Check the existence of all required sub-directories.
        for ds in data_dirs:
            if not os.path.isdir(data_dirs[ds]):
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

    @staticmethod
    def _get_idx_to_class(class_to_idx):
        """Creates an ordered list with the class ID's based on the class_to_idx dictionary provided.

        The standard dataset.class_to_idx dictionary has a mapping between the class ID (key) and the tensor index
        (value); however, the predict() method returns the indexes from which the class IDs need to be obtained.
        Performing a dictionary key look-up by value every time a class ID is needed is a costly process, hence, this
        method builds a list in which every position (i.e. index) of the list represents the corresponding class ID
        of the same index in the probabilities tensor/array. This list can be stored in the checkpoint and reused as
        many times as needed.

        :param class_to_idx: The class_to_idx dictionary from the 'train' dataset.
        :return: List of class ID's. Each position has the matching class ID for the network's probabilities output.
        """
        lst = [None] * len(class_to_idx)
        for key, value in class_to_idx.items():
            lst[value] = key
        return lst

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

    def _print_stats(self, epoch, step, running_loss, test_loss, accuracy, sample_size, data_length):
        """Print the stats for the running training phase.

        :param epoch: The current epoch iteration
        :param step: The current step, i.e. image batch number being processed.
        :param running_loss: The running loss for the current sample size.
        :param test_loss: The test loss for the current sample size.
        :param accuracy: The accuracy for the current sample size.
        :param sample_size: The size of the current processing sample.
        :param data_length: The length of the dataloader used to generate the statistics.
        """
        print(f"Epoch {epoch: >3} / {self.config['epochs']}, "
              f"Step {step: >5}: "
              f"Train loss: {running_loss / sample_size:7.3f}.. "
              f"Test loss: {test_loss / data_length:7.3f}.. "
              f"Test accuracy: {accuracy / data_length:6.3f}")

    @staticmethod
    def _read_category_names_json(file_path):
        """Reads the contents of the specified JSON file

        :param file_path: The path to the JSON file for class-to-idx mapping
        :return: The class-to-idx mapping
        """
        try:
            with open(file_path.strip(), 'r') as f:
                return json.load(f)
        except Exception:
            print(f'WARNING: error reading file "{file_path}"; no mapping used.')
            return None

    @staticmethod
    def _seconds_to_hhmmss(elapsed):
        hh = int((elapsed / 3600))
        mm = int((elapsed % 3600) / 60)
        ss = int((elapsed % 3600) % 60)
        return str(hh).zfill(2), str(mm).zfill(2), str(ss).zfill(2)

    @staticmethod
    def _spinning_cursor():
        """Yields spinning-cursor characters in sequence.

        :return: A spinning-cursor character.
        """
        while True:
            for cursor in '|/-\\':
                yield cursor + '\b'

    def _validation(self, data_loader, show_progress=True):
        """Test a trained model using the specified data_loader

        :param data_loader: The image set data_loader to be tested.
        :param show_progress: Whether a series of dots are shown to indicate progress
        :return: The (test_loss, accuracy) tuple.
        """
        spinner = self._spinning_cursor()
        accuracy = test_loss = 0
        with torch.no_grad():
            self.model.eval()  # Prevent model from training on validation.
            for images, labels in data_loader:
                print(next(spinner) if show_progress else '', end='', flush=True)
                images, labels = images.to(self.device), labels.to(self.device)
                output = self.model.forward(images)
                test_loss += self.criterion(output, labels).item()
                accuracy += self._get_accuracy(output, labels)
            self.model.train()
        return test_loss, accuracy, len(data_loader)

    def forward(self, features):
        """Performs a forward pass on the Neural Network.
        """
        return self.model.forward(features)

    def load_checkpoint(self, file_name, folder_name='checkpoints'):
        """Loads a Torch checkpoint with the trained network from a previous run.

        :param folder_name: The source folder (dir) name from where the checkpoint will be loaded.
        :param file_name: The source file name where the checkpoint is stored.
        """
        file_name = '' if file_name is None else file_name.strip()
        folder_name = '' if folder_name is None else folder_name.strip()
        checkpoint = torch.load(os.path.join(folder_name, self._enforce_checkpoint_extension(file_name)))
        self.config = checkpoint['config']
        self._initialize_network()  # Initialize the network with the retrieved configuration.
        self.model.load_state_dict(checkpoint['state_dict'])
        print('Checkpoint loaded successfully')  # Will not get here if exception.

    def predict(self, image_path, top_k=5):
        """Predict the specified top classes of the provided image (file_path).

        :param image_path: The file path of the image to classify/predict.
        :param top_k: The number of most likely classes to return.
        :return: The top_k most likely classes of the image (filename) specified.
        """
        image = Image.open(image_path)
        img_tensor = (self.data_transforms['test'](image)).float()
        img_variable = Variable(img_tensor.unsqueeze(0))
        img_variable = img_variable.to(self.device)

        with torch.no_grad():
            self.model.eval()  # Prevent model from training while predicting image.
            nn_output = self.model.forward(img_variable)
            ps = torch.exp(nn_output)
            probabilities, classes = ps.topk(top_k, dim=1)
            self.model.train()

        # Convert top probabilities and classes to python lists as per project requirements.
        probabilities = probabilities.tolist()[0]
        classes = classes.tolist()[0]

        return probabilities, classes

    def save_checkpoint(self, file_name, folder_name='checkpoints'):
        """Saves a Torch checkpoint with the trained network for later use.

        :param folder_name: The target folder (dir) name where the checkpoint will be saved.
        :param file_name: The target file name that will be used for the checkpoint. Common extension is .pth
        """
        checkpoint = {'config': self.config,
                      'state_dict': self.model.state_dict()}
        file_name = '' if file_name is None else file_name.strip()
        folder_name = '' if folder_name is None else folder_name.strip()
        if folder_name != '':
            if not os.path.isdir(folder_name):
                os.makedirs(folder_name)
        torch.save(checkpoint, os.path.join(folder_name, self._enforce_checkpoint_extension(file_name)))
        print('Checkpoint saved successfully')  # Will not get here if exception

    def test(self, show_progress=True):
        """Perform validation on the test dataset in order to establish the model's accuracy.

        :param show_progress: Whether a series of dots are shown to indicate progress
        """
        if self.data_dirs is None:
            print('No input data folders specified, unable to test the network.')
            return None

        test_loss, accuracy, _ = self._validation(self.data_loaders['test'], show_progress)
        print(f"Estimated network accuracy (on test dataset): {accuracy / len(self.data_loaders['test']):6.3f}")

    def train(self, print_every=5, show_progress=True):
        """Perform the training of the model's custom classifier.

        :param print_every: Print statistics every 'print_every' steps
        :param show_progress: Whether a series of dots are shown to indicate progress
        """
        if self.data_dirs is None:
            print('No input data folders specified, unable to train the network.')
            return None

        accuracy = epoch = running_loss = step = test_loss = 0
        start_time = time.time()
        for epoch in range(1, self.config['epochs']+1):
            running_loss = 0
            for images, labels in self.data_loaders['train']:
                step += 1
                print('.' if show_progress else '', end='', flush=True)
                images, labels = images.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()  # Don't forget to zero-out the gradients!!!
                output = self.forward(images)
                loss = self.criterion(output, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()

                if step % print_every == 0:
                    test_loss, accuracy, data_length = self._validation(self.data_loaders['valid'], show_progress)
                    self._print_stats(epoch, step, running_loss, test_loss, accuracy, print_every, data_length)
                    running_loss = 0
        self.config['category_to_idx'] = self._get_idx_to_class(self.data_sets['train'].class_to_idx)
        elapsed_time = time.time() - start_time

        print('\n\nDONE: ')
        self._print_stats(epoch, step, running_loss, test_loss, accuracy, print_every, data_length)
        print("Total training elapse time: {0}:{1}:{2}".format(*self._seconds_to_hhmmss(elapsed_time)))
