#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# PROGRAMMER: Pablo E. Hernandez
# DATE CREATED: 2019-05-02
# REVISED DATE:
# PURPOSE: Build a Neural Network to classify flower images.
from torch import nn


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

    def forward(self, features):
        """Performs a forward pass on the Neural Network.
        """
        return self.forward(features)
