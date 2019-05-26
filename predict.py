#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# PROGRAMMER: Pablo E. Hernandez
# DATE CREATED: 2019-05-18
# REVISED DATE:
# PURPOSE: Predict the top_k classes of a given flower image file.
#
# SAMPLE CALL:
# python predict.py 'flowers_small/test/1/image_06743.jpg' checkpoints/alexnet

import argparse
import os
from classifier import Classifier

# Initialize the command-line arguments.
parser = argparse.ArgumentParser(description='ND089 Flower classifier Neural Network prediction module')

parser.add_argument('input', action='store',
                    help='The path to the single image to predict its class')

parser.add_argument('checkpoint', action='store',
                    help='The pre-trained network checkpoint path')

parser.add_argument('--category_names', action='store', default='', dest='category_names',
                    help='The file path to the JSON file containing the category to names mapping')

parser.add_argument('--gpu', action='store_true', default=False, dest='gpu',
                    help='Indicates to use GPU for training, if one is available')

parser.add_argument('--top_k', action='store', default=1, type=int, dest='top_k',
                    help='Return top "k" most likely classes')

args = parser.parse_args()

# Create base network, and load the checkpoint for actual run-time parameters for classification.
network = Classifier(use_gpu=args.gpu, category_names_file=args.category_names)
folder, file = os.path.split(args.checkpoint)
network.load_checkpoint(file, folder)
probabilities, classes = network.predict(args.input, args.top_k)
print(probabilities)
print(classes)