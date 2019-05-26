#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# PROGRAMMER: Pablo E. Hernandez
# DATE CREATED: 2019-05-18
# REVISED DATE:
# PURPOSE: Train a Neural Network to recognize and classify flower images.
#
# SAMPLE CALL:
# python train.py flowers_small --arch=alexnet --epochs 1 --category_names cat_to_name_small.json --save_dir checkpoints

import argparse
from classifier import Classifier

# Initialize the command-line arguments.
parser = argparse.ArgumentParser(description='ND089 Flower classifier Neural Network training module')
parser.add_argument('data_dir', action='store',
                    help='The root directory where the train/test/valid images are stored.')

parser.add_argument('--arch', action='store', default='vgg16', dest='arch',
                    help='The model architecture name from torchvision.models')

parser.add_argument('--category_names', action='store', dest='category_names',
                    help='The file path to the JSON file containing the category to names mapping')

parser.add_argument('--checkpoint', action='store', default='', type=str, dest='checkpoint',
                    help='The checkpoint filename')

parser.add_argument('--dropout', action='store', default=0.2, type=float, dest='dropout',
                    help='The dropout for model training')

parser.add_argument('--epochs', action='store', default=1, type=int, dest='epochs',
                    help='The number of epochs for model training')

parser.add_argument('--gpu', action='store_true', default=False, dest='gpu',
                    help='Indicates to use GPU for training, if one is available')

parser.add_argument('--hidden_units', action='store', default=512, type=int, dest='hidden_units',
                    help='The number of hidden_units for model training')

parser.add_argument('--learning_rate', action='store', default=0.001, type=float, dest='learning_rate',
                    help='The learning_rate for model training')

parser.add_argument('--output_units', action='store', default=102, type=int, dest='output_units',
                    help='The output units required by the model')

parser.add_argument('--save_dir', action='store', default='', type=str, dest='save_dir',
                    help='The directory to save checkpoints')

args = parser.parse_args()

# If no checkpoint filename provided, use the architecture name as checkpoint name.
if args.checkpoint == '':
    args.checkpoint = args.arch

# Create, train, and save the network.
network = Classifier(data_dir=args.data_dir, output_units=args.output_units, architecture=args.arch, epochs=args.epochs,
                     hidden_units=args.hidden_units, learning_rate=args.learning_rate, dropout=args.dropout,
                     use_gpu=args.gpu, category_names_file=args.category_names)
print()
network.train(show_progress=True)
network.save_checkpoint(args.checkpoint, args.save_dir)
network.test()
print()
