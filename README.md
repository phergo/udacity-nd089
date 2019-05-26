# UdacityND089
Udacity AI Programming with Python Nanodegree Final Project

### Purpose
The purpose of this Udacity project was to build and train a deep neural network on the flower data set, and provide an application that others can use.  The application consists of a pair of Python scripts that run from the command line, one for training the neural network, and one to predict the class/category of a flower image.

### Installation

#### Prerequisites
This project was built with the prerequisites found in the `requirements.txt` file, and are as follows:

* Python v3.6.8
* Pytorch v1.1.0
* TorchVision v0.2.2
* Pillow v5.2.0
* NOTE: A GPU, while ideal, is not necessary.

#### Project files installation

Clone GitHub repository.

```
git clone https://github.com/phergo/UdacityND089.git
cd UdacityND089
```

The project contains a small sample of training (`train`), validation (`valid`) and testing (`test`) images under the `flowers_small/` directory for development purposes. The category names for these sample files can be found on the `cat_to_name_small.json` file.

If required, the full set of image files for the Nanodegree program can be installed as described below.  Please note is assumed you are already within the UdacityND089 directory.

```
mkdir flowers
cd flowers
curl https://s3.amazonaws.com/content.udacity-data.com/nd089/flower_data.tar.gz | tar xz
``` 

You should now have `test`, `train` and `valid` directories containing classification directories and flower images under the flowers directory.

Alternatively, you may download the images [here](https://s3.amazonaws.com/content.udacity-data.com/nd089/flower_data.tar.gz).

The full category-to-name mapping for this image set can be found in the `cat_to_name.json` file.

### Training the Neural Network
Train a new network on a data set with `train.py`.

* Basic usage: `python train.py data_directory`

The script prints out the training loss, validation loss, and validation accuracy as the network trains.

Script usage can be obtained as: `python train.py --help`:

```
usage: train.py [-h] [--arch ARCH] [--category_names CATEGORY_NAMES]
                [--checkpoint CHECKPOINT] [--dropout DROPOUT]
                [--epochs EPOCHS] [--gpu] [--hidden_units HIDDEN_UNITS]
                [--learning_rate LEARNING_RATE] [--output_units OUTPUT_UNITS]
                [--save_dir SAVE_DIR]
                data_dir

ND089 Flower classifier Neural Network training module

positional arguments:
  data_dir              The root directory where the train/test/valid images
                        are stored.

optional arguments:
  -h, --help            show this help message and exit
  --arch ARCH           The model architecture name from torchvision.models
  --category_names CATEGORY_NAMES
                        The file path to the JSON file containing the category
                        to names mapping
  --checkpoint CHECKPOINT
                        The checkpoint filename
  --dropout DROPOUT     The dropout for model training
  --epochs EPOCHS       The number of epochs for model training
  --gpu                 Indicates to use GPU for training, if one is available
  --hidden_units HIDDEN_UNITS
                        The number of hidden_units for model training
  --learning_rate LEARNING_RATE
                        The learning_rate for model training
  --output_units OUTPUT_UNITS
                        The output units required by the model
  --save_dir SAVE_DIR   The directory to save checkpoints
```

#### Training script sample run

```
python train.py flowers_small --arch=alexnet --epochs 10 --category_names cat_to_name_small.json --save_dir checkpoints

.....|Epoch   1 / 10, Step     5: Train loss:   2.812.. Test loss:   1.471.. Test accuracy:  0.744
.....|Epoch   2 / 10, Step    10: Train loss:   0.459.. Test loss:   0.743.. Test accuracy:  0.814
.....|Epoch   3 / 10, Step    15: Train loss:   0.107.. Test loss:   0.285.. Test accuracy:  0.907
.....|Epoch   3 / 10, Step    20: Train loss:   0.568.. Test loss:   1.008.. Test accuracy:  0.791
.....|Epoch   4 / 10, Step    25: Train loss:   0.312.. Test loss:   0.449.. Test accuracy:  0.837
.....|Epoch   5 / 10, Step    30: Train loss:   0.198.. Test loss:   0.733.. Test accuracy:  0.791
.....|Epoch   5 / 10, Step    35: Train loss:   0.359.. Test loss:   0.497.. Test accuracy:  0.884
.....|Epoch   6 / 10, Step    40: Train loss:   0.376.. Test loss:   0.481.. Test accuracy:  0.884
.....|Epoch   7 / 10, Step    45: Train loss:   0.205.. Test loss:   0.873.. Test accuracy:  0.860
.....|Epoch   8 / 10, Step    50: Train loss:   0.058.. Test loss:   0.291.. Test accuracy:  0.930
.....|Epoch   8 / 10, Step    55: Train loss:   0.288.. Test loss:   1.180.. Test accuracy:  0.791
.....|Epoch   9 / 10, Step    60: Train loss:   0.234.. Test loss:   0.368.. Test accuracy:  0.930
.....|Epoch  10 / 10, Step    65: Train loss:   0.072.. Test loss:   0.433.. Test accuracy:  0.907
.....|Epoch  10 / 10, Step    70: Train loss:   0.315.. Test loss:   0.889.. Test accuracy:  0.814


DONE: 
Epoch  10 / 10, Step    70: Train loss:   0.000.. Test loss:   0.889.. Test accuracy:  0.814
Total training elapse time: 00:02:18
Checkpoint saved successfully
|Estimated network accuracy (on test dataset):  0.904
```

### Predict using the Neural Network
Predict flower name from an image with `predict.py` along with the probability of that name. That is, you'll pass in a single image /path/to/image and return the flower name and class probability.

* Basic usage: `python predict.py /path/to/image checkpoints`

Script usage can be obtained as: `python predict.py --help`:
```
usage: predict.py [-h] [--category_names CATEGORY_NAMES] [--gpu]
                  [--top_k TOP_K]
                  input checkpoint

ND089 Flower classifier Neural Network prediction module

positional arguments:
  input                 The path to the single image to predict its class
  checkpoint            The pre-trained network checkpoint path

optional arguments:
  -h, --help            show this help message and exit
  --category_names CATEGORY_NAMES
                        The file path to the JSON file containing the category
                        to names mapping
  --gpu                 Indicates to use GPU for training, if one is available
  --top_k TOP_K         Return top "k" most likely classes
```

#### Training script sample run
```
python predict.py 'flowers_small/test/1/image_06743.jpg' checkpoints/alexnet

Checkpoint loaded successfully
[1.0]
['pink primrose']
```

