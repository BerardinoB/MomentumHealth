# MomentumHealth
Challenge for MomentumHealth

This repository trains a Self Supervised Learning (SSL) model to classify [MNISTFashon](https://github.com/zalandoresearch/fashion-mnist/tree/master?tab=readme-ov-file) dataset into 10 different classes using only up to 10% of the label data.

# Requirements
- numpy
- torch
- tqdm
- sklearn


# How to run the code
In "Package" you can find the train.py and test.py files for training and testing the model respectively. If you do not want to train the model from scratch you can find the weights of the classifier (i.e. encoder of the SSL model) in "checkpoint". I also provide a notebook on how to visualize some diagnostics of the training, such as train vs validation curves and %lables vs accuracy.

