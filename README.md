# Challenge
**Task Description:** Using FashionMNIST, devise a pipeline that allows the classification of the 10 clothes classes, without using any labels or only a portion of them (up to 10%). Be specific in explaining which particular SSL methods you used, as well as the transformations applied with regard to the downstream task. As far as possible, try using PyTorch lightning or PyTorch for that purpose.

Deliverable
The final output should be a GitHub repository containing:
- The complete code
- A README.md file explaining how to install, run and visualize results

The assignment shouldn’t take more than 8 hours to complete.

**Solution**
This repository trains a Self Supervised Learning (SSL) model to classify [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist/tree/master?tab=readme-ov-file) dataset into 10 different classes using only up to 10% of the label data. In the following, a visualization of the data:

![alt text](https://github.com/BerardinoB/MomentumHealth/blob/main/Images/fashionMNIST.png)

# Requirements
- numpy
- torch
- tqdm
- sklearn


# How to run the code
In "Package" you can find the train.py and test.py files for training and testing the model respectively. If you do not want to train the model from scratch you can find the weights of the classifier (i.e. encoder of the SSL model) in "checkpoint". If you want to have more information during training consider using "--verbose" flag when training the model  

python train.py --verbose

You can have some flexibility in adjusting the N epochs and the device where the code should run. You can find the complete list in "Package/train.py".

![alt text](https://github.com/BerardinoB/MomentumHealth/blob/main/Images/Image_Autoencoder.png)

The SSL model is essentially a vanilla autoencoder that learns a latent representation of the input into the Z space, which represents the first stage of the model training. In the second stage, the encoder part acts as a classifier. Only the bottleneck (i.e last layer, i.e Z) will be used to fine-tune using up the 10% of the available label data. All the details of the model can be found in "Package/Utils.py"
I also provide the weights of the classifier (i.e. encoder) for each percentage of supervision from 1% to 10%. Have a look at the "checkpoint" folder. Plotting the performance for each percentage of labeled data (on the hold-out test set) will produce the following plot

![alt text](https://github.com/BerardinoB/MomentumHealth/blob/main/Images/performance.png)
