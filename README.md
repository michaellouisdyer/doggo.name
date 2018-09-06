# doggo.name
### Dog breed classifier using deep learning


#### About the project
I love dogs but I can never tell what breed they are.
I also love playing around with neural networks. 
I decided to make an app that can tell me what a dog's breed is just by giving it a picture.

#### Data
I made a script that downloaded 400 pictures of each breed from image search engine.
In the end, I got 50,000 images of 188 different breeds, over 20gb of images.
The data was automatically split into 70% training, 20% validation and 10% holdout.
I saved these to a google cloud drive and saved a small subset of them to my personal computer.

#### Modeling
 Image classification has historically been a difficult task, but modern technologies such as cloud computing and accessible neural network frameworks have make great improvements in the last several years. In particular, Convolutional neural networks (CNNs) have had a huge impact; image classification error in international contests like the ImageNet ILSVRC decreased over 10x from 2010 to 2017, mainly due to advancements in CNNs.
 
 After I tested many different architectures on my personal computer, looking at training and validation performance, I decided to run three main models on the full dataset:
 
 1. Baseline 12-layer simple CNN.
 This is a very vanilla CNN with only two convolutional layers.  I used an adam optimizer as it has been shown to have high performance in image classification.
 2. Xception (no pre-loaded weights)
 The top performing neural net, but I wanted to see how it did when trained from scratch. I used an adam optimizer here as well.
 3. Xception (ImageNet weights)
 The top performing neural net, with pre-trained weights from the ImageNet dataset.
 Based on this paper http://cs231n.stanford.edu/reports/2015/pdfs/lediurfinal.pdf as well as some testing, I decided to do transfer learning by using an adam optimizer with a 0.1x learning rate on all of the layers except for the softmax activation layer. Learning rate by layer is not well supported in Keras, but thankfully I found a custom optimizer that implements this fairly easily. 
 
 #### Training
 I set up a Google Cloud Platform instance with a deep learning VM, 16 vCPUs, 4 NVIDIA Tesla P100 GPUs, 104GB of RAM and attached my 40gb SSD with the full dataset on it to the instance. I trained each model for 20 epochs (with image augmentation) and saved log files using tensorboard. I then ran an evalutation on the holdout set to get accuracy, a confusion matrix and a classification report by breed.

#### Results
Simple CNN: 2.4% holdout accuracy
Xception (no preloaded weights): 64.1% holdout accuracy
Xception (ImageNet weights): 82.7%

#### App
I saved the h5 model file and used it to pass predictions to a flask app, which I then dockerized and deployed to an AWS instance. See doggo.name for a demo.

#### This README is in the works, more to come soon!
