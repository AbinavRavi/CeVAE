## Introduction

### Motivation:
    Deep learning currently works on human level for tasks on image classification and this is possible only when the distribution of Input data and the distribution of the output data are similar i.e. if the classifier is being trained on classes of sausages, burger and fries as input then the classifier does a very human like job in identifying only these classes but when a baguette is shown the predictions tends to be overconfident and they predict the baguette as one of the three classes that it knows. This overconfident predictions can be very dangerous in applications that require high precision such as Autonomous driving or Medical Imaging. The cost of such overconfident predictions may be very expensive in above mentioned application and hence the classifier should be able to detect things that are out of it's input distribution thereby having a distinction between what it knows and what it doesn't. 

    In the medical field the cost of obtaining labels for task of segmentation and classification is a very expensive one and hence using deep learning classifiers in traditional sense with labels is very expensive and tedious with many labels for each class. Instead we can flip the problem by making the classifier learn about healthy class only and then flag the cases that do not conform to the definition of healthy according to the classifier as an anomaly. This is generally case of Unsupervised anomaly detection. 

    Previous work in the field of Anomaly detection in medical images generally use Autoencoders, Variational Autoencoders , Generative adversarial networks and its variations to learn the distribution of healthy images and then use the reconstruction to flag whether an image is anomalous or not <!--- TO DO: Improve this paragraph and include references--->

    references for 3rd para:
    * AnoGAN
    * f-ANoGAN
    * VAE
    * VAE-KL
    * CeVAE
    * PchVAE
    * AnoVAEGAN
    
### Problem Statement

Deep learning can learn very non-linear features in deep neural networks that does the task with precision. deep learning has been very successful in medical applcations. The goal of this thesis is to implement a framework that implements

* Self supervised learning to learn the distribution of the images of CT scan
* Develop anomaly detection with a new framework of Out of Distribution detection
* provide a Model that can detect the anomaly and localise the location of anomaly.