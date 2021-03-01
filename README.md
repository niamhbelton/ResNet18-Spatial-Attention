# Optimising Musculoskeletal Knee Injury Detection with Spatial Attention and Extracting Features for Explainability

This repository is under construction.
This is the official repository for implementing ResNet18 + Spatial Attention as outlined in the paper "Optimising Musculoskeletal Knee Injury Detection with Spatial Attention and Extracting Features for Explainability"



# Abstract


![GitHub Logo](/images/arc5.png)

# Dataset 
The MRNet dataset is available for download at the following link https://stanfordmlgroup.github.io/competitions/mrnet/.
This will download a folder named 'data'. There are two subfolders within data 'valid' and 'train'.

# Code Tutorial
Repository split into Single_plane and Multi_plane

## A Single Plane

### (i) Training 
The following arguments are to train the single plane models.
'-directory' is the directory to within the data folder. 

<img src="/images/train_arguments.png" width="800" height="300"/>


The following command is an example of running a model to detect abnormalities using the single plane technique on data from the axial plane.

<img src="/images/train_single_plane_command.png" width="550" height="20"/>

### (ii) Testing 

<img src="/images/test_single_plane_arguments.png" width="650" height="250"/>

<img src="/images/test_single_plane_command.png" width="700" height="20"/>


# Citations
Ahmed, B, (2019). Deep learning in medical imaging: How to automate the detection of knee injuries in MRI exams ?, URL: https://github.com/ahmedbesbes/mrnet


