# Optimising Musculoskeletal Knee Injury Detection with Spatial Attention and Extracting Features for Explainability

This repository is under construction.
This is the official repository for implementing ResNet18 + Spatial Attention as outlined in the paper "Optimising Musculoskeletal Knee Injury Detection with Spatial Attention and Extracting Features for Explainability"



# Abstract


![GitHub Logo](/images/arc5.png)

# Dataset 
The MRNet dataset is available for download at the following link https://stanfordmlgroup.github.io/competitions/mrnet/.
This will download a folder named 'data'. The dataset consists of 1,250 knee MRIs with image level labels. They are labelled as abnormal, having an acl tear and/or meniscus tear. Each MRI exam includes data from the axial, coronal and sagittal plane. Axial is a Proton-Density series, coronal is a T1-weighted series and sagittal is T2-weighted series.

# Code Tutorial
The repository is split into two folders; Single_plane and Multi_plane. The following sections outline how to train and test single-plane and multi-plane models.

## (A) Single-Plane

### (i) Training 
This section outlines how to train the single-plane models. The image below outlines the arguments for training the single plane models.

<img src="/images/train_arguments.png" width="800" height="300"/>

1. task - specify whether the model is training to detect ACL tears, meniscus tears or abnormalities. This must be equal to 'abnormal', 'acl' or 'meniscus'. 
2. prefix_name - specify the name of the model.
3. plane - specify the plane that the model is training on. This must be equal to 'axial', 'coronal' or 'sagittal'.
4. directory - specify the director where the 'data' folder is stored.
5. epochs - specify the number of epochs.
6. lr - specify the learning rate.
7. patience - specify the number of iterations where there is no decrease in the validation loss before early stopping is triggered.
8. log_every - specify how many iterations must complete before an update on model training is printed out.
9. seed - specify the seed.


The following command is an example of running a model to detect acl tears using the single plane technique on data from the axial plane.

<img src="/images/train_single_plane_command__.png" width="550" height="30"/>

### (ii) Testing 
This section outlines how to test the single-plane models. The image below outlines the arguments for testing single plane models.

<img src="/images/test_single_plane_arguments.png" width="650" height="250"/>

1. task - specify whether the model is training to detect ACL tears, meniscus tears or abnormalities. This must be equal to 'abnormal', 'acl' or 'meniscus'. 
2. plane - specify the plane that the model is training on. This must be equal to 'axial', 'coronal' or 'sagittal'.
3. model_name - specify the name of the model.
4. model_directory - specify the directory of the model.
5. data_directory - specify the director where the 'data' folder is stored.
6. log_every - specify how many iterations must complete before an update on model training is printed out.

The following command is an example of testing the trained acl detection model on the test set. The model was trained on data from the axial plane is being tested on data from the axial plane.
<img src="/images/test_single_plane__command.png" width="900" height="30"/>

## (B) Multi-Plane
### (i) Training
This section outlines how to train the multi-plane models. The image below outlines the arguments for training multi-plane models.

<img src="/images/train_multi_plane_arguments.png" width="800" height="250"/>


1. task - specify whether the model is training to detect ACL tears, meniscus tears or abnormalities. This must be equal to 'abnormal', 'acl' or 'meniscus'. 
2. prefix_name - specify the name of the model.
3. mod - specify where to fuse planes. This must be equal to 'mp1', 'mp2' or 'mp3'.
4. directory - specify the director where the 'data' folder is stored.
5. epochs - specify the number of epochs.
6. lr - specify the learning rate.
7. patience - specify the number of iterations where there is no decrease in the validation loss before early stopping is triggered.
8. log_every - specify how many iterations must complete before an update on model training is printed out.
9. seed - specify the seed.

The following command is an example of running a model to detect acl tears using multi-plane join 1.

<img src="/images/train_multi_plane_command__.png" width="600" height="30"/>

### (ii) Testing 
This section outlines how to test the multi-plane models. The image below outlines the arguments for testing multi-plane models.

<img src="/images/test_multi_plane_arguments.png" width="700" height="200"/>


1. task - specify whether the model is training to detect ACL tears, meniscus tears or abnormalities. This must be equal to 'abnormal', 'acl' or 'meniscus'. 
2. model_name - specify the name of the model.
3. model_directory - specify the directory of the model.
4. data_directory - specify the director where the 'data' folder is stored.
5. log_every - specify how many iterations must complete before an update on model training is printed out.

The following command is an example of testing the trained acl detection model on the test set. 
<img src="/images/test_multi_plane__command.png" width="900" height="30"/>

# Citations
Ahmed, B, (2019). Deep learning in medical imaging: How to automate the detection of knee injuries in MRI exams ?, URL: https://github.com/ahmedbesbes/mrnet


