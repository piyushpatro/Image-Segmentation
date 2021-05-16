# Image-Segmentation

This is a image segmentation project that is used to segment salt from rocks from the images obtained using Seismology.
The dataset can be obtained from https://www.kaggle.com/c/tgs-salt-identification-challenge/data

Model used- UNET with 1 input and 1 output channel along with filter config of [64, 128, 256, 512]
Train the model once and save it the your local machine and then laod its state dictionary in the main.py file


The api(main.py) created using Flask can be used for displaying the segmented image.
