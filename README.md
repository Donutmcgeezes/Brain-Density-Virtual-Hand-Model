# Brain-Density-Virtual-Hand-Model

# Overview
This project aims to, 1. Collect sEMG signals with the Quattrocento amplification and filtering device 2. Record camera footage of hand motion from multiple angle to be processed into a 3D reconstruction of a virtual hand 3. Map the sEMG signals to the 3D hand motion data with a Neural Network


https://github.com/user-attachments/assets/ae0c59ec-6c00-4016-9be2-546444edf4b8

# sEMG Data
64 channel sEMG signals were collected with a high density 64 channel electrode connected to a Quattrocento amplification and filtering (10-500Hz) device. The electrode was placed on the right hand, Flexor Carpi Ulnaris, and ASL letters A-E hand motion was used to generate the sEMG signals. OT Biolab+ was the software used to record and save the sEMG signals. The signal was exported as a csv file for labelling. The first column was Time while the next 64 columns correspond to each channel of the electrode.
PCA was implemented on the sEMG data, with the first 3 PCA components accounting for 80-95% of the variance in our data. The values of the 3 components could be added as 3 new columns to the exported sEMG data.

# 3D Motion Capture
A camera system was made to synchronise the recordings from 3 cameras placed at different angles to face the same point. A ChArUco board was used to make calibration recordings with the camera system. This was for intrinsic and extrinsic calibration.
3 cameras recorded synchronised footage of the ASL letters A-E hand motion. Footage from each camera was labelled with 20 key hand nodes using the SLEAP framework. By using the sleap-anipose python package, we calibrated our cameras with the 'calibrate' method which generated a toml file containing the intrinsic (Camera Matrix) and extrinsic parameters (Rotation Matrix & Translation Vector) of each camera. By exporting the 2D coordinates of each hand node for each camera from SLEAP, we triangulated each node in 3D space as shown in the video above.

Note: 3D Motion Capture was synchronised with sEMG data collection with an Arduino trigger.

# Decoding with LSTM Neural Network
Labelling data: Sampling frequency for sEMG (2048Hz) was different to the motion capture rate (~25Hz). We used this mismatch to allow a temporal sequence of sEMG data to correspond to a specific change in 3D coordinates. First, with 3D data with N frames, we take the difference between 2 consecutive frames, leaving N-1 datapoints of the frame to frame change in 3D coordinates. Secondly, we separate the sEMG signal into N-1 chunks and label each chunk with the corresponding change. 

Building the LSTM Neural Network: We utilised PyTorch to build 2 LSTM models with 2 hidden layers and 60 output layers. For raw sEMG signals, we used 64 input layer nodes while with 3 PCA components we used 3 input nodes. The 3D motion data was used as reference values to compute the loss function.
