# ViralPnumonia_CXR_Analysis
This repository hosts the code for a comprehensive deep learning-based analysis of chest X-rays (CXR) for detecting COVID-19 baseline CXRs from other viral pneumonia and normal cases, and predicting the severity score of COVID-19 CXRs.
## Main parts
The code includes:
* A multi-task learning model for segmenting thorax regions from CXRs.
* Transfer learning using COVID-19 lesion masks: 
  * Pre-training on a public COVID-19 lesion sementation dataset
  * Pre-training on an in-house COVID-19 lesion sementation dataset
* Diagnosis model
* Severity score prediction model
* Utility functions, data split, training, and prediction 

## Prerequisites
Required libreries have been listed in the code.
For the installation of the segmentation model pre-trained on ImageNet, please refer to https://github.com/qubvel/segmentation_models.

