# ViralPneumonia_CXR_Analysis
This repository hosts the code for a comprehensive deep learning-based analysis of chest X-rays (CXR) for detecting COVID-19 baseline CXRs from other viral pneumonia and normal cases, and predicting the severity score of COVID-19 CXRs.
## Main parts
The code includes:
* A multi-task learning model for segmenting thorax regions from CXRs.
* Transfer learning using COVID-19 lesion masks: 
  * Pre-training on a public COVID-19 lesion sementation dataset
  * Pre-training on an in-house COVID-19 lesion sementation dataset
* Diagnosis model
* Severity score prediction model
* Utility functions, data split, and training

## Severity scores of  86 COVID-19 CXRs evaluated by one of our radiologists
* Scoring Method Description: The scoring method, as explained in detail in the manuscript, involves:
    *  Evaluation of four lung quadrants based on the extent of involvement and the density of opacifications.
    *  Calculation of the total severity score for the CXR by summing the quadrants' scores.
* Scoring Table Description: The scoring table includes the following columns:
    *  ID: CXR ID extracted from the original file names.
    *  RULZ: Right Upper Lung Zone
    *  RLLZ: Right Lower Lung Zone
    *  LULZ: Left Upper Lung Zone
    *  LLLZ: Left Lower Lung Zone
    *  Total: Total severity score of CXRs

## Prerequisites and References
* Required libreries have been listed in the code.
* For the installation of the segmentation model pre-trained on ImageNet, please refer to https://github.com/qubvel/segmentation_models.
* The public COVID-19 lesion segmentation dataset used for the transfer learning:
  * Degerli A, Ahishali M, Yamac M, Kiranyaz S, Chowdhury ME, Hameed K, et al. COVID-19 infection map generation and detection from chest X-ray images. Health information science and systems. 2021 Apr 1;9(1):15.

