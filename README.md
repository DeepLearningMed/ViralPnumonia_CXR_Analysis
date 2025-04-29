# Differentiation of COVID-19 from other types of viral pneumonia and severity scoring on baseline chest radiographs: comparison of deep learning with multi-reader evaluation 
This repository hosts the code for a comprehensive deep learning-based analysis of chest X-rays (CXR) for detecting COVID-19 baseline CXRs from other viral pneumonia and normal cases, and predicting the severity score of COVID-19 CXRs. 
## Research information: Codes/material supporting the manuscript titled "Differentiation of COVID-19 from other types of viral pneumonia and severity scoring on baseline chest radiographs: comparison of deep learning with multi-reader evaluation", Revisions submitted to PLOS ONE, April 2025.
* Authors: Nastaran Enshaei1, Arash Mohammadi1, Farnoosh Naderkhani1, Nick Daneman2, Rawan Abu Mughli3, Reut Anconina3, Ferco H. Berger3, Robert Andrew Kozak4, Samira Mubareka5, Ana Maria Villanueva Campos3, Keshav Narang3, Thayalasuthan Vivekanandan3, Adrienne Kit Chan2, Philip Lam2, Nisha Andany2, Anastasia Oikonomou3*
* 1	Concordia Institute for Information Systems Engineering, Concordia University, Montreal, QC, Canada
* 2	Division of Infectious Diseases, Department of Medicine, Sunnybrook Health Sciences Centre, University of Toronto
* 3	Department of Medical Imaging, Sunnybrook Health Sciences Centre, University of Toronto
* 4	Biological Sciences Platform, Sunnybrook Research Institute and Shared Hospital Laboratory
* 5	Department of Microbiology, Sunnybrook Health Sciences Centre, University of Toronto

## Main parts
The code includes:
* A multi-task learning model for segmenting thorax regions from CXRs.
* Transfer learning using COVID-19 lesion masks: 
  * Pre-training on a public COVID-19 lesion sementation dataset
  * Pre-training on an in-house COVID-19 lesion sementation dataset
* Diagnosis model
* Severity score prediction model
* Utility functions, data split, and training
* DLSystemPipeline.py is used to build the network architecture and load the pre-trained weights.
* A download link for the trained model weights will be added here shortl. 

## Severity scores of  86 COVID-19 CXRs evaluated by one of our radiologists available in RadilologistEvaluatedSeverityScores.xlsx
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
* The COVID-19 external CXRs evaluated by one are radilogists are a subset of a public dataset intorduced by:
  * Wang, L., Lin, Z.Q. and Wong, A., 2020. Covid-net: A tailored deep convolutional neural network design for detection of covid-19 cases from chest x-ray images. Scientific reports, 10(1), p.19549.
