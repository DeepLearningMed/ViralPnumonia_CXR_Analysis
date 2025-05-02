# Differentiation of COVID-19 from other types of viral pneumonia and severity scoring on baseline chest radiographs: comparison of deep learning with multi-reader evaluation 
This repository hosts the code for a comprehensive deep learning-based analysis of chest X-rays (CXR) for detecting COVID-19 baseline CXRs from other viral pneumonia and normal cases, and predicting the severity score of COVID-19 CXRs. 
## Research information: Codes/material supporting the manuscript titled "Differentiation of COVID-19 from other types of viral pneumonia and severity scoring on baseline chest radiographs: comparison of deep learning with multi-reader evaluation", Revisions submitted to PLOS ONE, April 2025.
* Authors: Nastaran Enshaei1, Arash Mohammadi1, Farnoosh Naderkhani1, Nick Daneman2, Rawan Abu Mughli3, Reut Anconina3, Ferco H. Berger3, Robert Andrew Kozak4, Samira Mubareka5, Ana Maria Villanueva Campos3, Keshav Narang3, Thayalasuthan Vivekanandan3, Adrienne Kit Chan2, Philip Lam2, Nisha Andany2, Anastasia Oikonomou3*
* 1	Concordia Institute for Information Systems Engineering, Concordia University, Montreal, QC, Canada
* 2	Division of Infectious Diseases, Department of Medicine, Sunnybrook Health Sciences Centre, University of Toronto
* 3	Department of Medical Imaging, Sunnybrook Health Sciences Centre, University of Toronto
* 4	Biological Sciences Platform, Sunnybrook Research Institute and Shared Hospital Laboratory
* 5	Department of Microbiology, Sunnybrook Health Sciences Centre, University of Toronto

## Code

The `DLSystemPipeline.py` script is used to:
- Load and preprocess the data  
- Build the network architecture or load trained models  
- Perform predictions and evaluate the results  

The `code_CXR_Analysis.py` script contains the training and development pipeline, including:
- A multi-task learning model for thorax region segmentation on chest X-rays (CXRs)  
- Transfer learning using COVID-19 lesion masks:  
  - Pre-training on a public COVID-19 lesion segmentation dataset  
  - Pre-training on an in-house COVID-19 lesion segmentation dataset  
- Utility functions for data splitting and model training

## 🔗 Access Trained Models

The trained models developed as part of this project are available for public download:

📁 [**Download Thorax Segmentation Model**](https://liveconcordia-my.sharepoint.com/:f:/g/personal/n_enshae_live_concordia_ca/Ep9fAZ0hDzJMgvNqXBsIbXkBTWQ7EjvczA7X16Gl68AGnw?e=DIBIzp)

📁 [**Download Classification Models**](https://doi.org/10.6084/m9.figshare.28892216)

📁 [**Download Severity Scoring Model**](https://liveconcordia-my.sharepoint.com/:f:/g/personal/n_enshae_live_concordia_ca/Ep9fAZ0hDzJMgvNqXBsIbXkBTWQ7EjvczA7X16Gl68AGnw?e=DIBIzp)

These models are provided for **research and reproducibility purposes**.  

## External Validation of Diagnosis Module

- The external dataset used for the **first-step classification** is the publicly available **COVIDGR** dataset introduced by:

  > Tabik, S., et al., 2020. COVIDGR dataset and COVID-SDNet methodology for predicting COVID-19 based on chest X-ray images. *IEEE Journal of Biomedical and Health Informatics*, 24(12), pp.3595–3605.

- The dataset used for the **validation of the full diagnosis pipeline** is a **subset** of a public dataset introduced by the following works.  
  To see the exact subset used in our validation, please refer to `ExternalValidationData_Classification.csv`.

  > Chowdhury, M.E.H., Rahman, T., Khandakar, A., Mazhar, R., Kadir, M.A., Mahbub, Z.B., Islam, K.R., Khan, M.S., Iqbal, A., Al-Emadi, N., Reaz, M.B.I., and Islam, M.T., 2020. *Can AI help in screening viral and COVID-19 pneumonia?* *IEEE Access*, 8, pp.132665–132676.

  > Rahman, T., Khandakar, A., Qiblawey, Y., Tahir, A., Kiranyaz, S., Kashem, S.B.A., Islam, M.T., Maadeed, S.A., Zughaier, S.M., Khan, M.S., and Chowdhury, M.E., 2020. *Exploring the effect of image enhancement techniques on COVID-19 detection using chest X-ray images*. *arXiv preprint* arXiv:2012.02238.

## External Validation of Severity Scoring Module
* The COVID-19 CXRs used for the external validation of the severity scoring module are a subset of a public dataset intorduced by:
  * Wang, L., Lin, Z.Q. and Wong, A., 2020. Covid-net: A tailored deep convolutional neural network design for detection of covid-19 cases from chest x-ray images. Scientific reports, 10(1), p.19549.
* Severity scores of  these 86 COVID-19 CXRs evaluated by one of our radiologists available in RadilologistEvaluatedSeverityScores.xlsx
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
