# DSP-Project
Heart rate analysis and disease prediction
#Introduction

Cardiovascular diseases are among the leading causes of death worldwide, making early detection critically important. This project focuses on analyzing heart rate signals to identify abnormal patterns and predict potential health risks. The main objective is to develop a system that processes heart rate data, extracts meaningful features, and uses them to support disease prediction. The scope of the project includes signal processing, feature extraction, and basic predictive modeling.

Methodology

The system combines signal processing and data analysis techniques to extract useful information from heart rate data. Noise reduction is performed using digital filters such as Finite Impulse Response (FIR) filters to improve signal quality. Frequency-domain analysis is conducted using the Fast Fourier Transform (FFT) to identify periodic patterns and irregularities. These processed features are then used as inputs for predictive models, which estimate the likelihood of cardiovascular abnormalities.

Implementation / Simulation

The project was implemented using Python in a Jupyter Notebook / Google Colab environment. The workflow includes:

Importing and preprocessing heart rate datasets

Applying FIR filters to remove noise

Performing FFT for frequency analysis

Extracting features such as peak frequency and variability

Using basic machine learning models for prediction
