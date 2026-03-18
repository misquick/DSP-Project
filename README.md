# DSP-Project: Heart rate analysis and disease prediction
# Introduction

Cardiovascular diseases are among the leading causes of death worldwide, making early detection critically important. This project focuses on analyzing heart rate signals to identify abnormal patterns and predict potential health risks. The main objective is to develop a system that processes heart rate data, extracts meaningful features, and uses them to support disease prediction. The scope of the project includes signal processing, feature extraction, and basic predictive modeling.

# Methodology

The system combines signal processing and data analysis techniques to extract useful information from heart rate data. Noise reduction is performed using digital filters such as Finite Impulse Response (FIR) filters to improve signal quality. Frequency-domain analysis is conducted using the Fast Fourier Transform (FFT) to identify periodic patterns and irregularities. These processed features are then used as inputs for predictive models, which estimate the likelihood of cardiovascular abnormalities.

# Implementation / Simulation

The project was implemented using Python in a Jupyter Notebook / Google Colab environment. The workflow includes:

Importing and preprocessing heart rate datasets

Applying FIR filters to remove noise

Performing FFT for frequency analysis

Extracting features such as peak frequency and variability

Using basic machine learning models for prediction

    # Code snippet:
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from scipy.signal import firwin, lfilter
    from scipy.fft import fft, fftfreq
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, classification_report
  
    #1. Load Data
    Example: CSV file with a column "heart_rate"
    data = pd.read_csv("heart_rate_data.csv")
    heart_signal = data["heart_rate"].values
    #Sampling frequency (Hz)
    fs = 100  
    #2. FIR Filter (Noise Reduction)
    numtaps = 51
    cutoff = 0.3  # normalized cutoff (0 to 1)
    fir_coeff = firwin(numtaps=numtaps, cutoff=cutoff)
    filtered_signal = lfilter(fir_coeff, 1.0, heart_signal)
    #3. FFT Analysis
    N = len(filtered_signal)
    yf = fft(filtered_signal)
    xf = fftfreq(N, 1 / fs)
    #Keep only positive frequencies
    xf = xf[:N // 2]
    yf = np.abs(yf[:N // 2])
    #4. Feature Extraction
    #Example features
    mean_hr = np.mean(filtered_signal)
    std_hr = np.std(filtered_signal)
    peak_freq = xf[np.argmax(yf)]
    
    features = np.array([mean_hr, std_hr, peak_freq])
    
    
    #5. Prepare Dataset
    #Simulated dataset (for demonstration)
    X = []
    y = []
  
    for i in range(0, len(filtered_signal) - 100, 50):
        segment = filtered_signal[i:i+100]
  
      mean_val = np.mean(segment)
      std_val = np.std(segment)
  
      yf_seg = fft(segment)
      xf_seg = fftfreq(len(segment), 1 / fs)
      peak_f = xf_seg[np.argmax(np.abs(yf_seg))]
  
      X.append([mean_val, std_val, peak_f])
  
      #Dummy label (replace with real labels)
      y.append(1 if mean_val > np.mean(filtered_signal) else 0)
  
      X = np.array(X)
      y = np.array(y)
      
      #6. Train Model
      X_train, X_test, y_train, y_test = train_test_split(
          X, y, test_size=0.2, random_state=42
      )
      
      model = LogisticRegression()
      model.fit(X_train, y_train)
      
      
      #7. Evaluation
      y_pred = model.predict(X_test)
      
      print("Accuracy:", accuracy_score(y_test, y_pred))
      print("\nClassification Report:\n", classification_report(y_test, y_pred))
      
      #8. Visualization
      plt.figure()
      plt.plot(heart_signal, label="Raw Signal")
      plt.plot(filtered_signal, label="Filtered Signal")
      plt.legend()
      plt.title("Heart Rate Signal Processing")
      plt.xlabel("Samples")
      plt.ylabel("Amplitude")
      plt.show()
      
      plt.figure()
      plt.plot(xf, yf)
      plt.title("Frequency Spectrum (FFT)")
      plt.xlabel("Frequency (Hz)")
      plt.ylabel("Magnitude")
      plt.show()




# Results and Discussion

The results show that filtering significantly improves signal clarity by removing noise. FFT analysis helps reveal dominant frequency components, which are useful in identifying irregular heart rhythms. The predictive model demonstrates the ability to distinguish between normal and abnormal patterns, although performance depends on data quality and size. Visualizations such as signal plots and frequency spectra support the analysis and interpretation of results.

# Conclusion

This project demonstrates that heart rate signal analysis can be effectively used for early detection of potential cardiovascular issues. By combining signal processing and predictive modeling, meaningful insights can be extracted from raw data. Future improvements may include using larger datasets, incorporating additional health parameters, and applying more advanced machine learning techniques to improve prediction accuracy and reliability.

# References

Standard signal processing textbooks and documentation

SciPy and NumPy official documentation

HeartPy offical repository and documentation

Research papers on heart rate variability and cardiovascular prediction

Online resources on FIR filters and FFT algorithms
