# 🎙️ Grammar Scoring Engine from Voice Samples

## 📌 Overview

This project presents an end-to-end machine learning system that predicts grammar proficiency scores from spoken audio. The model combines acoustic speech features with linguistic features derived from automatic transcription to estimate grammar scores on a scale of 0–5.

The objective is to evaluate spoken grammar quality by leveraging both signal processing and natural language analysis techniques.

---

## 🧠 Methodology

### 🔊 Feature Engineering (99 Features per Sample)

**Acoustic Features (89)**  
Extracted using `librosa`:
- MFCCs and Delta MFCCs (mean & standard deviation)
- Pitch statistics
- RMS energy
- Zero Crossing Rate
- Spectral centroid, bandwidth, rolloff, contrast
- Chroma features
- Tempo
- Duration
- Silence ratio

**Linguistic Features (10)**  
Audio was transcribed using the `faster-whisper` tiny model. From the transcript, the following features were computed:
- Word count
- Average word length
- Unique words
- Vocabulary richness
- Sentence count
- Average sentence length
- Filler word count
- Capitalization errors
- Long word ratio

---

## 🤖 Model Selection

The following regression models were evaluated:

- RandomForest  
- XGBoost  
- LightGBM  
- Gradient Boosting  

RandomForest achieved the best validation performance and was selected as the final model.

---

## 📊 Results

- Training Samples: 409  
- Test Samples: 197  
- Validation RMSE: **0.6041**  
- Validation Pearson Correlation: **0.6304**

The final model was retrained on the full dataset and used to generate predictions for the test set.

---

## 🛠️ Tech Stack

- Python  
- librosa  
- faster-whisper  
- scikit-learn  
- matplotlib  
- seaborn  

---

## 📁 Repository Structure
