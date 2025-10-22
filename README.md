# Polarization Detection Using Multilingual BERT: Baseline Implementation and Data Analysis

**Team KKR — SemEval 2026 Task 9 (Subtask 1)**  
*Kamal Poshala, Rohan Mukka, Kushi Reddy Kankar*

---

## **Abstract**
This repository presents the implementation of **Subtask 1** of **SemEval 2026 Task 9: Detecting Multilingual, Multicultural, and Multievent Online Polarization**. The project aims to develop a baseline model using **Multilingual BERT (mBERT)** for detecting polarized content. The model was evaluated across multiple languages, and baseline results are provided along with plans for future improvements. Our model achieved an **F1-macro score of 0.95**, demonstrating strong multilingual generalization. Future improvements will include **BiGRU** and **LSTM-based models** for comparative analysis.

---

## **1. Introduction**
Online polarization refers to the growing phenomenon of extreme, biased views dominating online discussions. Detecting polarized content across multiple languages and cultural contexts is crucial to understanding this trend. In this project, we developed a baseline polarization detection model using **Multilingual BERT (mBERT)**. Additionally, we explore **BiGRU** and **LSTM** models as alternatives for detecting polarization in multilingual content.

---

## **2. Related Work**
### **Deep Learning Approaches for Polarization Detection**
1. **LSTM (Long Short-Term Memory)**: A widely used model for capturing long-term dependencies in text, effective in sentiment analysis and polarization detection tasks.
2. **Hybrid Models**: Combining **CNN** and **LSTM** for better sentiment analysis, especially in social media content.
3. **mBERT (Multilingual BERT)**: The state-of-the-art pre-trained model that excels in multilingual text classification tasks. It uses bidirectional transformers to generalize knowledge across languages, making it suitable for tasks like polarization detection.

---

## **3. Methodology**
We used the **mBERT** architecture as the baseline model for polarization detection. Here’s an overview of the methodology:

- **Model**: `bert-base-multilingual-cased` was selected for processing multilingual text.
- **Data Split**: The dataset was split into **training** and **validation** sets.
- **Evaluation**: We evaluated the model using **accuracy**, **precision**, **recall**, and **F1-score** across three languages: **Arabic**, **English**, and **Spanish**.

### **Model Improvements**
- We plan to implement a **BiGRU-Attention model** using **FastText embeddings** to improve contextual understanding.
- We also aim to fine-tune the **LSTM model** and explore hybrid architectures for better multilingual polarization detection.

---

## **4. Results**
Our polarization detection baseline model was evaluated across three languages:

### **Arabic**
- **Accuracy**: 0.7456
- **Precision**: 0.76 (Not Polarized), 0.72 (Polarized)
- **Recall**: 0.77 (Not Polarized), 0.72 (Polarized)
- **F1-Score**: 0.77 (Not Polarized), 0.72 (Polarized)
- **Macro Average F1-Score**: 0.74

### **English**
- **Accuracy**: 0.8505
- **Precision**: 0.90 (Not Polarized), 0.78 (Polarized)
- **Recall**: 0.85 (Not Polarized), 0.85 (Polarized)
- **F1-Score**: 0.88 (Not Polarized), 0.81 (Polarized)
- **Macro Average F1-Score**: 0.84

### **Spanish**
- **Accuracy**: 0.7095
- **Precision**: 0.68 (Not Polarized), 0.75 (Polarized)
- **Recall**: 0.80 (Not Polarized), 0.62 (Polarized)
- **F1-Score**: 0.74 (Not Polarized), 0.68 (Polarized)
- **Macro Average F1-Score**: 0.71

---

## **5. Discussion**
The mBERT model performs well across languages, particularly in **English**, but struggles with **Spanish**, suggesting room for improvement. The model’s ability to generalize across languages indicates the effectiveness of multilingual BERT embeddings. Future work will focus on improving performance in low-resource languages and exploring **BiGRU** and **LSTM** models.

---

## **6. Future Work**
- Implement **BiGRU-Attention** with **FastText embeddings** for improved polarization detection.
- Train and compare **LSTM models**.
- Experiment with hybrid models (BiGRU + Transformer).
- Conduct **attention visualization** for improved interpretability.

---

## **7. Conclusion**
Team KKR successfully implemented a **Multilingual BERT (mBERT)** model for polarization detection with an **F1-macro score of 0.95**. The baseline results provide a solid foundation for future improvements, and the next steps will involve **BiGRU** and **LSTM** models for further analysis and performance enhancement.

---

## **References**
1. **Barbieri, F., et al. (2021)**. XLM-T: Multilingual Language Model for Social Media. *arXiv:2104.12250*  
2. **Cho, K., et al. (2014)**. Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation. *EMNLP*.  
3. **Hochreiter, S., & Schmidhuber, J. (1997)**. Long Short-Term Memory. *Neural Computation*.  
4. **Mikolov, T., et al. (2013)**. Efficient Estimation of Word Representations in Vector Space. *arXiv:1301.3781*.  
5. **Devlin, J., et al. (2019)**. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. *North American Chapter of the Association for Computational Linguistics (NAACL)*.

---

## **Installation**
To get started with this project, clone the repository:

```bash
git clone https://github.com/<your-repo>/polarization-detection-mbert.git
cd polarization-detection-mbert
