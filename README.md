# Final-Year-Project-Muhammad-Hammad-Siddique

# **A Hybrid Machine Learning Approach for Multilingual Phishing Email Detection**

---

# Overview
This notebook develops and evaluates machine learning and deep learning models for **malicious URL classification**. 
The project preprocesses raw URLs to extract features such as URL length, HTTPS presence, number of dots, and IP address usage. 
It applies character-level TF-IDF vectorization as well as deep learning approaches (Embedding + Conv1D + Dense layers) 
to distinguish between benign and malicious URLs. Models including Logistic Regression, Gradient Boosting, and neural networks 
are trained and assessed using multiple metrics and visualizations.

---

# Dataset
- **Path/URL:** https://www.kaggle.com/datasets/subhajournal/phishingemails,  
- **Target column:** `type` (Benign vs Malicious)  
- **Feature column(s):** `url` and engineered features such as url_length, has_https, num_dots, has_ip  
- 

---

# Features & Preprocessing
- Convert URLs to lowercase (`df['url'] = df['url'].str.lower()`)  
- Feature engineering:  
  - URL length  
  - Presence of HTTPS  
  - Number of dots in URL  
  - Presence of IP address  
- TfidfVectorizer(analyzer='char', ngram_range=(1, 5)) for character-level embeddings  
- Label encoding using LabelBinarizer for target variable  
- Visualizations for feature distributions (URL length, HTTPS presence, IP address presence)  

---

# Models
- LogisticRegression with params: max_iter=1000, random_state=42  
- GradientBoostingClassifier with params: n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42  
- Deep Learning (Keras Sequential API):  
  - Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_chars)  
  - Conv1D(128, kernel_size=5, activation='relu')  
  - Dense(10, activation='relu')  
  - Dense(len(df['type'].unique()), activation='softmax')  

---

# Evaluation
- **Metrics:** accuracy_score, auc, classification_report, confusion_matrix, roc_auc_score, roc_curve  
- **Visualizations:**  
  - ROC curve  
  - Confusion matrix heatmap  
  - Matplotlib plots  
  - Seaborn plots  

---

# Environment & Requirements
- **Libraries:** matplotlib, numpy, pandas, seaborn, sklearn, tensorflow, time  
- **Install example:**
  ```bash
  pip install matplotlib numpy pandas scikit-learn seaborn tensorflow
  ```

---

# How to Run
1. Open the notebook in Jupyter/Colab.  
2. Ensure the dataset path variable (`file_path`) points to the URL dataset CSV file.  
3. Run all cells in sequence to preprocess URLs, train models, and generate results.  

---

# Methodology (What the Notebook Does)
1. Load the URL dataset into a pandas DataFrame.  
2. Preprocess the `url` column by converting to lowercase and engineering new features (length, HTTPS presence, number of dots, IP detection).  
3. Encode the target variable (`type`) using LabelBinarizer.  
4. Generate vector representations of URLs with character-level TF-IDF and Embedding layers.  
5. Split the dataset into training and testing sets using `train_test_split` (test_size=0.2, random_state=42, stratify=y).  
6. Train baseline ML models (Logistic Regression, Gradient Boosting) on TF-IDF features.  
7. Train a deep learning CNN-based model (Embedding + Conv1D + Dense layers) on character-level URL sequences.  
8. Evaluate models using classification_report, accuracy, ROC-AUC, and confusion matrices.  
9. Visualize feature distributions, model performance, ROC curves, and confusion matrix heatmaps.  

---

# Outputs You’ll See
- URL feature distribution plots (length, HTTPS presence, IP presence)  
- Classification reports for Logistic Regression, Gradient Boosting, and CNN models  
- Confusion matrix heatmaps  
- ROC curves with AUC values  
- Accuracy and performance metrics printed in the console  

---

# Reproducibility Notes
- train_test_split test_size=0.2  
- train_test_split random_state=42  
- stratify=y applied to preserve class balance  
- random_state=42 consistently used across models  

---

# Repository Tip
Maintain a CHANGELOG and tag notebook versions to track changes in preprocessing, model architectures, and evaluation results.

---

# Attribution
- **Dataset:** https://www.kaggle.com/datasets/subhajournal/phishingemails,   
- **Task:** Malicious URL classification (Benign vs Malicious URLs)  


---


