# NLP-Tweet-Classification-for-Disaster-Response
This project aims to classify tweets into disaster and non-disaster categories. Utilizing a dataset from Kaggle's "Natural Language Processing with Disaster Tweets" competition, the goal is to predict whether a tweet relates to a real disaster or not. This classification has significant implications for emergency response and news reporting.

# Dataset

The dataset, comprising 10,000 tweets, is sourced from the Kaggle Competition. Each tweet is labeled as either about a real disaster or not.

# Methodology

The project workflow is as follows:

**Data Preprocessing**: 
The tweets are loaded and preprocessed using Python and the spaCy library. This includes lemmatization to reduce words to their base forms.

**Feature Extraction**: Text data is transformed into a numerical format using the TF-IDF vectorization technique.
Model Selection: Different machine learning models, including Ridge Regression, Logistic Regression, and Support Vector Machine (SVM), were evaluated. The SVM model, yielding the highest F1 score, was chosen for the final classification task.

**Model Evaluation**: Model performance was assessed using the F1 score, a measure of a test's accuracy.
Pipeline Implementation: A pipeline combining TF-IDF vectorization and the SVM model was developed for efficient model training and prediction.

# Results
The SVM model achieved an F1 score of 0.8014, demonstrating its efficacy in accurately categorizing tweets.

# Technologies
Python
Libraries: Pandas, NumPy, Scikit-learn, spaCy
Matplotlib (for exploratory data analysis)

# Future Directions
Explore advanced NLP techniques like word embeddings and deep learning models.
Conduct extensive hyperparameter tuning for model optimization.
Investigate the application of the model in a real-time tweet analysis system.
