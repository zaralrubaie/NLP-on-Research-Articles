# Multi-label Classification of Academic Papers
This project uses machine learning (Logistic Regression with OneVsRestClassifier) to classify academic papers into multiple categories based on their titles and abstracts.

## Problem
Given a dataset of academic papers with TITLE and ABSTRACT, the goal is to predict one or more of the following categories:
- Computer Science
- Physics
- Mathematics
- Statistics
- Quantitative Biology
- Quantitative Finance
## Dataset:
The dataset contains:
- `TITLE`: Title of the paper
- `ABSTRACT`: Abstract text
- `6 Binary Labels`: One column for each category

## Approach
- Combined TITLE and ABSTRACT into a single text feature.
- Applied TF-IDF vectorization to convert text into numeric features.
- Trained a Logistic Regression model using OneVsRestClassifier for multi-label classification.
- Evaluated model using cross-validation.
- Saved predictions on the test set to a CSV file.

## 🧪 Evaluation
- Accuracy on train set: `0.71`
- Accuracy on validation set: ` 0.63`
- Mean cross-validation accuracy: `0.63%`
  
## Challenges & Notes:
- Slight Overfitting:
The model showed higher accuracy on the training set compared to the validation set, indicating some overfitting. Techniques like regularization tuning, dimensionality reduction, or trying other models.
- Limited Computational Power:
Tokenizing and lemmatizing large text fields like titles and abstracts using spaCy was computationally expensive. On a more powerful machine or with GPU acceleration, preprocessing time can be significantly reduced.
- Class Imbalance:
Some categories (e.g., Quantitative Finance) had far fewer examples compared to others (e.g., Computer Science). This could impact prediction quality. Approaches like SMOTE, class weights, or collecting more balanced data could help.
- Multi-label Complexity:
Since each paper can belong to multiple categories, careful evaluation was necessary. The model was trained using One-vs-Rest classification and evaluated using cross-validation to ensure robustness.

##  Project Structure
```
nlp-on-research-articles/
│
├─ README.md
├─ nlp-on-research-articles.ipynb    # Jupyter notebook with preprocessing, training, and evaluation
├─ models/
│   └─ multilabel_logreg_model.pkl    # Trained model
├─ outputs/
│   └─ test_predictions-NLP.csv       # Test set predictions
├─ requirements.txt                   # Libraries used

````


