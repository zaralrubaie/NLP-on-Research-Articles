# Multi-label Classification of Academic Papers
This project uses machine learning (Logistic Regression with OneVsRestClassifier) to classify academic papers into multiple categories based on their titles and abstracts.

## Problem: 
Given a dataset of academic papers with `TITLE` and `ABSTRACT`, the goal is to predict the presence of one or more of the following labels:

- Computer Science  
- Physics  
- Mathematics  
- Statistics  
- Quantitative Biology  
- Quantitative Finance  

This is a **multi-label classification** problem.

## Dataset:
The dataset contains:
- `TITLE`: Title of the paper
- `ABSTRACT`: Abstract text
- `6 Binary Labels`: One column for each category

## Approach
- Combined `TITLE` and `ABSTRACT` text.
- Applied **TF-IDF vectorization**.
- rained a **Logistic Regression model** using `OneVsRestClassifier`.
- Used **cross-validation** to evaluate model generalization.
- Saved predictions on the test set into a CSV file.

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


