# Catching Credit Card Fraud

Authors: Adam Blomfield

## Overview

In this **classification project**, we trained **3 machine learning models** to classify credit-card transactions as fraudulent or non-fraudulent.  The **data was resampled with both under and over sampling techniques**, and each model was fitted to each resampled dataset.  See the notebooks directory for data cleaning and predictive modelling code.

## Project Motivation and Metrics

Accurately classifyinig fraudulent transactions is beneficial to both banks and credit-card users.  An undetected fraud will cost the user and possibly the bank for the transaction amount.  Alternatively, if our models mis-classify a non-fraudulent transaction as a fraud, it causes inconvenience to the user, and requires staff at the bank to authorize the transaction.

For these reasons our focus was to find a model that could maximise True Positives (correctly identifying frauds) whilst minimizing False Negatives (fraudulent transactions being missclassified as legitimate).  Minimizing False Positives (classifying legitimate transactions as fraud) was seen as a lower priority, but still valuable. 

The performance metrics used were **F1 Score, Recall, PR Curve AUC and False Negatives**.

## Data Background
The [data set]('https://www.kaggle.com/mlg-ulb/creditcardfraud') consisted of 284,807 European credit card transactions over two days in September 2013.  The **dataset is highly imblanaced** with only 492 (0.17%) fraudulent transactions.  

The data includes the following features: class (fraud or valid), amount, time in seconds elapsed since the first transaction, and 28 other variables that are not identiified for confidentiality reasons. These masked variables have also been transformed using Principal Component Analysis.

## Techniques and Models Used
Different resampling techniques were used to combat the class imbalance and improve machine learning performance.  These were a combination of **SMOTE** (Synthetic Minority Over-sampling Technique) and **Random Undersampling**.   **Class weights** were also used to place more importance on fraud detection (Class 1).

The 3 Machine Learning Classififiers used were: 
1) **Gaussian Naive Bayes**
2) **Random Forest**
3) **XGBoost** (eXtreme Gradient Boosting)

To tune each model's hyperparameters **GridSearch** was used.  

## Findings
The **best performing model and resampling combination overall was using Random Forest with SMOTE** which produced the following performance metrics:

|                      | F1 Score | Recall | PR AUC |   TP   |   FP   |   TN    |   FN   |
|----------------------|----------|--------|--------|--------|--------|---------|--------|
|Random Forest & SMOTE |  0.880   | 0.865  |  0.882 | 0.150% | 0.018% | 99.809% | 0.023% |

The following confusion matrix has the absolute values for the best model combination:
[CM RF SMOTE](visuals/cm_rf_smote.pdf)

The charts below compare F1 and Recall scores across the different combinations.  

    The top scores overall were between Random Forest:Original and Random Forest:SMOTE.  

[F1 Barplot](visuals/barplot_of_f1_scores.pdf)

[Recall Barplot](visuals/barplot_of_recall_scores.pdf)

Further analysis of PR AUC and FN meant that Random Forest:SMOTE had the best performance.
[PR CURVE RF SMOTE](pr_curve_for_Random Forest with SMOTE.pdf)


## Further Improvements
The following are a few suggestions to improve the project:
1) Track overfitting by including train and test scores
2) Tune model to output probabilities, rather than 0,1
    * This would then allow the user to be more cautious and change a threshold from 0.5 to improve Recall
    * Predictions within a certain probability could be redirected to a secondary check, e.g. a human or another model
3) Compare different resampling techniques
    * The Random Undersampling is not suited to datasets that are not uniformly distributed, so the poor performance with our models could be due to too much data loss.



