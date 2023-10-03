# Credit Card Fraud Detecting using Machine Learning
Machine learning classification program to determine if a credit card transaction was fraudulent.

## Overview
This machine learning project aims to develop a model that can predict whether a credit card transaction is real or fraudulent.

## The Databse
The authors of this database gave the following context for the data:  
> The dataset contains transactions made by credit cards in September 2013 by European cardholders.
This dataset presents transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions. The dataset is highly unbalanced, the positive class (frauds) account for 0.172% of all transactions.  

> It contains only numerical input variables which are the result of a PCA transformation. Unfortunately, due to confidentiality issues, we cannot provide the original features and more background information about the data. Features V1, V2, … V28 are the principal components obtained with PCA, the only features which have not been transformed with PCA are 'Time' and 'Amount'. Feature 'Time' contains the seconds elapsed between each transaction and the first transaction in the dataset. The feature 'Amount' is the transaction Amount, this feature can be used for example-dependant cost-sensitive learning. Feature 'Class' is the response variable and it takes value 1 in case of fraud and 0 otherwise.  

> Given the class imbalance ratio, we recommend measuring the accuracy using the Area Under the Precision-Recall Curve (AUPRC). Confusion matrix accuracy is not meaningful for unbalanced classification. 

## Using this Project
Please follow these steps to use this project:  

1. Clone this repository to your machine  
`git clone https://github.com/joelhenningsen/credit-card-fraud-detection.git`
2. Go to the project directory  
`cd credit-card-fraud-detection`
3. Install all required dependencies by using this command:  
`pip install -r requirements.txt`

## Data Preprocessing, Building, and Training Models
The **src/data_preprocessing.py** module contains code for loading and preprocessing the dataset. The data is split into training and testing sets, and features are standardized for model training.  

The **src/model_training.py** module contains code for training a machine learning model on the preprocessed data. Various algorithms and techniques, including neural networks, tree ensembles, and regressions, can be explored in this module.

The **notebooks/evaluation.ipynb** Jupyter Notebook provides a detailed analysis of the trained model's performance. It includes visualizations, metrics, and insights into the model's ability to detect fraudulent transactions.

## Conclusion
This project serves as a practical demonstration of building a credit card fraud detection model using machine learning techniques. By exploring different algorithms and evaluation metrics, you can gain insights into the challenges of dealing with imbalanced datasets and the importance of fraud detection for for banks and companies.

## Acknowledgements
The dataset used for this project was downloaded from Kaggle. Thank you to the authors/contributors for the time you put into creating and anonymizing the data.  

Dataset: [kaggle.com/datasets/mlg-ulb/creditcardfraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

This project was partially inspired from this project idea on GeeksforGeeks: [www.geeksforgeeks.org/ml-credit-card-fraud-detection/](https://www.geeksforgeeks.org/ml-credit-card-fraud-detection/)

## Author
* Joel Henningsen
* LinkedIn: [linkedin.com/in/joel-henningsen](https://www.linkedin.com/in/joel-henningsen/)