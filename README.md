# Person-Recognition

## Overview
This project implements user handwriting classification based on sensor data preprocessing and machine learning classifiers. The data is read from a pickle file, features are extracted, preprocessed, and then classified using multiple classifiers. (Logistic Regression, k-Nearest Neighbors (k-NN), Random Forest, Gradient Boosting, Support Vector Machines (SVMs), and Multilayer Perceptron (Neural Network))

## Project Structure
* ```data_manipulation.py```: Preprocesses raw data, extracts features, scales them, and splits it into training and testing datasets.
*  ```logistic_regression.py```: Trains and evaluates Logistic Regression classifier.
*  ```kNN_classification.py```: Trains and evaluates kNN classifier.
*  ```SVM_classification.py```: Trains and evaluates SVM classifier.
*  ```random_forest_classification.py```: Trains and evaluates Random Forest classifier.
*  ```gradient_boosting_classifier.py```: Trains and evaluates Gradient Boosting classifier.
*  ```multilayer_perceptron.py```: Trains and evaluates Neural Network classifier.
*  ```dataset.pickle```: The input dataset in pickle format

## Usage
Install needed dependencies (Python, pandas, numpy, scikit-learn, matplotlib)

### 1. Prepare Dataset
Place your dataset pickle file named ```dataset.pickle``` in the root directory of the project. The file should contain the dictionary structure expected by ```data_manipulation.py```.
### 2. Data Preprocessing
Run the preprocessing script to load the data, extract features, scale, and split into train/test sets.
```bash
python data_manipulation.py
```
This script creates variables ```X_train```, ```X_test```, ```y_train```, ```y_test``` for use in your classifiers.
### 3. Run Classifiers
You can run any of the classifier scripts:
```bash
python gradient_boosting_classifier.py
```
This script will:
* Train the model on the training data
* Evaluate performance on the test set with Accuracy, F1, Recall, and ROC AUC scores
* Display a confusion matrix plot and ROC curves
* Show the most frequent misclassifications and feature importance
