# COVID Patient ICU Prediction Project

## Introduction

This data science project aims to predict whether a COVID-19 patient will require admission to the Intensive Care Unit (ICU) based on a given dataset. The project leverages several machine learning algorithms to make predictions, and evaluates their performance using metrics such as accuracy, confusion matrix, precision, recall, and F1 score.

## Machine Learning Algorithms Utilized

- **RandomForestClassifier**: A versatile ensemble learning method based on decision trees.
- **Support Vector Classifier (SVC)**: A classifier that seeks to find the optimal hyperplane that best separates different classes.
- **KNeighborsClassifier**: A simple and effective classification algorithm based on the nearest neighbors.
- **RadiusNeighborsClassifier**: A variation of K-nearest neighbors that considers only neighbors within a fixed radius.
- **GradientBoostingClassifier**: An ensemble learning technique that builds a strong classifier by combining multiple weaker ones.
- **AdaBoostClassifier**: Another ensemble method that focuses on improving the performance of weak classifiers.
- **XGBClassifier (XGBoost)**: An optimized gradient boosting library that delivers high-performance machine learning models.

## Evaluation Metrics

The performance of each machine learning algorithm is assessed using the following metrics:

- **Accuracy**: Measures the overall correctness of predictions.
- **Confusion Matrix**: Provides a breakdown of true positives, true negatives, false positives, and false negatives.
- **Precision**: Evaluates the accuracy of positive predictions.
- **Recall**: Measures the ability of the model to identify positive instances.
- **F1 Score**: Harmonic mean of precision and recall, providing a balance between them.

## Getting Started

To run this project and replicate the results:

1. Clone this repository to your local machine.

git clone https://github.com/anauhwar/Data-Science-Covid-ICU-Prediction.git
cd COVID-ICU-Prediction


2. Install the required Python packages by creating a virtual environment and using pip:

python -m venv venv
source venv/bin/activate (or venv\Scripts\activate on Windows)
pip install -r requirements.txt


3. Execute the Jupyter Notebook or Python script to train and evaluate the machine learning models.

4. Analyze the results and explore the predictive capabilities of each algorithm.

## Conclusion

This project provides insights into predicting ICU admissions for COVID-19 patients using various machine learning techniques. By comparing the performance metrics of different algorithms, we can determine which one offers the best predictive accuracy and potential for real-world application.
