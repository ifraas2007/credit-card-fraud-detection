# Credit Card Fraud Detection - Task 2

## Objective
Build a machine-learning system that classifies credit-card transactions as:
- 0 = Legitimate
- 1 = Fraud

The project experiments with:
1. Logistic Regression
2. Decision Tree
3. Random Forest

## Dataset
Use the Kaggle "Credit Card Transactions Fraud Detection Dataset" by Kartik Shenoy.

Download the dataset and place these files in this project folder:
- fraudTrain.csv
- fraudTest.csv

The dataset contains transaction information and the target `is_fraud`.

## Folder structure

credit_card_fraud_detection/
│
├── fraud_detection.py
├── requirements.txt
├── fraudTrain.csv
├── fraudTest.csv
└── results/

## Windows / VS Code setup

1. Install Python 3.10+.
2. Open this folder in VS Code.
3. Open Terminal.
4. Run:

   python -m venv venv

5. Activate the environment:

   venv\Scripts\activate

6. Install libraries:

   pip install -r requirements.txt

7. Put `fraudTrain.csv` and `fraudTest.csv` beside `fraud_detection.py`.

8. Run:

   python fraud_detection.py

## What the program does

- Loads the dataset.
- Extracts hour/day/month/day-of-week from the transaction timestamp.
- Removes direct identifiers and high-cardinality personal fields.
- One-hot encodes categorical features.
- Scales numerical features.
- Uses class weighting because fraud is much rarer than legitimate transactions.
- Trains Logistic Regression, Decision Tree and Random Forest.
- Calculates Accuracy, Precision, Recall and F1 Score.
- Creates confusion matrices.
- Creates a model comparison graph.
- Saves the final Random Forest pipeline.

## Important
Do not copy numerical results from someone else's project. Run the program and use the results generated on your computer.

## Demo flow

1. Show the dataset.
2. Show `is_fraud` as the target.
3. Explain preprocessing.
4. Explain the three algorithms.
5. Run the Python program.
6. Show the metric output.
7. Open `results/model_comparison.png`.
8. Open one confusion matrix.
9. Explain that fraud detection is an imbalanced classification problem, so recall, precision and F1 are important in addition to accuracy.
