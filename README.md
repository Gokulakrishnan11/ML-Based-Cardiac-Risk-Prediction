# ML-Based-Cardiac-Risk-Prediction
A machine learning project that uses Naive Bayes to predict heart disease based on patient data. Includes data preprocessing, model training, evaluation, and manual prediction. Built with Python and scikit-learn for healthcare analytics.

Skip to content
Chat history

You said:
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

## Data Collection

df=pd.read_excel(r"/content/sample_data/heart_disease_data_with_labels.xlsx")
print(df)


##Data preprocessing

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df['Gender']=le.fit_transform(df['Gender'])
df['Chest Pain Type']=le.fit_transform(df['Chest Pain Type'])
df['Fasting Blood Sugar > 120 mg/dl']=le.fit_transform(df['Fasting Blood Sugar > 120 mg/dl'])
df['Resting ECG']=le.fit_transform(df['Resting ECG'])
df['Exercise Induced Angina']=le.fit_transform(df['Exercise Induced Angina'])
df['Slope of Peak Exercise ST Segment']=le.fit_transform(df['Slope of Peak Exercise ST Segment'])
df['Thalassemia (Thal)']=le.fit_transform(df['Thalassemia (Thal)'])

print(df)

## Defining input,Output

x=df.drop(columns=['Heart Disease'])
y=df['Heart Disease'] 
ChatGPT said:
Your code looks good and is mostly correct for loading and preprocessing a heart disease dataset using pandas and LabelEncoder. Below is a quick review, with suggestions to improve readability, robustness, and maintainability:

✅ What your code does well:
Reads an Excel file correctly.

Applies LabelEncoder to multiple categorical columns.

Separates features (x) and target (y) for model training.

🔧 Suggestions for Improvement:
Use consistent variable naming:
Use X and `y

You said:
## 80% of data for train
## 20% of data for test


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=12)

print("df",df.shape)
print("x_train",x_train.shape)
print("x_test",x_test.shape)
print("y_train",y_train.shape)
print("y_test",y_test.shape)

##Modal Training
from sklearn.naive_bayes import GaussianNB
model=GaussianNB()
model.fit(x_train,y_train) 


## Model Evalution

y_predict=model.predict(x_test)
print("Y_predict",y_predict)
print("y_test",y_test)


## Accuracy

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_predict)
print("Accuracy",accuracy)


## model Prediction

test_data=model.predict([[36,1,0,109,281,1,2,99,0,2.5,2,1,0,0]])
if test_data==1:
  print("Person has heart disease")
else:
  print("Person does not have heart disease")
ChatGPT said:
Your full pipeline is solid for a beginner-level machine learning project using Naive Bayes for heart disease prediction. Here's a concise review with best practices, improvements, and potential enhancements:

✅ What’s Working Well:
Correct train-test split (80/20).

Naive Bayes model training and prediction.

Accuracy metric used for evaluation.

Includes a manual prediction example.

🔧 Suggested Improvements:
1. Display predictions in a cleaner way
Use y_test.values to show actual labels properly:

print("Predicted values:", y_predict)
print("Actual values   :", y_test.values)
2. Confusion Matrix & Classification Report (Optional but helpful)
To evaluate model performance better:

from sklearn.metrics import confusion_matrix, classification_report

cm = confusion_matrix(y_test, y_predict)
print("Confusion Matrix:\n", cm)

print("\nClassification Report:\n", classification_report(y_test, y_predict))
3. Check Input Dimensions for Manual Prediction
Make sure the input is reshaped as a 2D array and matches feature count:

import numpy as np

sample_input = np.array([[36,1,0,109,281,1,2,99,0,2.5,2,1,0,0]])
prediction = model.predict(sample_input)

if prediction[0] == 1:
    print("Person has heart disease")
else:
    print("Person does not have heart disease")
4. (Optional) Add Feature Names for Manual Prediction
To avoid errors, match the manual input exactly to the preprocessed features in x.columns. You can also build a function:

def predict_heart_disease(input_data):
    input_array = np.array([input_data])
    result = model.predict(input_array)
    return "Person has heart disease" if result[0] == 1 else "Person does not have heart disease"

print(predict_heart_disease([36,1,0,109,281,1,2,99,0,2.5,2,1,0,0]))
✅ Final Tip:
For better modeling performance, consider:

Trying RandomForestClassifier, LogisticRegression, or XGBoost.

Performing cross-validation for reliable metrics.

Doing feature scaling (e.g., StandardScaler) for algorithms sensitive to feature magnitude (not necessary for Naive Bayes though).

Would you like help comparing models or visualizing accuracy?

You said:
suggest a name fo the project
ChatGPT said:
Here are some project name suggestions based on your heart disease prediction model:

🔬 Technical & Descriptive
CardioPredict

HeartRisk Analyzer

SmartHeart AI

HeartGuard

CardioStat

💡 Creative & Catchy
PulseCheck

BeatSense

HeartWise

RedAlert: Heart Edition

RhythmIQ

🧪 Academic/Professional Tone
Heart Disease Detection Using Naive Bayes

ML-Based Cardiac Risk Prediction

Probabilistic Heart Diagnosis System

HeartCare AI: A Machine Learning Approach

Predictive Analytics for Cardiac Health

Would you like the name to be more formal, creative, or research-paper-style?

You said:
i am going to put that in git hub

so suggest a description
ChatGPT said:
Great! Here’s a concise and effective GitHub project description tailored for your heart disease prediction project using Naive Bayes:

✅ GitHub Project Description
Title: CardioPredict - Heart Disease Detection using Naive Bayes

Description:

CardioPredict is a machine learning project that uses the Naive Bayes algorithm to predict the likelihood of heart disease based on key medical attributes. This end-to-end pipeline includes data preprocessing, model training, evaluation, and real-time prediction using user-provided data. The goal is to demonstrate the potential of probabilistic models in early diagnosis of heart conditions.

Key Features:

Data preprocessing with Label Encoding

Train/test split (80/20)

Gaussian Naive Bayes classifier

Accuracy evaluation and confusion matrix

Manual prediction support with custom inputs

Tech Stack:
Python, Pandas, scikit-learn, Jupyter Notebook

Would you like me to help you write a full README.md for this project too?

You said:
Description cannot be more than 350 characters
ChatGPT said:
Got it! Here's a concise GitHub description under 350 characters:

A machine learning project that uses Naive Bayes to predict heart disease based on patient data. Includes data preprocessing, model training, evaluation, and manual prediction. Built with Python and scikit-learn for healthcare analytics.

You said:
readme file
ChatGPT said:
Here's a clean and professional README.md file for your heart disease prediction project:

# CardioPredict - Heart Disease Detection using Naive Bayes

CardioPredict is a machine learning project that predicts the presence of heart disease using patient data. It uses the Gaussian Naive Bayes algorithm for classification and provides an end-to-end workflow from data preprocessing to model evaluation and manual prediction.

---

## 🔍 Overview

This project demonstrates the application of supervised learning in healthcare analytics. It uses labeled data containing features like age, gender, chest pain type, cholesterol levels, and more to classify whether a person has heart disease.

---

## 📁 Dataset

- File: `heart_disease_data_with_labels.xlsx`
- Source: Preloaded dataset
- Features include:
  - Age
  - Gender
  - Chest Pain Type
  - Fasting Blood Sugar
  - Resting ECG
  - Max Heart Rate Achieved
  - Exercise Induced Angina
  - ST Depression
  - Slope of Peak ST
  - Thalassemia
  - ...and more

---

## ⚙️ Tech Stack

- Python
- pandas
- scikit-learn
- Jupyter Notebook

---

## 🧪 Workflow

1. **Data Collection**  
   Load the dataset using `pandas`.

2. **Data Preprocessing**  
   Encode categorical variables using `LabelEncoder`.

3. **Train-Test Split**  
   Use 80% data for training and 20% for testing.

4. **Model Training**  
   Fit a `GaussianNB` model using scikit-learn.

5. **Model Evaluation**  
   Evaluate using accuracy score and predict unseen data.

6. **Manual Prediction**  
   Pass custom patient data to predict the likelihood of heart disease.

---

## 📊 Example Prediction

```python
model.predict([[36,1,0,109,281,1,2,99,0,2.5,2,1,0,0]])
✅ Output
1 → Person has heart disease

0 → Person does not have heart disease

📌 Future Improvements
Implement more models (Random Forest, Logistic Regression)

Add a web interface (e.g., using Flask)

Feature importance visualization

Hyperparameter tuning

