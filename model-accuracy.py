import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import time as t
import sklearn.utils as u
import sklearn.preprocessing as pp
import sklearn.tree as tr
import sklearn.ensemble as es
import sklearn.metrics as m
import sklearn.linear_model as lm
import sklearn.neural_network as nn
import numpy as np
#import random as rnd
import warnings as w
w.filterwarnings('ignore')

# Load data
data = pd.read_csv("C:/Users/ACER/Downloads/StudentPerformancePrediction-ML-main/StudentPerformancePrediction-ML-main/AI-Data.csv")

# Menu for graph choices
ch = 0
while(ch != 10):
    print("1.Marks Class Count Graph\t2.Marks Class Semester-wise Graph\n3.Marks Class Gender-wise Graph\t4.Marks Class Nationality-wise Graph\n5.Marks Class Grade-wise Graph\t6.Marks Class Section-wise Graph\n7.Marks Class Topic-wise Graph\t8.Marks Class Stage-wise Graph\n9.Marks Class Absent Days-wise\t10.No Graph\n")
    ch = int(input("Enter Choice: "))
    if (ch == 1):
        print("Loading Graph....\n")
        t.sleep(1)
        print("\tMarks Class Count Graph")
        axes = sb.countplot(x='Class', data=data, order=['L', 'M', 'H'])
        plt.show()
    elif (ch == 2):
        print("Loading Graph....\n")
        t.sleep(1)
        print("\tMarks Class Semester-wise Graph")
        fig, axesarr = plt.subplots(1, figsize=(10, 6))
        sb.countplot(x='Semester', hue='Class', data=data, hue_order=['L', 'M', 'H'], axes=axesarr)
        plt.show()
    elif (ch == 3):
        print("Loading Graph..\n")
        t.sleep(1)
        print("\tMarks Class Gender-wise Graph")
        fig, axesarr = plt.subplots(1, figsize=(10, 6))
        sb.countplot(x='gender', hue='Class', data=data, order=['M', 'F'], hue_order=['L', 'M', 'H'], axes=axesarr)
        plt.show()
    elif (ch == 4):
        print("Loading Graph..\n")
        t.sleep(1)
        print("\tMarks Class Nationality-wise Graph")
        fig, axesarr = plt.subplots(1, figsize=(10, 6))
        sb.countplot(x='NationalITy', hue='Class', data=data, hue_order=['L', 'M', 'H'], axes=axesarr)
        plt.show()
    elif (ch == 5):
        print("Loading Graph: \n")
        t.sleep(1)
        print("\tMarks Class Grade-wise Graph")
        fig, axesarr = plt.subplots(1, figsize=(10, 6))
        sb.countplot(x='GradeID', hue='Class', data=data, order=['G-02', 'G-04', 'G-05', 'G-06', 'G-07', 'G-08', 'G-09', 'G-10', 'G-11', 'G-12'], hue_order = ['L', 'M', 'H'], axes=axesarr)
        plt.show()
    elif (ch ==6):
        print("Loading Graph..\n")
        t.sleep(1)
        print("\tMarks Class Section-wise Graph")
        fig, axesarr = plt.subplots(1, figsize=(10, 6))
        sb.countplot(x='SectionID', hue='Class', data=data, hue_order = ['L', 'M', 'H'], axes=axesarr)
        plt.show()
    elif (ch == 7):
        print("Loading Graph..\n")
        t.sleep(1)
        print("\tMarks Class Topic-wise Graph")
        fig, axesarr = plt.subplots(1, figsize=(10, 6))
        sb.countplot(x='Topic', hue='Class', data=data, hue_order = ['L', 'M', 'H'], axes=axesarr)
        plt.show()
    elif (ch == 8):
        print("Loading Graph..\n")
        t.sleep(1)
        print("\tMarks Class Stage-wise Graph")
        fig, axesarr = plt.subplots(1, figsize=(10, 6))
        sb.countplot(x='StageID', hue='Class', data=data, hue_order = ['L', 'M', 'H'], axes=axesarr)
        plt.show()
    elif (ch == 9):
        print("Loading Graph..\n")
        t.sleep(1)
        print("\tMarks Class Absent Days-wise Graph")
        fig, axesarr = plt.subplots(1, figsize=(10, 6))
        sb.countplot(x='StudentAbsenceDays', hue='Class', data=data, hue_order = ['L', 'M', 'H'], axes=axesarr)
        plt.show()

if(ch == 10):
    print("Exiting..\n")
    t.sleep(1)

# Data preprocessing
data = data.drop("gender", axis=1)
data = data.drop("StageID", axis=1)
data = data.drop("GradeID", axis=1)
data = data.drop("NationalITy", axis=1)
data = data.drop("PlaceofBirth", axis=1)
data = data.drop("SectionID", axis=1)
data = data.drop("Topic", axis=1)
data = data.drop("Semester", axis=1)
data = data.drop("Relation", axis=1)
data = data.drop("ParentschoolSatisfaction", axis=1)
data = data.drop("ParentAnsweringSurvey", axis=1)
data = data.drop("AnnouncementsView", axis=1)
u.shuffle(data)

# Encoding GradeID
gradeID_dict = {
    "G-01" : 1,
    "G-02" : 2,
    "G-03" : 3,
    "G-04" : 4,
    "G-05" : 5,
    "G-06" : 6,
    "G-07" : 7,
    "G-08" : 8,
    "G-09" : 9,
    "G-10" : 10,
    "G-11" : 11,
    "G-12" : 12
}
data = data.replace({"GradeID" : gradeID_dict})

# Label Encoding
for column in data.columns:
    if data[column].dtype == type(object):
        le = pp.LabelEncoder()
        data[column] = le.fit_transform(data[column])

# Splitting data
ind = int(len(data) * 0.70)
feats = data.values[:, 0:4]
lbls = data.values[:, 4]
feats_Train = feats[0:ind]
feats_Test = feats[(ind + 1):len(feats)]
lbls_Train = lbls[0:ind]
lbls_Test = lbls[(ind + 1):len(lbls)]

# Initialize accuracy counts
accuracies = {}

# Decision Tree Classifier
modelD = tr.DecisionTreeClassifier()
modelD.fit(feats_Train, lbls_Train)
lbls_predD = modelD.predict(feats_Test)
accD = m.accuracy_score(lbls_Test, lbls_predD)
accuracies['Decision Tree'] = accD
print("\nAccuracy using Decision Tree: ", str(round(accD * 100, 2)) + "%")

# Random Forest Classifier
modelR = es.RandomForestClassifier()
modelR.fit(feats_Train, lbls_Train)
lbls_predR = modelR.predict(feats_Test)
accR = m.accuracy_score(lbls_Test, lbls_predR)
accuracies['Random Forest'] = accR
print("\nAccuracy using Random Forest: ", str(round(accR * 100, 2)) + "%")

# Perceptron
modelP = lm.Perceptron()
modelP.fit(feats_Train, lbls_Train)
lbls_predP = modelP.predict(feats_Test)
accP = m.accuracy_score(lbls_Test, lbls_predP)
accuracies['Perceptron'] = accP
print("\nAccuracy using Linear Model Perceptron: ", str(round(accP * 100, 2)) + "%")

# Logistic Regression
modelL = lm.LogisticRegression()
modelL.fit(feats_Train, lbls_Train)
lbls_predL = modelL.predict(feats_Test)
accL = m.accuracy_score(lbls_Test, lbls_predL)
accuracies['Logistic Regression'] = accL
print("\nAccuracy using Linear Model Logistic Regression: ", str(round(accL * 100, 2)) + "%")

# Neural Network (MLP Classifier)
modelN = nn.MLPClassifier(activation="logistic")
modelN.fit(feats_Train, lbls_Train)
lbls_predN = modelN.predict(feats_Test)
accN = m.accuracy_score(lbls_Test, lbls_predN)
accuracies['Neural Network'] = accN
print("\nAccuracy using Neural Network MLP Classifier: ", str(round(accN * 100, 2)) + "%")

# Plotting the accuracy results
plt.figure(figsize=(10, 6))
plt.bar(accuracies.keys(), [value * 100 for value in accuracies.values()], color='skyblue')
plt.xlabel('Models')
plt.ylabel('Accuracy (%)')
plt.title('Model Accuracy Comparison')
plt.ylim(0, 100)  # Set y-axis limits to 0-100
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.show()
