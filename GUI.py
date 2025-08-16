import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import sklearn.utils as u
import sklearn.preprocessing as pp
import sklearn.tree as tr
import sklearn.ensemble as es
import sklearn.linear_model as lm
import sklearn.neural_network as nn
import numpy as np
import tkinter as tk
from tkinter import messagebox, ttk

# Load your dataset
try:
    data = pd.read_csv("C:/Users/ACER/Downloads/StudentPerformancePrediction-ML-main/StudentPerformancePrediction-ML-main/AI-Data.csv")
except FileNotFoundError:
    raise Exception("The dataset file was not found. Please check the path.")

# Data Preprocessing
data = data.drop(["gender", "StageID", "GradeID", "NationalITy", "PlaceofBirth", 
                  "SectionID", "Topic", "Semester", "Relation", 
                  "ParentschoolSatisfaction", "ParentAnsweringSurvey", 
                  "AnnouncementsView"], axis=1)

u.shuffle(data)

# Encode categorical variables
gradeID_dict = {f"G-{str(i).zfill(2)}": i for i in range(1, 13)}
data = data.replace({"GradeID": gradeID_dict})

for column in data.columns:
    if data[column].dtype == type(object):
        le = pp.LabelEncoder()
        data[column] = le.fit_transform(data[column])

# Split the data into training and testing sets
ind = int(len(data) * 0.70)
feats = data.values[:, :-1]
lbls = data.values[:, -1]
feats_Train, feats_Test = feats[0:ind], feats[ind:]
lbls_Train, lbls_Test = lbls[0:ind], lbls[ind:]

# Train the models
modelD = tr.DecisionTreeClassifier()
modelD.fit(feats_Train, lbls_Train)

modelR = es.RandomForestClassifier()
modelR.fit(feats_Train, lbls_Train)

modelP = lm.Perceptron()
modelP.fit(feats_Train, lbls_Train)

modelL = lm.LogisticRegression()
modelL.fit(feats_Train, lbls_Train)

modelN = nn.MLPClassifier(activation="logistic")
modelN.fit(feats_Train, lbls_Train)

# Function to make predictions and plot graph
def make_prediction():
    try:
        # Get user input from the GUI
        rai = int(raised_hands_var.get())
        res = int(visited_resources_var.get())
        dis = int(discussions_var.get())
        absc = int(absences_var.get())
        
        # Prepare the input array for prediction
        arr = np.array([[rai, res, dis, absc]])
        
        # Make predictions using all models
        predD = modelD.predict(arr)[0]
        predR = modelR.predict(arr)[0]
        predP = modelP.predict(arr)[0]
        predL = modelL.predict(arr)[0]
        predN = modelN.predict(arr)[0]

        # Get prediction probabilities
        probD = modelD.predict_proba(arr)[0]
        probR = modelR.predict_proba(arr)[0]
        probP = np.array([None, None])  # No probability for Perceptron
        probL = modelL.predict_proba(arr)[0]
        probN = modelN.predict_proba(arr)[0]

        # Mapping predictions to labels
        predictions = {
            'Decision Tree': 'HML'[predD],
            'Random Forest': 'HML'[predR],
            'Perceptron': 'HML'[predP],
            'Logistic Regression': 'HML'[predL],
            'Neural Network': 'HML'[predN]
        }

        # Prepare percentage output
        percentages = {
            'Decision Tree': f"{probD[predD]:.2%}",
            'Random Forest': f"{probR[predR]:.2%}",
            'Perceptron': "N/A",  # No probability for Perceptron
            'Logistic Regression': f"{probL[predL]:.2%}",
            'Neural Network': f"{probN[predN]:.2%}"
        }

        # Display predictions and percentages
        result = "\n".join(f"{model}: {pred} ({percentages[model]})" for model, pred in predictions.items())
        messagebox.showinfo("Predictions", result)

        # Create a bar chart for the inputs
        input_labels = ['Raised Hands', 'Visited Resources', 'Discussions', 'Absences']
        input_values = [rai, res, dis, absc]

        plt.figure(figsize=(10, 5))
        
        # User input values bar chart
        plt.subplot(1, 2, 1)
        plt.bar(input_labels, input_values, color='skyblue')
        plt.ylabel('Count')
        plt.title('User Input Values')
        plt.xticks(rotation=45)

        # Prediction probabilities bar chart
        models = ['Decision Tree', 'Random Forest', 'Logistic Regression', 'Neural Network']
        prob_values = [probD[predD], probR[predR], probL[predL], probN[predN]]
        prob_values = [p if p is not None else 0 for p in prob_values]  # Handle None for Perceptron

        plt.subplot(1, 2, 2)
        plt.bar(models, prob_values, color='orange')
        plt.ylabel('Probability')
        plt.title('Prediction Probabilities')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.show()

    except ValueError as ve:
        messagebox.showerror("Input Error", "Please enter valid integers for all fields.")
        print(f"ValueError: {ve}")  # Log error to console
    except Exception as e:
        messagebox.showerror("Error", str(e))
        print(f"Error: {e}")  # Log error to console

# Create the GUI
root = tk.Tk()
root.title("Student Performance Prediction")

# Labels and inputs for user
tk.Label(root, text="Raised Hands:").grid(row=0, column=0)
raised_hands_var = tk.StringVar()
ttk.Entry(root, textvariable=raised_hands_var).grid(row=0, column=1)

tk.Label(root, text="Visited Resources:").grid(row=1, column=0)
visited_resources_var = tk.StringVar()
ttk.Entry(root, textvariable=visited_resources_var).grid(row=1, column=1)

tk.Label(root, text="Discussions:").grid(row=2, column=0)
discussions_var = tk.StringVar()
ttk.Entry(root, textvariable=discussions_var).grid(row=2, column=1)

tk.Label(root, text="Absences (1 for Under-7, 0 for Above-7):").grid(row=3, column=0)
absences_var = tk.StringVar()
ttk.Entry(root, textvariable=absences_var).grid(row=3, column=1)

# Prediction button
predict_button = ttk.Button(root, text="Predict", command=make_prediction)
predict_button.grid(row=4, columnspan=2)

# Start the GUI event loop
root.mainloop()
