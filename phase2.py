import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Sample traffic accident dataset (you would use real-world data here)
# The columns could include: 'Traffic_Volume', 'Weather_Conditions', 'Road_Type', 'Time_of_Day', 'Accident'
# 'Accident' is the target variable: 0 = No Accident, 1 = Accident occurred

# Sample data for illustration purposes
data = {
    'Traffic_Volume': [500, 2000, 1500, 2500, 3000, 800, 1500, 1000, 2000, 1200],
    'Weather_Conditions': ['Clear', 'Rain', 'Clear', 'Fog', 'Clear', 'Clear', 'Rain', 'Clear', 'Clear', 'Fog'],
    'Road_Type': ['Highway', 'Urban', 'Highway', 'Urban', 'Highway', 'Highway', 'Urban', 'Urban', 'Highway', 'Urban'],
    'Time_of_Day': ['Day', 'Night', 'Day', 'Night', 'Day', 'Night', 'Day', 'Night', 'Day', 'Night'],
    'Accident': [0, 1, 0, 1, 0, 0, 1, 0, 0, 1]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Encode categorical variables using one-hot encoding
df_encoded = pd.get_dummies(df, drop_first=True)

# Features (independent variables) and target variable (dependent)
X = df_encoded.drop('Accident', axis=1)
y = df_encoded['Accident']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Plot confusion matrix
plt.figure(figsize=(7,5))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', xticklabels=["No Accident", "Accident"], yticklabels=["No Accident", "Accident"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Feature importance
feature_importance = model.feature_importances_
features = X.columns
plt.barh(features, feature_importance)
plt.title("Feature Importance")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.show()
