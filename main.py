# importing the dependencies
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Read the data from file and dummy variables

rock_mine = pd.read_csv("Copy of sonar data.csv")
df = rock_mine.copy()
df_dummies = pd.get_dummies(df["R"])
df.drop("R", axis = 1, inplace=True)
df = pd.concat([df, df_dummies["R"]], axis=1)

# Creating independent and dependant variables
X = df.drop("R", axis = 1)
y = df["R"]

# Test & Train split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# Creating model and training the model
lr = LogisticRegression(solver="liblinear")
lr_model = lr.fit(X_train, y_train)

# Accurancy of the test dataset
y_pred = lr_model.predict(X_test)
accuracy_score(y_test, y_pred)

# Making a preditive system

input_data = (0.0522,0.0437,0.0180,0.0292,0.0351,0.1171,0.1257,
              0.1178,0.1258,0.2529,0.2716,0.2374,0.1878,0.0983,
              0.0683,0.1503,0.1723,0.2339,0.1962,0.1395,0.3164,
              0.5888,0.7631,0.8473,0.9424,0.9986,0.9699,1.0000,
              0.8630,0.6979,0.7717,0.7305,0.5197,0.1786,0.1098,
              0.1446,0.1066,0.1440,0.1929,0.0325,0.1490,0.0328,
              0.0537,0.1309,0.0910,0.0757,0.1059,0.1005,0.0535,
              0.0235,0.0155,0.0160,0.0029,0.0051,0.0062,0.0089,
              0.0140,0.0138,0.0077,0.0031) # output is M

input_numpy_array = np.asarray(input_data)

input_numpy_array.reshape(1,-1)

prediction = lr_model.predict([input_numpy_array])


if prediction[0] == 0:
    print("The object is a mine")
else:
    print("The object is a rock")

