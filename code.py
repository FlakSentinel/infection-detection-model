import numpy as nm
import matplotlib.pyplot as mtp
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

#Imports the CSV file
records=pd.read_csv('covrecords.csv')

#Extracts the required columns/fields from the CSV file
x=records.iloc[:,3:4].values
y=records.iloc[:,4].values

#Splits the data into training set and testing set
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
x_test_copy=x_test

#Uses Feature Scaling
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)

#Creates the Logistic Regression Model
model=LogisticRegression(random_state=0)
model.fit(x_train,y_train)

#Determines the threshold value
for i in range(50):
    y_pred=model.predict_proba(sc.transform([[i]]))
    if (y_pred[0][0]>=y_pred[0][1]):
                         print("The threshold CT value is",i-1,"\n")
                         break

#Creates the set of predicted values using the model.
y_pred=model.predict(x_test)

#Plots the graph for true values
mtp.scatter(x_test_copy,y_test,color="red")
mtp.ylabel("Positive Value")
mtp.xlabel("CT Value")
mtp.title("True Values")
mtp.savefig("true.png")
mtp.clf()

#Plots the graph for predicted values
mtp.scatter(x_test_copy,y_pred,color="orange")
mtp.ylabel("Positive Value")
mtp.xlabel("CT Value")
mtp.title("Predicted Values")
mtp.savefig("pred.png")
mtp.clf()

#Creates the confusion matrix
cm=confusion_matrix(y_test,y_pred)
print("Confusion Matrix:")
print(cm)

print("The accuracy of the model is",((cm[0][0]+cm[1][1])/(cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1]))*100,"%")
