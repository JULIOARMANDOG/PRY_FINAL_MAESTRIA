import pandas as pd
import numpy as np
import matplotlib.pyplot  as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,confusion_matrix


#load properties 
df_features=pd.read_csv("./uci-har/features.txt",sep=r'\s+',header=None)
feature_names=df_features[1].tolist()

#load data train
X_train=pd.read_csv("./uci-har/train/X_train.txt",sep=r'\s+',header=None)
y_train=pd.read_csv("./uci-har/train/y_train.txt",sep=r'\s+',header=None)

#load data testing
X_test=pd.read_csv("./uci-har/test/X_test.txt",sep=r'\s+',header=None)
y_test=pd.read_csv("./uci-har/test/y_test.txt",sep=r'\s+',header=None)

#load activity labels
activities=pd.read_csv("./uci-har/activity_labels.txt",sep=r'\s+',header=None,index_col=0)
#map activity label with code activity
y_train=y_train[0].map(activities[1])
y_test=y_test[0].map(activities[1])

# Scaler data
scaler=StandardScaler()
X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.fit_transform(X_test)
#randon forest
clasification_forest=RandomForestClassifier(n_estimators=100,random_state=46)
#train model
clasification_forest.fit(X_train_scaled,y_train)
#Do prediction
activity_prediction=clasification_forest.predict(X_test_scaled)
#print report
print(classification_report(y_test,activity_prediction))


#read file that have data from mobile devices
#df_devices=pd.read_csv('./data/dispositivos.csv')
#print(df_devices.head())
#print the descriptive statistics for categorical variable
#print(df_devices.describe(include=['object']))
#the statistical values for categorical variable is :
#count : 10299 rows withput null value for this variable [Activity]
#unique: 6 : number of different activities
#top :   LAYING , most frecuent activity in the data set
#freq :  1944 , number of times with the most frecuent activity : LAYING
#activity_counts=df_devices['Activity'].value_counts()
#plot the bar graphics to show the differents activity bry frecuency
#activity_counts.plot(kind='bar',color='skyblue',edgecolor='black')
#plt.title("Activity Distribution")
#plt.xlabel('Activity')
#plt.ylabel('Frecuency')
#plt.xticks(rotation=45)
#plt.tight_layout()
#plt.show()
#cake graphics to observe class imbalance
#plt.figure(figsize=(8,6))
#activity_counts.plot(kind='pie',autopct='%1.1f%%')
#plt.title('Distibution by Activity')
#plt.ylabel('')
#plt.show()





