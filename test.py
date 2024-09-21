import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import cv2
import os
'''
dataset = pd.read_csv('pima-indians-diabetes.csv')
le = LabelEncoder()
dataset['Group'] = pd.Series(le.fit_transform(dataset['Group']))
dataset.drop(['Class'], axis = 1,inplace=True)
dataset.fillna(0, inplace = True)
dataset = dataset.values
col = dataset.shape[1] - 1
X = dataset[:,0:dataset.shape[1]]
Y = dataset[:,col]
Y = Y.astype('int')


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)
prediction_data = rfc.predict(X_test) 
accuracy = accuracy_score(y_test,prediction_data)*100
print(accuracy)
'''
dirs =['Pneumonia_Bacterial_Klebsiella', 'Pneumonia_Viral_Varicella', 'Pneumonia_Bacterial_E.Coli', 'Pneumonia_Viral_Influenza', 'Pneumonia_Viral_Influenza_H1N1',
       'Pneumonia_Viral_SARS', 'Pneumonia_Viral_MERS-CoV', 'Pneumonia_Bacterial_Legionella', 'Pneumonia_Bacterial_Chlamydophila', 'Tuberculosis',
       'Pneumonia_Fungal_Aspergillosis', 'Pneumonia_Lipoid', 'Pneumonia_Viral_Herpes ', 'Pneumonia_Bacterial_Streptococcus', 'Pneumonia_Bacterial',
       'Pneumonia_Viral_COVID-19', 'Pneumonia', 'Pneumonia_Aspiration', 'Pneumonia_Bacterial_Mycoplasma', 'Pneumonia_Bacterial_Staphylococcus_MRSA',
       'Pneumonia_Fungal_Pneumocystis', 'Pneumonia_Bacterial_Nocardia']
strs = ''
labels = []
dataset = pd.read_csv("dataset.csv")
for i in range(len(dataset)):
    msg = dataset.get_value(i, 'clinical_notes')
    label = dataset.get_value(i, 'finding')
    imgname = dataset.get_value(i, 'filename')
    if label != 'No Finding' and label != 'todo' and label != 'Unknown' and '.gz' not in imgname:
       label = label.replace("/","_") 
       img = cv2.imread('img/'+imgname)
       print(label+" "+str(imgname))
       cv2.imwrite('images/'+label+"/"+imgname,img)
       labels.append(label)
      

print(set(labels))
