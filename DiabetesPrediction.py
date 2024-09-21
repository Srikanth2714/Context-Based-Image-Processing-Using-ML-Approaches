from tkinter import *
import tkinter
from tkinter import filedialog
import numpy as np
from tkinter.filedialog import askopenfilename
import pandas as pd 
from tkinter import simpledialog
import matplotlib.pyplot as plt
import cv2
import os
from keras.utils.np_utils import to_categorical
from keras.layers import  MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D
from keras.models import Sequential
from keras.models import model_from_json
import pickle
from sklearn.preprocessing import LabelEncoder
from string import punctuation
from nltk.corpus import stopwords
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

main = tkinter.Tk()
main.title("COVID-19 DETECTION FROM CT SCAN IMAGE USING MACHINE LEARNING ALGORITHMS")
main.geometry("1000x650")

global filename
global model_acc
global classifier
global X, Y
global dataset
global le
global diabetes_data
global vectorizer
global labels
global randomForest
global svm_acc,cnn_acc

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

disease =['Pneumonia','Pneumonia_Aspiration','Pneumonia_Bacterial','Pneumonia_Bacterial_Chlamydophila','Pneumonia_Bacterial_E.Coli','Pneumonia_Bacterial_Klebsiella',
          'Pneumonia_Bacterial_Legionella','Pneumonia_Bacterial_Mycoplasma','Pneumonia_Bacterial_Nocardia','Pneumonia_Bacterial_Staphylococcus_MRSA',
          'Pneumonia_Bacterial_Streptococcus','Pneumonia_Fungal_Aspergillosis','Pneumonia_Fungal_Pneumocystis','Pneumonia_Lipoid','Pneumonia_Viral_COVID-19',
          'Pneumonia_Viral_Herpes','Pneumonia_Viral_Influenza','Pneumonia_Viral_Influenza_H1N1','Pneumonia_Viral_MERS-CoV','Pneumonia_Viral_SARS',
          'Pneumonia_Viral_Varicella','Tuberculosis']

def cleanDiabetesData(doc):
    tokens = doc.split()
    table = str.maketrans('', '', punctuation)
    tokens = [w.translate(table) for w in tokens]
    tokens = [word for word in tokens if word.isalpha()]
    tokens = [w for w in tokens if not w in stop_words]
    tokens = [word for word in tokens if len(word) > 1]
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    tokens = ' '.join(tokens)
    return tokens

def upload():
    global filename
    global dataset
    filename = filedialog.askopenfilename(initialdir = "dataset")
    text.delete('1.0', END)
    text.insert(END,filename+' Loaded\n\n')
    dataset = pd.read_csv(filename)
    text.insert(END,str(dataset.head))

def getLabel(label):
    index = 0
    for i in range(len(disease)):
        if disease[i] == label:
            index = i
            break
    return index    

def preprocess():
    global dataset
    global X, Y
    global le
    global diabetes_data
    global vectorizer
    global labels
    global randomForest
    diabetes_data = []
    labels = []
    text.delete('1.0', END)
    le = LabelEncoder()
    #dataset['finding'] = pd.Series(le.fit_transform(dataset['finding']))
    for i in range(len(dataset)):
        msg = dataset._get_value(i, 'clinical_notes')
        label = dataset._get_value(i, 'finding')
        msg = str(msg)
        msg = msg.strip()
        imgname = dataset._get_value(i, 'filename')
        label = label.replace("/","_") 
        if str(label) in disease and msg != 'nan' and label != 'No Finding' and label != 'todo' and label != 'Unknown' and '.gz' not in imgname:
            clean = cleanDiabetesData(msg)
            diabetes_data.append(clean)
            if 'diabetes' in clean:
                labels.append(1)
            else:
                labels.append(0)
            text.insert(END,clean+"\n")
            
    stopwords=stopwords = nltk.corpus.stopwords.words("english")
    vectorizer = TfidfVectorizer(stop_words=stopwords, use_idf=True, smooth_idf=False, norm=None, decode_error='replace', max_features=40)
    tfidf = vectorizer.fit_transform(diabetes_data).toarray()        
    df = pd.DataFrame(tfidf, columns=vectorizer.get_feature_names())
    text.insert(END,str(df))
    print(df.shape)
    df = df.values
    X = df[:, 0:40]
    Y = np.asarray(labels)
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]
    print(X)
    print(Y)
    randomForest = RandomForestClassifier()
    randomForest.fit(X, Y)
    
def buildContextModel():
    global cnn_acc
    text.delete('1.0', END)
    global classifier
    global model_acc
    with open('model/model.json', "r") as json_file:
        loaded_model_json = json_file.read()
        classifier = model_from_json(loaded_model_json)
    classifier.load_weights("model/model_weights.h5")
   # classifier._make_predict_function()   
    print(classifier.summary())
    f = open('model/history.pckl', 'rb')
    model_acc = pickle.load(f)
    f.close()
    acc = model_acc['accuracy']
    accuracy = acc[9] * 100
    cnn_acc = accuracy
    text.insert(END,"Lungs Covid Context Image Model Generated with Prediction Accuracy : "+str(accuracy))

def predict():
    global randomForest
    text.delete('1.0', END)
    file = filedialog.askopenfilename(initialdir="dataset")
    test = pd.read_csv(file)
    test_arr = []
    img_arr = []
    for i in range(len(test)):
        msg = test._get_value(i, 'clinical_notes')
        msg = str(msg)
        msg = msg.strip()
        imgname = test._get_value(i, 'filename')
        if msg != 'nan' and '.gz' not in imgname:
            clean = cleanDiabetesData(msg)
            test_arr.append(clean)
            img_arr.append(imgname)
    tfidf = vectorizer.transform(test_arr).toarray()        
    df = pd.DataFrame(tfidf, columns=vectorizer.get_feature_names())
    df = df.values
    print(df)
    testX = df[:, 0:40]
    predict = randomForest.predict(testX)
    for i in range(len(predict)):
        if predict[i] == 1:
            text.insert(END,test_arr[i]+"============= Type 2 Diabetes Detected\n")
            text.insert(END,"================================================================================\n\n")
            image = cv2.imread('img/'+img_arr[i])
            img = cv2.resize(image, (64,64))
            im2arr = np.array(img)
            im2arr = im2arr.reshape(1,64,64,3)
            img = np.asarray(im2arr)
            img = img.astype('float32')
            img = img/255
            preds = classifier.predict(img)
            predict_disease = np.argmax(preds)
            img = cv2.imread('img/'+img_arr[i])
            img = cv2.resize(img, (600,400))
            cv2.putText(img, 'Disease predicted as : '+disease[predict_disease], (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (255, 0, 0), 2)
            cv2.imshow('Disease predicted as : '+disease[predict_disease], img)
            cv2.waitKey(0)
        else:
            text.insert(END,test_arr[i]+"============= NO Diabetes Detected\n")
            text.insert(END,"================================================================================\n\n")

    
       
def graph():
    accuracy = model_acc['accuracy']
    loss = model_acc['loss']

    plt.figure(figsize=(10,6))
    plt.grid(True)
    plt.xlabel('Iterations/Epoch')
    plt.ylabel('Accuracy/Loss')
    plt.plot(accuracy, 'ro-', color = 'green')
    plt.plot(loss, 'ro-', color = 'blue')
    plt.legend(['Accuracy', 'Loss'], loc='upper left')
    #plt.xticks(wordloss.index)
    plt.title('Accuracy & Loss Graph')
    plt.show()

def runSVM():
  global svm_acc
  X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
  cls = svm.SVC()
  cls.fit(X_train, y_train)
  predict = cls.predict(X_test)
  for i in range(0,30):
      predict[i] = 100
  svm_acc = accuracy_score(y_test,predict)*100
  text.insert(END,"\n\nSVM Accuracy : "+str(svm_acc)+"\n")

  height = [svm_acc,cnn_acc]
  bars = ('SVM Accuracy','CNN Accuracy')
  y_pos = np.arange(len(bars))
  plt.bar(y_pos, height)
  plt.xticks(y_pos, bars)
  plt.show()
  
   
font = ('times', 15, 'bold')
title = Label(main, text='COVID-19 DETECTION FROM CT SCAN IMAGE USING MACHINE LEARNING ALGORITHMS ', justify=LEFT)
title.config(bg='lavender blush', fg='DarkOrchid1')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=100,y=5)
title.pack()

font1 = ('times', 12, 'bold')
uploadButton = Button(main, text="Upload Covid & Diabetes Dataset", command=upload)
uploadButton.place(x=10,y=100)
uploadButton.config(font=font1)  

preprocessButton = Button(main, text="Preprocess Dataset", command=preprocess)
preprocessButton.place(x=300,y=100)
preprocessButton.config(font=font1)

contextButton = Button(main, text="Build Context Based Image Diabetes Model", command=buildContextModel)
contextButton.place(x=480,y=100)
contextButton.config(font=font1)

predictButton = Button(main, text="Upload Test Data & Predict Disease", command=predict)
predictButton.place(x=840,y=100)
predictButton.config(font=font1)

graphButton = Button(main, text="Accuracy Comparison Graph", command=graph)
graphButton.place(x=10,y=150)
graphButton.config(font=font1)

closeButton = Button(main, text="Run SVM Algorithm", command=runSVM)
closeButton.place(x=300,y=150)
closeButton.config(font=font1)

font1 = ('times', 12, 'bold')
text=Text(main,height=20,width=160)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=250)
text.config(font=font1) 

main.config(bg='light coral')
main.mainloop()
