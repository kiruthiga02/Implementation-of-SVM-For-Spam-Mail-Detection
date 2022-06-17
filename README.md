# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Moodle-Code Runner

## Algorithm
1. Import the necessary packages using import statement.
2. Read the given csv file and print the number of contents to be displayed.
3. Split the dataset using train_test_split.
4. Calculate Y_Pred and accuracy.
5. Display the result.

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: M.KIRUTHIGA
RegisterNumber: 212219040061 
*/
import pandas as pd
data=pd.read_csv("spam.csv",encoding='latin-1')
data.head()
data.info()
data.isnull().sum()
x=data["EmailText"].values
y=data["Label"].values
from sklearn.model_selection import train_test_split 
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)
from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```

## Output:


## Data.head():

![image](https://user-images.githubusercontent.com/98682825/174351431-28dad2cd-f27d-400f-a996-d986ec170b08.png)


## Data.info():

![image](https://user-images.githubusercontent.com/98682825/174351393-54534012-f88d-4c65-b9ac-51de1cee131c.png)


## Data.isnull().sum():

![image](https://user-images.githubusercontent.com/98682825/174351361-d331ae4f-8f6a-4775-824e-e8e870daa409.png)


## Y_Pred:

![image](https://user-images.githubusercontent.com/98682825/174351347-e7ddd437-d176-4234-8242-dfc5bc5a4503.png)


## Accuracy:

![image](https://user-images.githubusercontent.com/98682825/174351331-65ee7a22-254a-42f0-a551-c294d27ef13f.png)


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
