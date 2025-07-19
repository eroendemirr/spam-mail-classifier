import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay

df=pd.read_csv("veriler.csv")

X=df.drop("label",axis=1)
y=df["label"]

X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.23,random_state=0)

model=LogisticRegression()
model.fit(X_train,y_train)

y_pred=model.predict(X_test)

print(classification_report(y_test,y_pred))

cm=confusion_matrix(y_test,y_pred,labels=model.classes_)

disp=ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=model.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.show()
