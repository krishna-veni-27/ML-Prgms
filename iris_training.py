import pandas as pd
import sklearn.neighbors as ng
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.model_selection import train_test_split
import math
import joblib
mydata = pd.read_csv("Iris.csv")
x = mydata[["SepalLengthCm","SepalWidthCm","PetalLengthCm","PetalWidthCm"]]
y = mydata[["Species"]]
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)
knn_iris_model = ng.KNeighborsClassifier(n_neighbors=3)
knn_iris_model.fit(x_train,y_train)
print("Training Completed......")
joblib.dump(knn_iris_model,"knn_iris_model.pkl")
test_result = knn_iris_model.predict(x_test)
print("Accuracy",accuracy_score(y_test,test_result)*100)
print("Rounded Accuracy ",round(accuracy_score(y_test,test_result)*100,2))
print("Confusion Matrix\n",confusion_matrix(y_test,test_result))