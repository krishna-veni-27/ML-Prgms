import pandas as pd
from sklearn.preprocessing import LabelEncoder
import sklearn.neighbors as ng
from sklearn.metrics import r2_score,mean_squared_error,root_mean_squared_error,confusion_matrix
from sklearn.model_selection import train_test_split
import joblib
import math
mydata = pd.read_csv("dataset_weight_finder.csv")
GenderEncoder = LabelEncoder()
BodyTypeEncoder = LabelEncoder()
mydata["Gender_enc"] = GenderEncoder.fit_transform(mydata["Gender"])
mydata["BodyType_enc"] = BodyTypeEncoder.fit_transform(mydata["BodyType"])
x = mydata[["Age","Gender_enc","BodyType_enc","Height"]]
y = mydata[["Weight"]]
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2)
knn_weight_model = ng.KNeighborsClassifier(n_neighbors = 3)
knn_weight_model.fit(x_train,y_train)
print("Training Completed......")
joblib.dump(knn_weight_model,"knn_weight_model.pkl")
test_result = knn_weight_model.predict(x_test)
print("MSE",mean_squared_error(y_test,test_result))
print("R2 score",r2_score(y_test,test_result))
print("RMSE", root_mean_squared_error(y_test,test_result))
print("Confusion Matrix\n",confusion_matrix(y_test,test_result))