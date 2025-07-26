import pandas as pd
import sklearn.neighbors as ng
from sklearn.metrics import r2_score,mean_squared_error,accuracy_score
from sklearn.model_selection import train_test_split
import joblib
import math
mydata = pd.read_csv("heart.csv")
x = mydata[["age","sex","cp","trestbps","chol","fbs","restecg","thalach","exang","oldpeak","slope","ca","thal"]]
y = mydata[["target"]]
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)
knn_heart_model = ng.KNeighborsClassifier(n_neighbors=5)
knn_heart_model.fit(x_train,y_train)
print("Training completed......")
joblib.dump(knn_heart_model,"knn_heart_model.pkl")
test_result = knn_heart_model.predict(x_test)
print("MSE",mean_squared_error(y_test,test_result))
print("RMSE",math.sqrt(mean_squared_error(y_test,test_result)))
print("R2 score",r2_score(y_test,test_result))
print("Accuracy",accuracy_score(y_test,test_result)*100)

