import pandas as pd
import sklearn.neighbors as ng
import matplotlib.pyplot as pt
import joblib

mydata = pd.read_csv("data.csv")

x = mydata[["height"]]
y = mydata[["weight"]]

knn_height_model = ng.KNeighborsRegressor(n_neighbors=3)
knn_height_model.fit(x,y) #training the model
print("Training Completed")
joblib.dump(knn_height_model,"knn_height_model.pkl")
