import joblib
knn_iris_model = joblib.load("knn_iris_model.pkl")
print(knn_iris_model.predict([[4.4,2.9,1.4,0.2]]))