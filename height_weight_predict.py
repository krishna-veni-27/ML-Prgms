import joblib
knn_height_model = joblib.load("knn_height_model.pkl")
input = int(input("Enter height for prediction : "))
print(knn_height_model.predict([[input]]))