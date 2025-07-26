import joblib 
knn_heart_model = joblib.load("knn_heart_model.pkl") 
result = knn_heart_model.predict([[46,1,0,120,249,0,0,144,0,0.8,2,0,3]])
if result[0] == 1:
    print("Patient have heart disease")
else:
    print("Patient doesn't have heart disease")