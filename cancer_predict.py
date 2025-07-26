import joblib 
cancer_model = joblib.load("cancer_model.pkl")
result = cancer_model.predict([[18.65,17.6,123.7,1076,0.1099,0.1686,0.1974,0.1009,0.1907,0.06049,0.6289,0.6633,4.293,71.56,0.006294,0.03994,0.05554,0.01695,0.02428,0.003535,22.82,21.32,150.6,1567,0.1679,0.509,0.7345,0.2378,0.3799,0.09185]])
if result[0] == "M":
    print("cancerous")
else:
    print("non cancerous")
 