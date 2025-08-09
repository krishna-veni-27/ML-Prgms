import joblib
import numpy as np
car_model = joblib.load("car_model.pkl")
result = car_model.predict(np.array([[2012,1,1,16.4]]))
print(round(result[0][0],0))