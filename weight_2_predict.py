import joblib
import numpy as np
model = joblib.load("model.pkl")
print(model.predict(np.array([[32,1,1,172]])))
