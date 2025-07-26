import pandas as pd
from sklearn.linear_model import LinearRegression
mydata = pd.read_csv("weight_prediction_dataset.csv")
print(mydata)

x = mydata[["height","age","bmi","muscle_mass","body_fat"]]
y = mydata[["weight"]]

model = LinearRegression()
model.fit(x,y)
print(model.predict([[160,35,83.5,36,22]]))