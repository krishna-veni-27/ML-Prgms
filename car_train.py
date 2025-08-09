import pandas as pd
from sklearn.preprocessing import LabelEncoder
import sklearn.neighbors as ng
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import joblib
mydata = pd.read_csv("car_data.csv")
BrandEncoder = LabelEncoder()
FuelTypeEncoder = LabelEncoder()
mydata["Brand_enc"] = BrandEncoder.fit_transform(mydata["Brand"])
mydata["FuelType_enc"] = FuelTypeEncoder.fit_transform(mydata["FuelType"])
x = mydata[["Year","Brand_enc","FuelType_enc","Mileage"]]
y = mydata[["Price"]]
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2)
car_model = Sequential()
car_model.add(Dense(10,activation = "relu",input_shape =(4,)))
car_model.add(Dense(10,activation = "relu"))
car_model.add(Dense(1))
car_model.compile(optimizer = "adam",loss ="mse")
car_model.fit(x_train,y_train,epochs = 10)
print("Training Completed......")
joblib.dump(car_model,"car_model.pkl")
test_result = car_model.predict(x_test)
