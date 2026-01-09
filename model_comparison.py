import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

house = pd.read_csv("housing.csv").dropna()

y = house.median_house_value

features = ["housing_median_age", "total_bedrooms", "total_rooms"]
X = house[features]

trainX, valX, trainy, valy = train_test_split(X, y, random_state=2)

house_model = DecisionTreeRegressor(max_leaf_nodes=125,random_state=1)
house_model.fit(trainX, trainy)

val_prediction = house_model.predict(valX)
print("Decision Tree Regressor: ",mean_absolute_error(valy,val_prediction))

random_tree = RandomForestRegressor(max_leaf_nodes= 125,random_state=0)
random_tree.fit(trainX, trainy)


prediction = random_tree.predict(valX)
print("Random Forest Regressor: ",mean_absolute_error(valy, prediction))