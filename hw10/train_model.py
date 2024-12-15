import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import joblib

data = pd.read_csv('../realty_data.csv')

features = ['total_square', 'rooms', 'floor']
target = 'price'

data = data.dropna(subset=features + [target])

X = data[features]
y = data[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=239)

model = GradientBoostingRegressor(random_state=239)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

joblib.dump((model, X.columns), "../real_estate_model.pkl")
