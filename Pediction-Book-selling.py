import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
df=pd.read_csv("bestsellers with categories.csv")
print(df)
sns=sns.countplot(data=df)
plt.show()
  
miss=df.columns[df.isna().any()]
print(miss)

from sklearn.preprocessing import LabelEncoder, StandardScaler

label_encoder = LabelEncoder()
df['Author_encoded'] = label_encoder.fit_transform(df['Author'])
df['Genre_encoded'] = label_encoder.fit_transform(df['Genre'])
print(df)

from sklearn.model_selection import train_test_split

import numpy as np

df['sales'] = (df['Reviews'] * df['User Rating'] * np.random.uniform(0.5, 1.5, len(df))).astype(int)

print(df['sales'])

X = df[['Author_encoded', 'Reviews', 'User Rating', 'Year', 'Genre_encoded', 'Price']]
y = df['sales']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

from sklearn.ensemble import RandomForestRegressor

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

rf_model.fit(X_train_scaled, y_train)

y_pred = rf_model.predict(X_test_scaled)

from sklearn.metrics import mean_absolute_error, mean_squared_error

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f"Mean Absolute Error (MAE): {mae}")
print(f"Root Mean Squared Error (RMSE): {rmse}")

from sklearn.metrics import r2_score

r2 = r2_score(y_test, y_pred)*100
print(f"R-squared (R2): {r2}")

feature_importances = rf_model.feature_importances_
features = X.columns
importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)


from sklearn.preprocessing import OrdinalEncoder

ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
ordinal_encoder.fit(df[['Author', 'Genre']])

new_book = {
    'Author_encoded': ordinal_encoder.transform([['J.K. Rowling', 'Fiction']])[0][0],
    'Reviews': 0,
    'User Rating': 0,
    'Year': 2013,
    'Genre_encoded': ordinal_encoder.transform([['J.K. Rowling', 'Fiction']])[0][1],
    'Price': 19.99
}

new_book_df = pd.DataFrame([new_book])
new_book_scaled = scaler.transform(new_book_df)

predicted_sales = rf_model.predict(new_book_scaled)
print(f"Predicted Sales for the new book: {predicted_sales[0]}")