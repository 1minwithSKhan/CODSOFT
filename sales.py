import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
try:
    df = pd.read_csv('advertising (1).csv', encoding='latin1')
except FileNotFoundError:
    print('Error: advertising (1).csv not found in the current directory.')
    exit(1)
except Exception as e:
    print(f'Error loading CSV: {e}')
    exit(1)
print('Columns:', df.columns)
print('Missing values:', df.isnull().sum())
print(df.head())
features = ['TV', 'Radio', 'Newspaper'] 
target = 'Sales' 
for col in features:
    df[col] = df[col].fillna(df[col].mean())
df[target] = df[target].fillna(df[target].mean())
X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print(f'RMSE: {rmse:.2f}')
print(f'R2 Score: {r2:.2f}')
def predict_sales(tv, radio, newspaper):
    new_data = pd.DataFrame({
        'TV': [tv],
        'Radio': [radio],
        'Newspaper': [newspaper]
    })
    return model.predict(new_data)[0]
