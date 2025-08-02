import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
try:
    movie_df = pd.read_csv('IMDb Movies India2.csv', encoding='latin1')
except FileNotFoundError:
    print('Error: IMDb Movies India2.csv not found in the current directory.')
    exit(1)
except Exception as e:
    print(f'Error loading CSV: {e}')
    exit(1)
def explore_data(df):
    print('Columns:', df.columns)
    print('Missing values:', df.isnull().sum())
    print(df.head())
explore_data(movie_df)
features = ['Genre', 'Director', 'Actor 1', 'Actor 2', 'Actor 3']
target = 'Rating'
movie_df = movie_df.dropna(subset=[target])
for col in features:
    movie_df[col] = movie_df[col].fillna('Unknown')
X_cat = movie_df[features]
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
X_encoded = encoder.fit_transform(X_cat)
y = movie_df[target].values
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print(f'RMSE: {rmse:.2f}')
print(f'R2 Score: {r2:.2f}')
feature_names = encoder.get_feature_names_out(features)
importances = model.feature_importances_
indices = np.argsort(importances)[::-1][:10]
print('Top 10 Feature Importances:')
for i in indices:
    print(f'{feature_names[i]}: {importances[i]:.4f}')
def predict_new_movie(genre, director, actor1, actor2, actor3):
    new_df = pd.DataFrame({
        'Genre': [genre],
        'Director': [director],
        'Actor 1': [actor1],
        'Actor 2': [actor2],
        'Actor 3': [actor3]
    })
    new_encoded = encoder.transform(new_df)
    if np.all(new_encoded == 0):
        print('Warning: All input values are unknown to the model. Prediction may not be meaningful.')
    try:
        pred = model.predict(new_encoded)
        return pred[0]
    except Exception as e:
        print(f'Error during prediction: {e}')
        return None
result = predict_new_movie('Action', 'Christopher Nolan', 'Amitabh Bachchan', 'Shah Rukh Khan', 'Deepika Padukone')
if result is not None:
    print(f'Predicted Rating: {result:.2f}')