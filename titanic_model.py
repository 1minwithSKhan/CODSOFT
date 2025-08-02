import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
df = pd.read_csv("Titanic-Dataset.csv")  # Ensure this file is in the same folder
df_cleaned = df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'])
imputer = SimpleImputer(strategy='most_frequent')
df_cleaned[['Age', 'Embarked']] = imputer.fit_transform(df_cleaned[['Age', 'Embarked']])
le_sex = LabelEncoder()
df_cleaned['Sex'] = le_sex.fit_transform(df_cleaned['Sex'])
le_embarked = LabelEncoder()
df_cleaned['Embarked'] = le_embarked.fit_transform(df_cleaned['Embarked'])
X = df_cleaned.drop('Survived', axis=1)
y = df_cleaned['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
