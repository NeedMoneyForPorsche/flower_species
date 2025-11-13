import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import joblib

df = pd.read_csv('Irisflower.csv')

df.dropna(inplace=True)

X = df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
y = df['Species']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=42)

model = DecisionTreeClassifier()
model.fit(X_train, y_train)

joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("Successful!")
