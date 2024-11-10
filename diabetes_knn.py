import pandas as pd
import missingno as msno
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score

df = pd.read_csv('diabetes.csv')
print("Columns in dataset:", df.columns)

missing_values = df.isnull().sum()
missing_percentage = (missing_values / len(df)) * 100
print("Missing values percentage:\n", missing_percentage)

msno.matrix(df)
plt.show()
sns.heatmap(df.isnull(), cbar=False, cmap="viridis")
plt.show()

df.fillna(df.median(), inplace=True)

df.to_csv('diabetes_cleaned.csv', index=False)

print("First rows of cleaned dataset:\n", df.head())

X = df.drop(columns=['Outcome'])
y = df['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
print("Accuracy:", accuracy)
print("Classification Report:\n", report)
