import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


sns.set(style="whitegrid")


df = pd.read_csv("dataset/Ecommerce Customers")

print("first 5 data:\n")
print(df.head())
print("\n Column Info:")
print(df.info())
print("\n Summary Statistics:")
print(df.describe())


plt.figure(figsize=(10, 6))
sns.heatmap(df.select_dtypes(include='number').corr(), annot=True, cmap='coolwarm', fmt=".2f")

plt.title("Correlation Heatmap")
plt.show()


selected_features = ['Avg. Session Length', 'Time on App', 'Time on Website',
                     'Length of Membership', 'Yearly Amount Spent']
sns.pairplot(df[selected_features])
plt.suptitle("Pairwise Feature Relationships", y=1.02)
plt.show()


sns.lmplot(x='Length of Membership', y='Yearly Amount Spent', data=df)
plt.title("Strongest Relationship: Length of Membership vs. Spending")
plt.show()

X = df[['Avg. Session Length', 'Time on App', 'Time on Website', 'Length of Membership']]
y = df['Yearly Amount Spent']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)


mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)


print("\n Model Equation Details:")
print(f"Intercept: {model.intercept_:.2f}")
for feature, coef in zip(X.columns, model.coef_):
    print(f"  {feature}: {coef:.2f}")

print("\n Model Evaluation Metrics:")
print(f"  Mean Absolute Error (MAE): {mae:.2f}")
print(f"  Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"  R² Score (Accuracy): {r2:.4f}")


plt.figure(figsize=(8,5))
plt.scatter(y_test, y_pred, color='darkblue', alpha=0.6)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.xlabel("Actual Yearly Amount Spent")
plt.ylabel("Predicted Yearly Amount Spent")
plt.title("Actual vs Predicted Spending")
plt.grid(True)
plt.show()
