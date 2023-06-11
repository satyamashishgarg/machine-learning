import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load the dataset
df = pd.read_csv('data.csv')

# Split the dataset into features (X) and target variable (y)
X = df[['feature1', 'feature2', 'feature3']]
y = df['target']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a linear regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
score = model.score(X_test, y_test)
print(f"R-squared score: {score}")

# You can now use the trained model to make predictions on new data
new_data = [[3.5, 4.2, 2.1], [2.8, 3.9, 1.5]]
predictions = model.predict(new_data)
print(f"Predictions: {predictions}")
