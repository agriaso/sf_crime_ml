import requests
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Replace 'your_api_endpoint' with the actual API endpoint
api_endpoint = "https://data.sfgov.org/resource/wg3w-h783.json?$limit=100"

# Make a request to the API
response = requests.get(api_endpoint)

# Check if the request was successful (status code 200)
if response.status_code == 200:
    df = pd.read_json("https://data.sfgov.org/resource/wg3w-h783.json?$limit=100")
    print(df.head())
    # Extract features and target variable
    features = df["incident_time"]  # Replace with actual feature names
    target = df[df["incident_category"] == "Larceny Theft"]  # Replace with actual target variable

    # Convert lists to numpy arrays
    X = features.to_numpy()
    y = target.to_numpy()

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a simple machine learning model (Random Forest Regressor in this example)
    model = RandomForestRegressor()
    model.fit(X_train, y_train)

    # Make predictions on the test set
    predictions = model.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, predictions)
    print(f'Mean Squared Error: {mse}')

else:
    print(f"Error: {response.status_code}")
