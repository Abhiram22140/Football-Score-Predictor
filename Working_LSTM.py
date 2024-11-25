import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# Load and preprocess your data
data = pd.read_csv(r"C:\Users\ROG\Desktop\SEM-5\ML\project codes\final\Updated_Dataset_with_Goal_Difference.csv")

# Step 1: Encode `round` as ordinal values
round_mapping = {
    "GROUP_STANDINGS": 1,
    "QUARTER_FINALS": 3,
    "THIRD_PLAY_OFF": 2,
    "SEMIFINAL": 4,
    "FINAL": 5
}
data["round_ordinal"] = data["round"].map(round_mapping)

# Step 2: One-Hot Encode the `round` column
round_one_hot = pd.get_dummies(data["round"], prefix="round")
data = pd.concat([data, round_one_hot], axis=1)

# Drop the original `round` column
data.drop(columns=["round"], inplace=True)

# Step 3: Handle potential missing values
data.dropna(inplace=True)

# Time-based features transformation (sine and cosine)
data['hour_sin'] = np.sin(2 * np.pi * data['hour'] / 24)
data['hour_cos'] = np.cos(2 * np.pi * data['hour'] / 24)
data['month_sin'] = np.sin(2 * np.pi * data['month'] / 12)
data['month_cos'] = np.cos(2 * np.pi * data['month'] / 12)
data['day_of_week_sin'] = np.sin(2 * np.pi * data['day_of_week'] / 7)
data['day_of_week_cos'] = np.cos(2 * np.pi * data['day_of_week'] / 7)

# Select relevant features for input
features = ['hour_sin', 'hour_cos', 'month_sin', 'month_cos', 
            'day_of_week_sin', 'day_of_week_cos', 'match_attendance', 
            'stadium_capacity', 'goal_difference', 'home_team_form', 
            'away_team_form', 'HomeTeam Injuries', 'AwayTeam Injuries', 
            'round_ordinal'] + list(round_one_hot.columns)  # Include round features

# Target variables
target = ['home_score', 'away_score']

# Normalize the features
scaler = MinMaxScaler()
data[features] = scaler.fit_transform(data[features])

# Prepare features (X) and target (y)
X = data[features]
y = data[target]

# Reshape the data for LSTM
def create_sequences(X, y, sequence_length=5):
    X_seq, y_seq = [], []
    for i in range(len(X) - sequence_length):
        X_seq.append(X.iloc[i:i+sequence_length].values)
        y_seq.append(y.iloc[i+sequence_length].values)
    return np.array(X_seq), np.array(y_seq)

# Create sequences for all previous matches (not just one)
sequence_length = 5
X_seq, y_seq = create_sequences(X, y, sequence_length=sequence_length)

# Split the data into training and test sets (use temporal split)
X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, shuffle=False)

# Build the LSTM model
model = Sequential()
model.add(LSTM(units=50, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dense(2))  # Two outputs: home_score and away_score

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

# Model summary
model.summary()

# Evaluate model performance
y_pred = model.predict(X_test)

# Handle potential NaN in predictions
y_pred = np.nan_to_num(y_pred)

# Convert predictions and true values back to original scale
y_test_original = y_test
y_pred_original = y_pred

# Calculate performance metrics
mae = mean_absolute_error(y_test_original, y_pred_original)
mse = mean_squared_error(y_test_original, y_pred_original)
rmse = np.sqrt(mse)
r2 = r2_score(y_test_original, y_pred_original)

# Print performance metrics
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R² Score: {r2:.2f}")

# Plotting function for Actual vs Predicted Scores
def plot_predictions(y_test, y_pred):
    plt.figure(figsize=(10, 6))
    # Plot for Home Scores
    plt.plot(y_test[:, 0], label="Actual Home Scores", marker='o', linestyle='-', color='blue')
    plt.plot(y_pred[:, 0], label="Predicted Home Scores", marker='x', linestyle='--', color='orange')
    plt.title("Actual vs Predicted Home Scores", fontsize=16)
    plt.xlabel("Match Index", fontsize=12)
    plt.ylabel("Home Score", fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 6))
    # Plot for Away Scores
    plt.plot(y_test[:, 1], label="Actual Away Scores", marker='o', linestyle='-', color='green')
    plt.plot(y_pred[:, 1], label="Predicted Away Scores", marker='x', linestyle='--', color='red')
    plt.title("Actual vs Predicted Away Scores", fontsize=16)
    plt.xlabel("Match Index", fontsize=12)
    plt.ylabel("Away Score", fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

# Generate the plots
plot_predictions(y_test_original, y_pred_original)

# Function to predict home and away scores given home_team and away_team names
def predict_score(home_team_input, away_team_input):
    historical_data = data[(data['home_team'] == home_team_input) & (data['away_team'] == away_team_input)]
    num_matches = len(historical_data)
    if num_matches == 0:
        print(f"No historical data found for the match between {home_team_input} and {away_team_input}.")
        return None
    latest_data = historical_data.tail(min(sequence_length, num_matches))
    input_features = latest_data[features].values.reshape(1, len(latest_data), -1)
    predicted_scores = model.predict(input_features)
    predicted_home_score = round(predicted_scores[0][0])
    predicted_away_score = round(predicted_scores[0][1])
    return predicted_home_score, predicted_away_score

# User input for predictions
home_team_input = input("Enter home team name: ")
away_team_input = input("Enter away team name: ")
predicted_scores = predict_score(home_team_input, away_team_input)

if predicted_scores is not None:
    predicted_home_score, predicted_away_score = predicted_scores
    print(f"Predicted Home Score for {home_team_input}: {predicted_home_score}")
    print(f"Predicted Away Score for {away_team_input}: {predicted_away_score}")
else:
    print("Prediction could not be made due to insufficient historical data.")
