import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tkinter import Tk, Label, Frame, Button, StringVar, OptionMenu, Toplevel, messagebox
from math import sqrt

# Load and Prepare the Dataset
file_path = r"C:\Users\ROG\Desktop\finalized ml.xlsx"
data = pd.read_excel(file_path)

# Feature Engineering
data['GoalDifference'] = data['HomeTeamGoals'] - data['AwayTeamGoals']
data['Outcome'] = data.apply(lambda row: 'Home Win' if row['HomeTeamGoals'] > row['AwayTeamGoals'] 
                             else 'Away Win' if row['HomeTeamGoals'] < row['AwayTeamGoals'] else 'Draw', axis=1)

# Display Correlation Matrix Separately
def show_correlation_matrix():
    numeric_data = data.select_dtypes(include=[np.number])
    plt.figure(figsize=(10, 8))
    correlation_matrix = numeric_data.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title("Correlation Matrix")
    plt.show()

# Prepare data for regression
X = data[['HomeTeam Form', 'AwayTeam Form', 'HomeTeam Keyplayers', 'AwayTeam Keyplayers', 'HomeTeam Injuries', 'AwayTeam Injuries', 'GoalDifference']]
y_home_goals = data['HomeTeamGoals']
y_away_goals = data['AwayTeamGoals']

# Split the data
X_train, X_test, y_train_home, y_test_home = train_test_split(X, y_home_goals, test_size=0.2, random_state=42)
_, _, y_train_away, y_test_away = train_test_split(X, y_away_goals, test_size=0.2, random_state=42)

# Train Random Forest Models for Home and Away Goals
rf_home = RandomForestRegressor(n_estimators=100, random_state=42)
rf_away = RandomForestRegressor(n_estimators=100, random_state=42)

rf_home.fit(X_train, y_train_home)
rf_away.fit(X_train, y_train_away)

# Evaluate Model Performance
y_pred_home = rf_home.predict(X_test)
y_pred_away = rf_away.predict(X_test)

mae_home = mean_absolute_error(y_test_home, y_pred_home)
mse_home = mean_squared_error(y_test_home, y_pred_home)
rmse_home = sqrt(mse_home)
r2_home = r2_score(y_test_home, y_pred_home)

mae_away = mean_absolute_error(y_test_away, y_pred_away)
mse_away = mean_squared_error(y_test_away, y_pred_away)
rmse_away = sqrt(mse_away)
r2_away = r2_score(y_test_away, y_pred_away)

# Display Model Performance and Feature Importance in New Window
def show_model_performance():
    performance_window = Toplevel(root)
    performance_window.title("Model Performance Metrics and Explainability")
    performance_window.geometry("500x400")
    performance_window.configure(bg="#e0f7fa")

    Label(performance_window, text="Random Forest Regressor Performance Metrics", font=("Helvetica", 14), bg="#e0f7fa").pack(pady=10)
    Label(performance_window, text=f"Mean Absolute Error (Home Goals): {mae_home:.4f}", font=("Helvetica", 12), bg="#e0f7fa").pack()
    Label(performance_window, text=f"Mean Squared Error (Home Goals): {mse_home:.4f}", font=("Helvetica", 12), bg="#e0f7fa").pack()
    Label(performance_window, text=f"Root Mean Squared Error (Home Goals): {rmse_home:.4f}", font=("Helvetica", 12), bg="#e0f7fa").pack()
    Label(performance_window, text=f"R^2 Score (Home Goals): {r2_home:.4f}", font=("Helvetica", 12), bg="#e0f7fa").pack()
    Label(performance_window, text=f"Mean Absolute Error (Away Goals): {mae_away:.4f}", font=("Helvetica", 12), bg="#e0f7fa").pack()
    Label(performance_window, text=f"Mean Squared Error (Away Goals): {mse_away:.4f}", font=("Helvetica", 12), bg="#e0f7fa").pack()
    Label(performance_window, text=f"Root Mean Squared Error (Away Goals): {rmse_away:.4f}", font=("Helvetica", 12), bg="#e0f7fa").pack()
    Label(performance_window, text=f"R^2 Score (Away Goals): {r2_away:.4f}", font=("Helvetica", 12), bg="#e0f7fa").pack(pady=5)

    # Feature Importance Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    feature_importances = rf_home.feature_importances_
    features = X.columns
    sns.barplot(x=feature_importances, y=features, ax=ax, palette="viridis")
    ax.set_title("Feature Importances for Home Goals Prediction")
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    plt.tight_layout()
    plt.show()

# GUI for User Input and Prediction
def predict():
    try:
        # Get selected teams
        home_team = home_team_var.get()
        away_team = away_team_var.get()

        # Check if home and away teams are the same
        if home_team == away_team:
            raise ValueError("Home Team and Away Team names cannot be the same. Please select different teams.")

        # Retrieve team data for prediction
        home_team_data = data[data['HomeTeamName'] == home_team].iloc[0]
        away_team_data = data[data['AwayTeamName'] == away_team].iloc[0]

        # Prepare input for prediction
        input_data = np.array([[home_team_data['HomeTeam Form'], away_team_data['AwayTeam Form'],
                                home_team_data['HomeTeam Keyplayers'], away_team_data['AwayTeam Keyplayers'],
                                home_team_data['HomeTeam Injuries'], away_team_data['AwayTeam Injuries'],
                                home_team_data['GoalDifference']]])

        # Predict scores
        predicted_home_goals = round(rf_home.predict(input_data)[0])
        predicted_away_goals = round(rf_away.predict(input_data)[0])

        # Determine outcome for the home team
        if predicted_home_goals > predicted_away_goals:
            outcome_statement = "Home Team Wins!"
        elif predicted_home_goals < predicted_away_goals:
            outcome_statement = "Home Team Loses!"
        else:
            outcome_statement = "It's a Draw!"

        # Display prediction result and outcome
        result_text.set(f"Predicted Scoreline:\nHome Team {predicted_home_goals} - Away Team {predicted_away_goals}\n{outcome_statement}")
        
    except ValueError as ve:
        messagebox.showerror("Error", str(ve))
    except Exception as e:
        messagebox.showerror("Error", "An unexpected error occurred: " + str(e))

# GUI Setup
root = Tk()
root.title("Football Match Scoreline Predictor")
root.geometry("500x600")
root.configure(bg="#e0f7fa")

# Header Label
header_label = Label(root, text="Football Match Scoreline Predictor", font=("Helvetica", 18, "bold"), fg="#004d40", bg="#e0f7fa")
header_label.pack(pady=20)

# Team Selection Frame
team_frame = Frame(root, bg="#e0f7fa")
team_frame.pack(pady=10)

# Team selection dropdowns
home_team_var = StringVar()
away_team_var = StringVar()
home_team_var.set("Select Home Team")
away_team_var.set("Select Away Team")

Label(team_frame, text="Home Team:", font=("Helvetica", 12), bg="#e0f7fa").grid(row=0, column=0, padx=5, pady=5)
home_team_dropdown = OptionMenu(team_frame, home_team_var, *data['HomeTeamName'].unique())
home_team_dropdown.config(font=("Helvetica", 12))
home_team_dropdown.grid(row=0, column=1, padx=5, pady=5)

Label(team_frame, text="Away Team:", font=("Helvetica", 12), bg="#e0f7fa").grid(row=1, column=0, padx=5, pady=5)
away_team_dropdown = OptionMenu(team_frame, away_team_var, *data['AwayTeamName'].unique())
away_team_dropdown.config(font=("Helvetica", 12))
away_team_dropdown.grid(row=1, column=1, padx=5, pady=5)

# Prediction button
predict_button = Button(root, text="Predict Scoreline", command=predict, font=("Helvetica", 14), bg="#00796b", fg="white")
predict_button.pack(pady=20)

# Show Model Performance button
performance_button = Button(root, text="Show Model Performance", command=show_model_performance, font=("Helvetica", 12), bg="#0288d1", fg="white")
performance_button.pack(pady=5)



# Show Correlation Matrix button
correlation_button = Button(root, text="Show Correlation Matrix", command=show_correlation_matrix, font=("Helvetica", 12), bg="#0288d1", fg="white")
correlation_button.pack(pady=5)

# Exit Button
exit_button = Button(root, text="Exit", command=root.quit, font=("Helvetica", 12), bg="#d32f2f", fg="white")
exit_button.pack(pady=5)

# Result Label for displaying prediction
result_text = StringVar()
result_label = Label(root, textvariable=result_text, font=("Helvetica", 12), fg="#004d40", bg="#e0f7fa", justify="center")
result_label.pack(pady=20)

# Run the GUI
root.mainloop()

