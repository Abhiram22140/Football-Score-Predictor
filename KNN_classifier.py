import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from tkinter import Tk, Label, Frame, Button, StringVar, OptionMenu, Toplevel, messagebox

# Load and Prepare the Dataset
file_path = r"C:\Users\ROG\Desktop\finalized ml.xlsx"
data = pd.read_excel(file_path)

# Feature Engineering
data['GoalDifference'] = data['HomeTeamGoals'] - data['AwayTeamGoals']
data['Outcome'] = data.apply(lambda row: 'Home Win' if row['HomeTeamGoals'] > row['AwayTeamGoals'] 
                             else 'Away Win' if row['HomeTeamGoals'] < row['AwayTeamGoals'] else 'Draw', axis=1)

# Mapping outcomes to numeric values for classification
outcome_mapping = {'Home Win': 1, 'Away Win': -1, 'Draw': 0}
data['Outcome_encoded'] = data['Outcome'].map(outcome_mapping)

# Display Correlation Matrix Separately
def show_correlation_matrix():
    numeric_data = data.select_dtypes(include=[np.number])
    plt.figure(figsize=(10, 8))
    correlation_matrix = numeric_data.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title("Correlation Matrix")
    plt.show()

# Prepare data for classification
X = data[['HomeTeam Form', 'AwayTeam Form', 'HomeTeam Keyplayers', 'AwayTeam Keyplayers', 'HomeTeam Injuries', 'AwayTeam Injuries', 'GoalDifference']]
y = data['Outcome_encoded']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train KNN Classifier
knn_classifier = KNeighborsClassifier(n_neighbors=5)
knn_classifier.fit(X_train, y_train)

# Evaluate Model Performance
y_pred = knn_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

# Display Model Performance Metrics in New Window
def show_model_performance():
    performance_window = Toplevel(root)
    performance_window.title("KNN Classifier Performance Metrics")
    performance_window.geometry("400x300")
    performance_window.configure(bg="#e0f7fa")

    Label(performance_window, text="KNN Classifier Performance Metrics", font=("Helvetica", 14), bg="#e0f7fa").pack(pady=10)
    Label(performance_window, text=f"Accuracy: {accuracy:.4f}", font=("Helvetica", 12), bg="#e0f7fa").pack()
    Label(performance_window, text=f"Precision: {precision:.4f}", font=("Helvetica", 12), bg="#e0f7fa").pack()
    Label(performance_window, text=f"Recall: {recall:.4f}", font=("Helvetica", 12), bg="#e0f7fa").pack()
    Label(performance_window, text=f"F1 Score: {f1:.4f}", font=("Helvetica", 12), bg="#e0f7fa").pack()

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

        # Predict outcome
        predicted_outcome_encoded = knn_classifier.predict(input_data)[0]
        
        # Decode the prediction
        outcome_decoding = {1: 'Home Win', -1: 'Away Win', 0: 'Draw'}
        predicted_outcome = outcome_decoding[predicted_outcome_encoded]

        # Display prediction result
        result_text.set(f"Predicted Outcome:\n{predicted_outcome}")
        
    except ValueError as ve:
        messagebox.showerror("Error", str(ve))
    except Exception as e:
        messagebox.showerror("Error", "An unexpected error occurred: " + str(e))

# GUI Setup
root = Tk()
root.title("Football Match Outcome Predictor")
root.geometry("500x500")
root.configure(bg="#e0f7fa")

# Header Label
header_label = Label(root, text="Football Match Outcome Predictor", font=("Helvetica", 18, "bold"), fg="#004d40", bg="#e0f7fa")
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
predict_button = Button(root, text="Predict Outcome", command=predict, font=("Helvetica", 14), bg="#00796b", fg="white")
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
