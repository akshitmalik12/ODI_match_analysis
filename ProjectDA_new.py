import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox

# Load dataset
data = pd.read_csv('/Users/akshit/Downloads/updated_odi_cricket.csv')

# Normalize ground, weather, and pitch condition columns
data['Ground Name'] = data['Ground Name'].str.strip().str.title()
data['Weather Conditions'] = data['Weather Conditions'].str.strip().str.title()
data['Pitch Conditions'] = data['Pitch Conditions'].str.strip().str.title()

# Encode categorical columns
label_encoder_ground = LabelEncoder()
label_encoder_weather = LabelEncoder()
label_encoder_pitch = LabelEncoder()
label_encoder_winner = LabelEncoder()

for col, encoder in zip(['Ground Name', 'Weather Conditions', 'Pitch Conditions', 'Winner'],
                         [label_encoder_ground, label_encoder_weather, label_encoder_pitch, label_encoder_winner]):
    data[col] = encoder.fit_transform(data[col])

# Feature for batting/bowling first decision
data['Choice'] = np.where(data['Team 1 Score'] > data['Team 2 Score'], 'bat', 'bowl')
label_encoder_choice = LabelEncoder()
data['Choice'] = label_encoder_choice.fit_transform(data['Choice'])

# Prepare data for classification
X_conditions = data[['Ground Name', 'Weather Conditions', 'Pitch Conditions']]
y_choice = data['Choice']
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_conditions, y_choice, test_size=0.2, random_state=42)
classifier = RandomForestClassifier(random_state=42)
classifier.fit(X_train_c, y_train_c)

# Prepare data for regression
X_score = data[['Ground Name', 'Weather Conditions', 'Pitch Conditions']]
y_score = data['Team 1 Score']
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_score, y_score, test_size=0.2, random_state=42)
regressor = RandomForestRegressor(random_state=42)
regressor.fit(X_train_r, y_train_r)

# Insights for pitch conditions
pitch_condition_insights = {
    'Dry': "Dry pitches generally provide assistance to spinners and slow bowlers, causing the ball to grip and turn.",
    'Flat': "Flat pitches are typically good for batting, with minimal assistance for bowlers. They tend to produce high-scoring matches.",
    'Grassy': "Grassy pitches offer movement for pacers, especially with the new ball. Bowlers can extract swing early in the match.",
    'Wet': "Wet pitches offer more assistance to fast bowlers, who can extract bounce and swing, making batting challenging."
}

# Insights for each ground
ground_insights = {
    "Chinnaswamy Stadium": "Chinnaswamy Stadium in Bengaluru is renowned for its batting-friendly pitches, offering true bounce and consistent carry. The ground's small boundaries often lead to high-scoring matches.",
    "Wankhede Stadium": "Wankhede Stadium in Mumbai features a pitch that offers extra bounce due to its red soil composition. The stadium's proximity to the sea allows pacers to extract some help with the new ball.",
    "Feroz Shah Kotla": "Feroz Shah Kotla Ground in Delhi is known for its slow and low pitches, which favor spinners. The dry conditions often lead to cracks, providing turn and bounce as the match progresses.",
    "Mcg": "The MCG offers a pitch that provides assistance to both fast bowlers and spinners. The large boundaries and unpredictable bounce add to the challenge for batsmen.",
    "Lord'S": "Lord's features a pitch that offers assistance to both fast bowlers and spinners. The ground's slope can influence the ball's movement, adding an element of unpredictability.",
    "Eden Gardens": "Eden Gardens offers a flat pitch conducive to high-scoring games, but the pitch can become slower as the game progresses, favoring spinners later on.",
    "Narendra Modi Stadium": "The Narendra Modi Stadium is renowned for its true bounce and even carry, ensuring a fair competition between bat and ball. Historically, the pitch has favored batsmen, especially in the early stages of matches.",
    "The Oval": "The Oval offers a pitch that provides assistance to both fast bowlers and spinners. The large boundaries and variable bounce add to the challenge for batsmen.",
    "Scg": "The SCG features a pitch that offers assistance to both fast bowlers and spinners. Early on, the pitch offers swing, but it becomes more batting-friendly as the match progresses.",
    "Trent Bridge": "Trent Bridge offers a pitch that provides assistance to both fast bowlers and spinners. The large boundaries and variable bounce add to the challenge for batsmen."
}

# Function to retrieve insights based on pitch and weather conditions
def get_insights(pitch, weather, ground):
    pitch_insight = pitch_condition_insights.get(pitch, "No specific pitch insights available.")
    ground_insight = ground_insights.get(ground, "No specific ground insights available.")
    return f"{pitch_insight}\n\n{ground_insight}"

def make_predictions():
    ground_input = ground_combobox.get().strip().title()
    weather_input = weather_combobox.get().strip().title()
    pitch_input = pitch_combobox.get().strip().title()

    try:
        ground_encoded = label_encoder_ground.transform([ground_input])[0]
        weather_encoded = label_encoder_weather.transform([weather_input])[0]
        pitch_encoded = label_encoder_pitch.transform([pitch_input])[0]
    except ValueError:
        messagebox.showerror("Error", "Invalid input. Please enter valid ground, weather, or pitch condition.")
        return

    decision = classifier.predict([[ground_encoded, weather_encoded, pitch_encoded]])[0]
    decision_label = label_encoder_choice.inverse_transform([decision])[0]
    predicted_score = regressor.predict([[ground_encoded, weather_encoded, pitch_encoded]])[0]

    total_matches = len(data[data['Ground Name'] == ground_encoded])
    matches_won_bat_first = len(data[(data['Ground Name'] == ground_encoded) & (data['Choice'] == 1)])
    matches_won_bowl_first = len(data[(data['Ground Name'] == ground_encoded) & (data['Choice'] == 0)])

    # Create a bar graph and histogram side by side
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    # Histogram
    ax[0].hist(data['Team 1 Score'], bins=20, color='skyblue', edgecolor='black')
    ax[0].set_title('Distribution of Team 1 Scores')
    ax[0].set_xlabel('Team 1 Score')
    ax[0].set_ylabel('Frequency')

    # Bar graph
    match_history = data[data['Ground Name'] == ground_encoded].reset_index()
    match_outcomes = ["Batting First Win" if choice == 1 else "Bowling First Win" for choice in match_history['Choice']]
    colors = ['blue' if outcome == "Batting First Win" else 'green' for outcome in match_outcomes]
    ax[1].bar(range(total_matches), match_history['Team 1 Score'], color=colors)
    ax[1].set_title(f'Match History at {ground_input}')
    ax[1].set_xlabel('Match Index')
    ax[1].set_ylabel('Team 1 Score')
    ax[1].legend(['Batting First Win', 'Bowling First Win'])

    # Pie chart for batting first vs bowling first wins
    fig_pie, ax_pie = plt.subplots(figsize=(5, 5))
    win_counts = [matches_won_bat_first, matches_won_bowl_first]
    ax_pie.pie(win_counts, labels=['Batting First', 'Bowling First'], autopct='%1.1f%%', startangle=90, colors=['orange', 'green'])
    ax_pie.set_title('Win Proportions: Batting First vs Bowling First')

    # Clear previous widgets before displaying the new ones
    for widget in graph_frame.winfo_children():
        widget.destroy()

    # Add the bar graph and histogram to the Tkinter window
    canvas1 = FigureCanvasTkAgg(fig, master=graph_frame)
    canvas1.draw()
    canvas1.get_tk_widget().pack()

    # Add the pie chart to the Tkinter window
    canvas2 = FigureCanvasTkAgg(fig_pie, master=graph_frame)
    canvas2.draw()
    canvas2.get_tk_widget().pack()

    insights_message = get_insights(pitch_input, weather_input, ground_input)

    result_label.config(text=(
        f"Statistics:\n"
        f"Total Matches: {total_matches}\n"
        f"Matches Won Batting First: {matches_won_bat_first}\n"
        f"Matches Won Bowling First: {matches_won_bowl_first}\n\n"
        f"Recommended Choice: {decision_label}\n"
        f"Predicted Team 1 Score: {predicted_score:.0f}\n"
        f"Insights: {insights_message}"
    ))

# Create the GUI window
root = tk.Tk()
root.title("Cricket Prediction")
root.geometry("800x800")

# GUI Components
ground_label = tk.Label(root, text="Select Ground Name:", font=("Arial", 12))
ground_label.pack(pady=5)
ground_combobox = ttk.Combobox(root, values=label_encoder_ground.classes_.tolist())
ground_combobox.pack(pady=5)

weather_label = tk.Label(root, text="Select Weather Condition:", font=("Arial", 12))
weather_label.pack(pady=5)
weather_combobox = ttk.Combobox(root, values=label_encoder_weather.classes_.tolist())
weather_combobox.pack(pady=5)

pitch_label = tk.Label(root, text="Select Pitch Condition:", font=("Arial", 12))
pitch_label.pack(pady=5)
pitch_combobox = ttk.Combobox(root, values=label_encoder_pitch.classes_.tolist())
pitch_combobox.pack(pady=5)

predict_button = tk.Button(root, text="Make Prediction", command=make_predictions)
predict_button.pack(pady=10)

result_label = tk.Label(root, text="", font=("Arial", 12))
result_label.pack(pady=10)

graph_frame = tk.Frame(root)
graph_frame.pack(pady=20)

root.mainloop()
