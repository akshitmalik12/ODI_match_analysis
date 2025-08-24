# backend/app.py
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# Load and preprocess dataset
data = pd.read_csv('updated_odi_cricket.csv')

data['Ground Name'] = data['Ground Name'].str.strip().str.title()
data['Weather Conditions'] = data['Weather Conditions'].str.strip().str.title()
data['Pitch Conditions'] = data['Pitch Conditions'].str.strip().str.title()

label_encoder_ground = LabelEncoder()
label_encoder_weather = LabelEncoder()
label_encoder_pitch = LabelEncoder()
label_encoder_winner = LabelEncoder()

for col, encoder in zip(['Ground Name', 'Weather Conditions', 'Pitch Conditions', 'Winner'],
                       [label_encoder_ground, label_encoder_weather, label_encoder_pitch, label_encoder_winner]):
    data[col] = encoder.fit_transform(data[col])

data['Choice'] = np.where(data['Team 1 Score'] > data['Team 2 Score'], 'bat', 'bowl')
label_encoder_choice = LabelEncoder()
data['Choice'] = label_encoder_choice.fit_transform(data['Choice'])

# Train models
X_cond = data[['Ground Name', 'Weather Conditions', 'Pitch Conditions']]
y_choice = data['Choice']
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_cond, y_choice, test_size=0.2, random_state=42)
classifier = RandomForestClassifier(random_state=42)
classifier.fit(X_train_c, y_train_c)

y_score = data['Team 1 Score']
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_cond, y_score, test_size=0.2, random_state=42)
regressor = RandomForestRegressor(random_state=42)
regressor.fit(X_train_r, y_train_r)

pitch_condition_insights = {
    'Dry': "Dry pitches generally provide assistance to spinners and slow bowlers.",
    'Flat': "Flat pitches are good for batting, with minimal assistance for bowlers.",
    'Grassy': "Grassy pitches offer movement for pacers with the new ball.",
    'Wet': "Wet pitches assist fast bowlers with bounce and swing."
}

ground_insights = {
    "Chinnaswamy Stadium": "Batting-friendly pitches with high scores.",
    "Wankhede Stadium": "Pitch offers extra bounce and some help for pacers.",
    "Feroz Shah Kotla": "Slow, low pitches favoring spinners.",
    "Mcg": "Assists both fast bowlers and spinners with unpredictable bounce.",
    "Lord'S": "Pitch assists fast bowlers and spinners with slope influence.",
    "Eden Gardens": "Flat pitch conducive to high scoring.",
    "Narendra Modi Stadium": "True bounce, favors batsmen early in matches.",
    "The Oval": "Assists both fast bowlers and spinners.",
    "Scg": "Offers swing early and batting friendly later.",
    "Trent Bridge": "Assists both fast bowlers and spinners."
}

# Flask app initialization
app = Flask(__name__)
CORS(app)  # Enable CORS for all domains (adjust in production)

@app.route('/')
def home():
    return jsonify({"message": "Cricket Match Prediction API"})

@app.route('/categories', methods=['GET'])
def get_categories():
    return jsonify({
        "grounds": label_encoder_ground.classes_.tolist(),
        "weathers": label_encoder_weather.classes_.tolist(),
        "pitches": label_encoder_pitch.classes_.tolist()
    })

@app.route('/predict', methods=['POST'])
def predict():
    data_input = request.get_json()
    ground_input = data_input.get("ground", "").strip().title()
    weather_input = data_input.get("weather", "").strip().title()
    pitch_input = data_input.get("pitch", "").strip().title()

    print("Received inputs:", ground_input, weather_input, pitch_input)
    print("Ground encoder classes:", label_encoder_ground.classes_)
    print("Weather encoder classes:", label_encoder_weather.classes_)
    print("Pitch encoder classes:", label_encoder_pitch.classes_)

    try:
        ground_encoded = label_encoder_ground.transform([ground_input])[0]
        weather_encoded = label_encoder_weather.transform([weather_input])  # <-- Fix: missing 
        pitch_encoded = label_encoder_pitch.transform([pitch_input])        # <-- Fix: missing 
    except Exception as e:
        print("Encoding error:", e)
        return jsonify({"error": "Invalid input values for ground, weather, or pitch."}), 400

    # Prepare a DataFrame with feature names as expected by the model
    import pandas as pd
    X_test = pd.DataFrame(
        [[ground_encoded, weather_encoded, pitch_encoded]],
        columns=['Ground Name', 'Weather Conditions', 'Pitch Conditions']
    )

    decision_encoded = classifier.predict(X_test)[0]
    decision = label_encoder_choice.inverse_transform([decision_encoded])

    predicted_score = regressor.predict(X_test)[0]

    total_matches = len(data[data['Ground Name'] == ground_encoded])
    matches_won_bat = len(data[(data['Ground Name'] == ground_encoded) & (data['Choice'] == 1)])
    matches_won_bowl = len(data[(data['Ground Name'] == ground_encoded) & (data['Choice'] == 0)])

    pitch_insight = pitch_condition_insights.get(pitch_input, "No specific pitch insights available.")
    ground_insight = ground_insights.get(ground_input, "No specific ground insights available.")

    return jsonify({
        "recommended_choice": decision[0],
        "predicted_score": round(predicted_score),
        "total_matches": total_matches,
        "matches_won_batting_first": matches_won_bat,
        "matches_won_bowling_first": matches_won_bowl,
        "insights": {
            "pitch": pitch_insight,
            "ground": ground_insight
        }
    })


if __name__ == '__main__':
    app.run(debug=True)
