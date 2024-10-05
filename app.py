from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import datetime as dt

app = Flask(__name__)

# Load the trained model
model = joblib.load('rf_tuned_model.pkl')

# Extract date-related features
def get_date_features(date_str):
    date = dt.datetime.strptime(date_str, '%Y-%m-%d')
    period_of_day = get_period_of_day(date)
    return {
        'DayOfWeek': date.weekday(),
        'WeekOfYear': date.isocalendar()[1],
        'Day': date.day,
        'Month': date.month,
        'Year': date.year,
        'PeriodOfDay': period_of_day
    }

# Custom function to return period of day
def get_period_of_day(date):
    hour = 12  # Placeholder for hour of the day, adjust based on your data
    minute = 0
    period_of_day = (hour * 2) + (minute // 30)
    return period_of_day

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        selected_holiday = request.form['Holiday']
        holiday_flag = 0 if selected_holiday == "None" else 1

        # Handle date feature extraction
        date_str = request.form['Date']
        date_features = get_date_features(date_str)

        # Extract other input features from the form
        input_features = [
            float(request.form['ForecastWindProduction']),
            float(request.form['SystemLoadEA']),
            float(request.form['SMPEA']),
            float(request.form['ORKTemperature']),
            float(request.form['ORKWindspeed']),
            float(request.form['CO2Intensity']),
            float(request.form['ActualWindProduction']),
            float(request.form['SystemLoadEP2'])
        ]

        # Combine all features
        features = np.array([
            date_features['DayOfWeek'],
            date_features['WeekOfYear'],
            date_features['Day'],
            date_features['Month'],
            date_features['Year'],
            date_features['PeriodOfDay'],
            holiday_flag,
            *input_features
        ]).reshape(1, -1)

        # Predict using the trained model
        prediction = model.predict(features)
        prediction_value = float(prediction[0])

        # Return the prediction as JSON
        return jsonify({'prediction': f'{prediction_value:.2f}'})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == "__main__":
    app.run(debug=True)

