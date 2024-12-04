from flask import Flask, render_template, request, redirect, url_for, session
import numpy as np
import pickle
import torch

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Used for session management

# Dictionary to store users for login validation
users = {
    'admin': {'password': 'password'}
}

# Load your saved PyTorch model and scaler
model = torch.load('model.pth')
scaler = pickle.load(open('scaler.pkl', 'rb'))

# Home Route (protected by login)
@app.route('/')
def home():
    if 'logged_in' not in session:
        return redirect(url_for('login'))  # Redirect to login if not logged in
    return render_template('home.html')

# Login Route
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        # Check if the username exists and the password matches
        if username in users and users[username]['password'] == password:
            session['logged_in'] = True  # Set session to indicate user is logged in
            return redirect(url_for('home'))  # Redirect to home after successful login
        else:
            return "Invalid credentials. Please try again."
    
    return render_template('login.html')

# Signup Route
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        # Check if the username already exists
        if username in users:
            return "Username already exists. Please choose another one."
        
        # Save the user details (only username and password)
        users[username] = {'password': password}
        return redirect(url_for('login'))  # Redirect to login page after successful signup
    return render_template('signup.html')

# Logout Route
@app.route('/logout')
def logout():
    session.pop('logged_in', None)  # Remove logged_in from session
    return redirect(url_for('login'))  # Redirect to login page after logout

# Prediction Route
@app.route('/predict', methods=['POST'])
def predict():
    if 'logged_in' not in session:
        return redirect(url_for('login'))  # Redirect to login if not logged in
    
    if request.method == 'POST':
        # Get user inputs
        flight_number = request.form['flight_number']
        airport_code = request.form['airport_code']
        flight_date = request.form['flight_date']
        source = request.form['source']
        destination = request.form['destination']
        air_time = float(request.form['air_time'])
        distance = float(request.form['distance'])

        # Prepare the input for the model
        user_input = np.array([[air_time, distance]])
        user_input_scaled = scaler.transform(user_input)

        # Make prediction using the model
        predictions = model.predict(user_input_scaled)

        # Prepare the result based on the prediction
        if predictions[0][1] >= 0.5:
            prediction_result = f"The flight is delayed by {(predictions[0][0])} minutes."
        else:
            prediction_result = "The flight is not delayed."

        # Pass user details and prediction result to the template
        return render_template('result.html', 
                               prediction=prediction_result,
                               flight_number=flight_number,
                               airport_code=airport_code,
                               flight_date=flight_date,
                               source=source,
                               destination=destination,
                               air_time=air_time,
                               distance=distance)
    return render_template('home.html')

if __name__ == '__main__':
    app.run(debug=True)
