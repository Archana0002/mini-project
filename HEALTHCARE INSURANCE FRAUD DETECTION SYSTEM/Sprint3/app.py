import pickle
import numpy as np
from flask import Flask, request, jsonify, render_template
import joblib
from datetime import datetime
import pandas as pd
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# # Load your pre-trained ML model (pickled)
# with open('/home/alignminds/Desktop/Collab/Model/random_forest.pkl', 'rb') as model_file:
#     model = pickle.load(model_file)

model = joblib.load('C:\\Users\\ARCHANA\\OneDrive\\Desktop\\paper\\App1\\Model\\random_forest.pkl')

# # Utility function to calculate age
# def calculate_age(dob):
#     """Calculate age based on date of birth."""
#     today = datetime.today()
#     dob = pd.to_datetime(dob)
#     return today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))




# # Preprocessing function

def preprocess_input(user_input):
    # Convert input to DataFrame for consistency
    input_df = pd.DataFrame([user_input])

    # Convert date strings to datetime objects
    for date_col in ['ClaimStartDt', 'ClaimEndDt', 'AdmissionDt', 'DischargeDt', 'DOB', 'DOD']:
        input_df[date_col] = pd.to_datetime(input_df[date_col], format='%d-%m-%Y', errors='coerce')

    # Feature Engineering: Add derived features
    input_df['ClaimDuration'] = (input_df['ClaimEndDt'] - input_df['ClaimStartDt']).dt.days
    input_df['AdmissionYear'] = input_df['AdmissionDt'].dt.year
    input_df['DischargeYear'] = input_df['DischargeDt'].dt.year
    input_df['AgeAtAdmission'] = (input_df['AdmissionDt'] - input_df['DOB']).dt.days // 365

    # Drop the original datetime columns if not needed
    input_df = input_df.drop(columns=['ClaimStartDt', 'ClaimEndDt', 'AdmissionDt', 'DischargeDt', 'DOB', 'DOD'])

    # Encode categorical variables using LabelEncoder
    categorical_cols = ['BeneID', 'ClaimID', 'Provider', 'RenalDiseaseIndicator']
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        input_df[col] = le.fit_transform(input_df[col])
        label_encoders[col] = le  # Store the encoder for future use

    # Ensure the DataFrame is in the correct order of features for the model
    feature_order = [
        'BeneID', 'ClaimID', 'Provider', 'InscClaimAmtReimbursed', 'DeductibleAmtPaid',
        'Gender', 'Race', 'RenalDiseaseIndicator', 'State', 'County',
        'NoOfMonths_PartACov', 'NoOfMonths_PartBCov', 
        'IPAnnualReimbursementAmt', 'IPAnnualDeductibleAmt',
        'OPAnnualReimbursementAmt', 'OPAnnualDeductibleAmt', 'Age', 
        'ClaimDuration', 'AdmissionYear', 'DischargeYear', 'AgeAtAdmission'
    ]

    input_df = input_df[feature_order]

    return input_df





# Function to predict fraud
def predict_fraud(preprocessed_input):
    """Make predictions using the trained model."""
    prediction = model.predict(preprocessed_input)
    return prediction




# Route to render the HTML page
@app.route('/')
def index():
    return render_template('index.html')

# Flask route for predictions
@app.route('/submit_claim', methods=['POST'])
def get_prediction():
    try:
        # Get user input from request
        # user_input = request.json
        # print("User Inputs are...",user_input)


        user_input = {key: request.form[key] for key in request.form}
        print("User Inputs are...", user_input)

        preprocessed_data = preprocess_input(user_input)
        print("preprocessed data...",preprocessed_data)
        
        # Make prediction
        prediction = predict_fraud(preprocessed_data)
        print("Prediction Result is...",prediction)
        
        # Return the result
        result = "Fraud" if prediction[0] == 1 else "Not Fraud"
        return jsonify({'The Insurance claim is ': result})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
