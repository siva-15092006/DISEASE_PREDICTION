# app.py
# Improved backend for the Medicine Recommendation System web app using Flask
# Enhanced with better error handling, logging, and improved Gemini API integration

from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import pandas as pd
import google.generativeai as genai
from flask_cors import CORS
import os
import logging
from datetime import datetime

app = Flask(__name__)
CORS(app)

# In-memory history storage (for session)
prediction_history = []

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Enhanced symptoms dictionary with more comprehensive mapping
symptoms_dict = {
    'itching': 0, 'skin_rash': 1, 'nodal_skin_eruptions': 2, 'continuous_sneezing': 3, 'shivering': 4, 'chills': 5, 
    'joint_pain': 6, 'stomach_pain': 7, 'acidity': 8, 'ulcers_on_tongue': 9, 'muscle_wasting': 10, 'vomiting': 11, 
    'burning_micturition': 12, 'spotting_urination': 13, 'fatigue': 14, 'weight_gain': 15, 'anxiety': 16, 
    'cold_hands_and_feets': 17, 'mood_swings': 18, 'weight_loss': 19, 'restlessness': 20, 'lethargy': 21, 
    'patches_in_throat': 22, 'irregular_sugar_level': 23, 'cough': 24, 'high_fever': 25, 'sunken_eyes': 26, 
    'breathlessness': 27, 'sweating': 28, 'dehydration': 29, 'indigestion': 30, 'headache': 31, 'yellowish_skin': 32, 
    'dark_urine': 33, 'nausea': 34, 'loss_of_appetite': 35, 'pain_behind_the_eyes': 36, 'back_pain': 37, 
    'constipation': 38, 'abdominal_pain': 39, 'diarrhoea': 40, 'mild_fever': 41, 'yellow_urine': 42, 
    'yellowing_of_eyes': 43, 'acute_liver_failure': 44, 'fluid_overload': 45, 'swelling_of_stomach': 46, 
    'swelled_lymph_nodes': 47, 'malaise': 48, 'blurred_and_distorted_vision': 49, 'phlegm': 50, 'throat_irritation': 51, 
    'redness_of_eyes': 52, 'sinus_pressure': 53, 'runny_nose': 54, 'congestion': 55, 'chest_pain': 56, 
    'weakness_in_limbs': 57, 'fast_heart_rate': 58, 'pain_during_bowel_movements': 59, 'pain_in_anal_region': 60, 
    'bloody_stool': 61, 'irritation_in_anus': 62, 'neck_pain': 63, 'dizziness': 64, 'cramps': 65, 'bruising': 66, 
    'obesity': 67, 'swollen_legs': 68, 'swollen_blood_vessels': 69, 'puffy_face_and_eyes': 70, 'enlarged_thyroid': 71, 
    'brittle_nails': 72, 'swollen_extremeties': 73, 'excessive_hunger': 74, 'extra_marital_contacts': 75, 
    'drying_and_tingling_lips': 76, 'slurred_speech': 77, 'knee_pain': 78, 'hip_joint_pain': 79, 'muscle_weakness': 80, 
    'stiff_neck': 81, 'swelling_joints': 82, 'movement_stiffness': 83, 'spinning_movements': 84, 'loss_of_balance': 85, 
    'unsteadiness': 86, 'weakness_of_one_body_side': 87, 'loss_of_smell': 88, 'bladder_discomfort': 89, 
    'foul_smell_of_urine': 90, 'continuous_feel_of_urine': 91, 'passage_of_gases': 92, 'internal_itching': 93, 
    'toxic_look_(typhos)': 94, 'depression': 95, 'irritability': 96, 'muscle_pain': 97, 'altered_sensorium': 98, 
    'red_spots_over_body': 99, 'belly_pain': 100, 'abnormal_menstruation': 101, 'dischromic_patches': 102, 
    'watering_from_eyes': 103, 'increased_appetite': 104, 'polyuria': 105, 'family_history': 106, 'mucoid_sputum': 107, 
    'rusty_sputum': 108, 'lack_of_concentration': 109, 'visual_disturbances': 110, 'receiving_blood_transfusion': 111, 
    'receiving_unsterile_injections': 112, 'coma': 113, 'stomach_bleeding': 114, 'distention_of_abdomen': 115, 
    'history_of_alcohol_consumption': 116, 'fluid_overload_1': 117, 'blood_in_sputum': 118, 'prominent_veins_on_calf': 119, 
    'palpitations': 120, 'painful_walking': 121, 'pus_filled_pimples': 122, 'blackheads': 123, 'scurring': 124, 
    'skin_peeling': 125, 'silver_like_dusting': 126, 'small_dents_in_nails': 127, 'inflammatory_nails': 128, 
    'blister': 129, 'red_sore_around_nose': 130, 'yellow_crust_ooze': 131
}

diseases_list = {
    15: 'Fungal infection', 4: 'Allergy', 16: 'GERD', 9: 'Chronic cholestasis', 14: 'Drug Reaction', 
    33: 'Peptic ulcer disease', 1: 'AIDS', 12: 'Diabetes', 17: 'Gastroenteritis', 6: 'Bronchial Asthma', 
    23: 'Hypertension', 30: 'Migraine', 7: 'Cervical spondylosis', 32: 'Paralysis (brain hemorrhage)', 
    28: 'Jaundice', 29: 'Malaria', 8: 'Chicken pox', 11: 'Dengue', 37: 'Typhoid', 40: 'Hepatitis A', 
    19: 'Hepatitis B', 20: 'Hepatitis C', 21: 'Hepatitis D', 22: 'Hepatitis E', 3: 'Alcoholic hepatitis', 
    36: 'Tuberculosis', 10: 'Common Cold', 34: 'Pneumonia', 13: 'Dimorphic hemorrhoids (piles)', 
    18: 'Heart attack', 39: 'Varicose veins', 26: 'Hypothyroidism', 24: 'Hyperthyroidism', 25: 'Hypoglycemia', 
    31: 'Osteoarthritis', 5: 'Arthritis', 0: '(vertigo) Paroxysmal Positional Vertigo', 2: 'Acne', 
    38: 'Urinary tract infection', 35: 'Psoriasis', 27: 'Impetigo'
}

# Common symptoms for easy selection
common_symptoms = [
    'fever', 'headache', 'cough', 'cold', 'sore throat', 'runny nose', 'body ache', 'fatigue',
    'nausea', 'vomiting', 'diarrhea', 'constipation', 'stomach pain', 'chest pain', 'back pain',
    'joint pain', 'muscle pain', 'dizziness', 'shortness of breath', 'loss of appetite',
    'weight loss', 'weight gain', 'skin rash', 'itching', 'sweating', 'chills', 'anxiety',
    'depression', 'insomnia', 'blurred vision'
]

# Symptom aliases for better matching
symptom_aliases = {
    'fever': ['high_fever', 'mild_fever'],
    'headache': ['headache'],
    'cold': ['common_cold', 'runny_nose', 'congestion'],
    'cough': ['cough'],
    'stomach_ache': ['stomach_pain', 'abdominal_pain', 'belly_pain'],
    'body_ache': ['muscle_pain', 'joint_pain'],
    'sore_throat': ['throat_irritation', 'patches_in_throat'],
    'breathing_problem': ['breathlessness'],
    'skin_problem': ['skin_rash', 'itching'],
    'eye_problem': ['redness_of_eyes', 'watering_from_eyes'],
    'urination_problem': ['burning_micturition', 'frequent_urination']
}

# Load ML model
try:
    with open('svc.pkl', 'rb') as model_file:
        svc = pickle.load(model_file)
    logger.info("ML model loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    svc = None

# Load databases with error handling
def load_csv_safe(filename, default_columns=None):
    try:
        df = pd.read_csv(filename)
        logger.info(f"Loaded {filename} successfully")
        return df
    except Exception as e:
        logger.error(f"Error loading {filename}: {e}")
        if default_columns:
            return pd.DataFrame(columns=default_columns)
        return pd.DataFrame()

sym_des = load_csv_safe("symtoms_df.csv")
precautions = load_csv_safe("precautions_df.csv")
workout = load_csv_safe("workout_df.csv")
description = load_csv_safe("description.csv")
medications = load_csv_safe('medications.csv')
diets = load_csv_safe("diets.csv")

def normalize_symptom(symptom):
    """Normalize symptom input for better matching"""
    symptom = symptom.lower().strip()
    symptom = symptom.replace(' ', '_')
    
    # Check direct match
    if symptom in symptoms_dict:
        return symptom
    
    # Check aliases
    for key, aliases in symptom_aliases.items():
        if symptom in [alias.lower() for alias in aliases]:
            return aliases[0]  # Return first alias
    
    # Check partial matches
    for known_symptom in symptoms_dict.keys():
        if symptom in known_symptom or known_symptom in symptom:
            return known_symptom
    
    return None

def helper(dis):
    """Enhanced helper function with better error handling"""
    try:
        desc = description[description['Disease'] == dis]['Description']
        desc = " ".join([w for w in desc]) if not desc.empty else "No description available."

        pre = precautions[precautions['Disease'] == dis][['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']]
        pre = [col for col in pre.values.flatten() if pd.notna(col)] if not pre.empty else ["No precautions available."]

        med = medications[medications['Disease'] == dis]['Medication']
        med = [m for m in med.values if pd.notna(m)] if not med.empty else ["No medications available."]

        die = diets[diets['Disease'] == dis]['Diet']
        die = [d for d in die.values if pd.notna(d)] if not die.empty else ["No diet recommendations available."]

        wrkout = workout[workout['disease'] == dis]['workout']
        wrkout = wrkout.tolist() if not wrkout.empty else ["No workout recommendations available."]

        return desc, pre, med, die, wrkout
    except Exception as e:
        logger.error(f"Error in helper function: {e}")
        return "Error retrieving information", ["N/A"], ["N/A"], ["N/A"], ["N/A"]

# Gemini API key
GEMINI_API_KEY = "API key"

@app.route('/')
def index():
    return render_template('index2.html', 
                         symptoms=list(symptoms_dict.keys())[:50],  # First 50 symptoms
                         common_symptoms=common_symptoms)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        symptoms_input = data.get('symptoms', '')
        
        if not symptoms_input:
            return jsonify({'error': 'No symptoms provided.'}), 400
        
        # Parse and normalize symptoms
        raw_symptoms = [s.strip().lower() for s in symptoms_input.split(',') if s.strip()]
        user_symptoms = []
        unknown_symptoms = []
        
        for symptom in raw_symptoms:
            normalized = normalize_symptom(symptom)
            if normalized:
                user_symptoms.append(normalized)
            else:
                unknown_symptoms.append(symptom)
        
        logger.info(f"Raw symptoms: {raw_symptoms}")
        logger.info(f"Normalized symptoms: {user_symptoms}")
        logger.info(f"Unknown symptoms: {unknown_symptoms}")
        
        # If we have some known symptoms, try ML model first
        if user_symptoms and svc:
            input_vector = np.zeros(len(symptoms_dict))
            for symptom in user_symptoms:
                if symptom in symptoms_dict:
                    input_vector[symptoms_dict[symptom]] = 1
            
            # Check if any symptoms were actually set
            if np.sum(input_vector) > 0:
                try:
                    predicted_code = svc.predict([input_vector])[0]
                    predicted_disease = diseases_list.get(predicted_code, "Unknown disease")
                    
                    if predicted_disease != "Unknown disease":
                        desc, pre, med, die, wrkout = helper(predicted_disease)
                        
                        # Add to history
                        prediction_history.insert(0, {
                            'symptoms': ', '.join(user_symptoms),
                            'disease': predicted_disease,
                            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M')
                        })
                        
                        result = {
                            'disease': predicted_disease,
                            'description': desc,
                            'precautions': pre,
                            'medications': med,
                            'diets': die,
                            'workouts': wrkout,
                            'source': 'model',
                            'matched_symptoms': user_symptoms,
                            'unknown_symptoms': unknown_symptoms
                        }
                        return jsonify(result)
                except Exception as e:
                    logger.error(f"ML model prediction error: {e}")
        
        # Fallback to Gemini API
        logger.info("Falling back to Gemini API")
        try:
            genai.configure(api_key=GEMINI_API_KEY)
            model = genai.GenerativeModel('gemini-1.5-flash')
            
            prompt = f"""
            Act as an experienced medical AI assistant. A patient has reported the following symptoms: {', '.join(raw_symptoms)}.
            
            Please provide:
            1. **Most Likely Conditions** (2-3 possibilities with confidence levels)
            2. **Detailed Description** of the most likely condition
            3. **Immediate Care Recommendations** (4-5 actionable steps)
            4. **Lifestyle Modifications** (diet, exercise, habits)
            5. **When to See a Doctor** (urgent signs to watch for)
            6. **Medications** (over-the-counter options, if applicable)
            
            Format your response clearly with headers and bullet points.
            
            **MEDICAL DISCLAIMER**: This is AI-generated information for educational purposes only. Always consult with qualified healthcare professionals for proper diagnosis and treatment.
            """
            
            response = model.generate_content(prompt)
            return jsonify({
                'result': response.text,
                'source': 'gemini',
                'input_symptoms': raw_symptoms
            })
            
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            return jsonify({
                'error': 'Unable to analyze symptoms. Please try again or consult a healthcare professional.',
                'details': str(e)
            }), 500

    except Exception as e:
        logger.error(f"General error in predict: {e}")
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

@app.route('/history')
def history():
    """Display prediction history"""
    return render_template('history.html', history=prediction_history)

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': svc is not None,
        'timestamp': datetime.now().isoformat()
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)