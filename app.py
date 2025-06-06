import streamlit as st
import pickle
import numpy as np
from streamlit_option_menu import option_menu
import copy 

# --- Configuration & Data (Hardcoded for Simplicity) ---
# 1. User Credentials
USERS = {
    "abhay1": "pass1",
    "shruti1": "pass2"
}

DOCTORS_DATA_TEMPLATE = {
    "Dr. Lakshita Mathpal (Cardiologist)": {
        "specialty": "Heart Disease",
        "availability": ["Monday 10:00 AM", "Wednesday 2:00 PM", "Friday 9:00 AM"],
        "image": "images/dr lakuuu copy.jpeg"# <-- Make sure this matches your actual filename
    },
    "Dr. Sakshi Semwal (Endocrinologist)": {
        "specialty": "Diabetes",
        "availability": ["Tuesday 9:00 AM", "Thursday 11:00 AM", "Friday 1:00 PM"],
        "image": "images/dr sak copy.jpeg" # <-- Make sure this matches your actual filename
    },
    "Dr. Harshit Banaula (Neurologist)": {
        "specialty": "Parkinson's Disease",
        "availability": ["Monday 3:00 PM", "Wednesday 10:00 AM", "Thursday 4:00 PM"],
        "image": "images/dr h copy.jpeg"# <-- Make sure this matches your actual filename
    }
}
# --- Disease-Specific Precautions ---
PRECAUTIONS = {
    "Diabetes": {
        "Diabetic": [
            "Monitor blood sugar regularly as advised by your doctor.",
            "Follow a balanced diabetic diet plan.",
            "Engage in regular physical activity.",
            "Take prescribed medications consistently.",
            "Attend regular check-ups with your healthcare provider."
        ],
        "Not Diabetic": [
            "Maintain a healthy lifestyle with a balanced diet and regular exercise.",
            "Be aware of diabetes risk factors (family history, obesity, etc.).",
            "Consider regular health check-ups."
        ]
    },
    "Heart Disease": {
        "Has Heart Disease": [
            "Follow your doctor's treatment plan strictly, including medications.",
            "Adopt a heart-healthy diet (low in sodium, saturated/trans fats).",
            "Engage in regular, moderate exercise as approved by your doctor.",
            "Manage stress through relaxation techniques.",
            "Avoid smoking and limit alcohol consumption."
        ],
        "Does Not Have Heart Disease": [
            "Maintain a heart-healthy lifestyle: balanced diet, regular exercise.",
            "Manage blood pressure, cholesterol, and blood sugar levels.",
            "Avoid smoking and excessive alcohol.",
            "Manage stress effectively."
        ]
    },
    "Parkinson's Disease": {
        "Has Parkinson's Disease": [
            "Adhere to the medication schedule prescribed by your neurologist.",
            "Engage in regular exercise, focusing on balance, flexibility, and strength (e.g., Tai Chi, boxing).",
            "Work with physical, occupational, and speech therapists as recommended.",
            "Ensure a safe home environment to prevent falls.",
            "Maintain a healthy diet and stay hydrated."
        ],
        "Does Not Have Parkinson's Disease": [
            "Maintain an active lifestyle and a healthy diet.",
            "Be aware of early symptoms if there's a family history.",
            "Protect against head injuries where possible."
        ]
    }
}


# --- Model Loading ---
try:
    diabetes_model = pickle.load(open('saved_models/diabetes_model.sav', 'rb'))
    heart_disease_model = pickle.load(open('saved_models/heart_disease_model.sav', 'rb'))
    parkinsons_model = pickle.load(open('saved_models/parkinsons_model.sav', 'rb'))
except FileNotFoundError:
    st.error("Critical Error: Model files not found. Ensure 'saved_models' directory and .sav files exist.")
    st.stop()
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()

# --- Helper Functions for Prediction ---
def predict_diabetes(input_data):
    # ... (same as before)
    input_data_as_numpy_array = np.asarray(input_data, dtype=float)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    prediction = diabetes_model.predict(input_data_reshaped)
    return "Diabetic" if prediction[0] == 1 else "Not Diabetic"

def predict_heart_disease(input_data):
    # ... (same as before)
    input_data_as_numpy_array = np.asarray(input_data, dtype=float)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    prediction = heart_disease_model.predict(input_data_reshaped)
    return "Has Heart Disease" if prediction[0] == 1 else "Does Not Have Heart Disease"

def predict_parkinsons(input_data):
    # ... (same as before)
    input_data_as_numpy_array = np.asarray(input_data, dtype=float)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    prediction = parkinsons_model.predict(input_data_reshaped)
    return "Has Parkinson's Disease" if prediction[0] == 1 else "Does Not Have Parkinson's Disease"


# --- Session State Initialization ---
def initialize_session_state():
    if 'logged_in' not in st.session_state: st.session_state.logged_in = False
    if 'username' not in st.session_state: st.session_state.username = ""
    if 'patient_details_submitted' not in st.session_state: st.session_state.patient_details_submitted = False
    if 'patient_age' not in st.session_state: st.session_state.patient_age = ""
    if 'patient_gender' not in st.session_state: st.session_state.patient_gender = "Prefer not to say"
    if 'appointment_booked' not in st.session_state: st.session_state.appointment_booked = False
    if 'booked_doctor' not in st.session_state: st.session_state.booked_doctor = None
    if 'booked_slot' not in st.session_state: st.session_state.booked_slot = None
    if 'show_prediction_interface' not in st.session_state: st.session_state.show_prediction_interface = False
    if 'prediction_made' not in st.session_state: st.session_state.prediction_made = False
    if 'last_prediction_disease' not in st.session_state: st.session_state.last_prediction_disease = ""
    if 'last_prediction_result' not in st.session_state: st.session_state.last_prediction_result = ""
    if 'doctors_availability' not in st.session_state:
        st.session_state.doctors_availability = copy.deepcopy(DOCTORS_DATA_TEMPLATE)

initialize_session_state() # Call it once at the start

# --- MODULE 1: LOGIN PAGE ---
def display_login_page():
    st.title("User Login")
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        login_button = st.form_submit_button("Login")

        if login_button:
            if username in USERS and USERS[username] == password:
                st.session_state.logged_in = True
                st.session_state.username = username
                # Reset other states for a fresh flow after login
                st.session_state.patient_details_submitted = False
                st.session_state.appointment_booked = False
                st.session_state.show_prediction_interface = False
                st.session_state.prediction_made = False
                st.session_state.doctors_availability = copy.deepcopy(DOCTORS_DATA_TEMPLATE)
                st.rerun()
            else:
                st.error("Invalid username or password")

# --- MODULE 2: PATIENT DETAILS PAGE ---
def display_patient_details_page():
    st.title(f"Welcome, {st.session_state.username}!")
    st.subheader("Please provide your details:")
    with st.form("patient_details_form"):
        age = st.number_input("Your Age", min_value=1, max_value=120, step=1, value=st.session_state.get('patient_age_input', 30))
        gender = st.selectbox("Your Gender", ["Male", "Female", "Other", "Prefer not to say"], index=3 if st.session_state.patient_gender == "Prefer not to say" else ["Male", "Female", "Other", "Prefer not to say"].index(st.session_state.patient_gender))
        submit_details = st.form_submit_button("Submit Details & Proceed to Appointments")

        if submit_details:
            if not age:
                st.warning("Please enter your age.")
            else:
                st.session_state.patient_age = age
                st.session_state.patient_gender = gender
                st.session_state.patient_details_submitted = True
                st.session_state.patient_age_input = age # Store for pre-filling form if they come back
                st.rerun()

# --- MODULE 3: APPOINTMENT & PREDICTION APP ---
def display_main_app():
    # Sidebar for welcome and logout
    st.sidebar.success(f"Welcome, {st.session_state.username}!")
    st.sidebar.info(f"Age: {st.session_state.patient_age}, Gender: {st.session_state.patient_gender}")
    if st.sidebar.button("Logout"):
        for key in list(st.session_state.keys()): del st.session_state[key]
        initialize_session_state() # Re-initialize after clearing
        st.rerun()

    # --- Main Page Content ---
    if st.session_state.prediction_made:
        display_thank_you_precautions_page()
    elif not st.session_state.appointment_booked:
        display_appointment_booking_page()
    elif st.session_state.appointment_booked and not st.session_state.show_prediction_interface:
        st.success(f"Appointment Confirmed with {st.session_state.booked_doctor} at {st.session_state.booked_slot}.")
        if st.button("Proceed to Health Prediction", key="proceed_to_predict"):
            st.session_state.show_prediction_interface = True
            st.rerun()
    elif st.session_state.show_prediction_interface:
        display_prediction_interface()


def display_appointment_booking_page():
    st.title("Doctor Appointment")
    st.header("Book an Appointment")
    doctor_options = list(st.session_state.doctors_availability.keys())
    selected_doctor_name = st.selectbox("Choose a Doctor:", doctor_options, key="doctor_select")

    if selected_doctor_name:
        doctor_info = st.session_state.doctors_availability[selected_doctor_name]
        col1, col2 = st.columns([1, 2])
        with col1:
            if "image" in doctor_info and doctor_info["image"]:
                try:
                    st.image(doctor_info["image"], width=150)
                except Exception as e:
                    st.warning(f"Could not load image for {selected_doctor_name}. Path: {doctor_info['image']}. Error: {e}")
            else:
                st.markdown("_(No image provided)_") # Placeholder if image path is missing or empty
        with col2:
            st.write(f"**Specialty:** {doctor_info['specialty']}")
            available_slots = doctor_info["availability"]
            if available_slots:
                selected_slot = st.selectbox("Available Slots:", available_slots, key="slot_select_" + selected_doctor_name.replace(" ", "_"))
                if st.button("Book Appointment", key="book_button"):
                    st.session_state.appointment_booked = True
                    st.session_state.booked_doctor = selected_doctor_name
                    st.session_state.booked_slot = selected_slot
                    if selected_slot in st.session_state.doctors_availability[selected_doctor_name]["availability"]:
                        st.session_state.doctors_availability[selected_doctor_name]["availability"].remove(selected_slot)
                    st.rerun()
            else:
                st.warning("No available slots for this doctor at the moment.")

def display_prediction_interface():
    st.title("Health Prediction")
    st.header(f"Prediction for {st.session_state.booked_doctor.split('(')[0].strip()}") # Show doctor name without specialty

    with st.sidebar:
         st.subheader("Prediction Menu")
         selected_disease = option_menu('Select Disease',
                                       ['Diabetes Prediction',
                                        'Heart Disease Prediction',
                                        'Parkinson\'s Prediction'],
                                       menu_icon='hospital-fill',
                                       icons=['activity', 'heart', 'person'],
                                       default_index=0,
                                       key="disease_option_menu")
    st.subheader(f"{selected_disease}")

    # --- Diabetes Prediction Inputs ---
    if selected_disease == 'Diabetes Prediction':
        with st.form("diabetes_form"):
            # ... (inputs as before)
            col1, col2, col3 = st.columns(3)
            with col1: Pregnancies = st.text_input('Number of Pregnancies')
            with col2: Glucose = st.text_input('Glucose Level (mg/dL)')
            with col3: BloodPressure = st.text_input('Blood Pressure (mm Hg)')
            with col1: SkinThickness = st.text_input('Skin Thickness (mm)')
            with col2: Insulin = st.text_input('Insulin Level (mu U/ml)')
            with col3: BMI = st.text_input('BMI (kg/m²)')
            with col1: DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function')
            with col2: Age_diabetes = st.text_input('Age (years)', value=str(st.session_state.patient_age) if st.session_state.patient_age else "") # Pre-fill age
            
            submit_button_diabetes = st.form_submit_button('Get Diabetes Prediction')

        if submit_button_diabetes:
            inputs = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age_diabetes]
            if not all(inputs):
                st.warning("Please fill all fields.")
            else:
                try:
                    diagnosis_raw = predict_diabetes(inputs)
                    diagnosis_full = f"The prediction is: {diagnosis_raw}"
                    st.session_state.last_prediction_disease = "Diabetes"
                    st.session_state.last_prediction_result_raw = diagnosis_raw # Store raw result for precaution logic
                    st.session_state.last_prediction_result = diagnosis_full
                    st.session_state.prediction_made = True
                    st.rerun()
                except ValueError: st.error("Invalid numerical input.")
                except Exception as e: st.error(f"Prediction error: {e}")

    # --- Heart Disease Prediction Inputs ---
    elif selected_disease == 'Heart Disease Prediction':
        with st.form("heart_form"):
            # ... (inputs as before, pre-fill age and sex if possible)
            col1, col2, col3 = st.columns(3)
            with col1: age_heart = st.text_input('Age (years)', value=str(st.session_state.patient_age) if st.session_state.patient_age else "")
            sex_options = [0, 1] # 0 for female, 1 for male
            default_sex_index = 0 # Default to female
            if st.session_state.patient_gender == "Male": default_sex_index = 1
            elif st.session_state.patient_gender == "Female": default_sex_index = 0

            with col2: sex_heart = st.selectbox('Sex', sex_options, index=default_sex_index, format_func=lambda x: 'Female' if x == 0 else 'Male')
            with col3: cp = st.selectbox('Chest Pain Type', [0, 1, 2, 3], help="0: Typical, 1: Atypical, 2: Non-anginal, 3: Asymptomatic")
            # ... other inputs ...
            with col1: trestbps = st.text_input('Resting Blood Pressure (mm Hg)')
            with col2: chol = st.text_input('Serum Cholesterol (mg/dl)')
            with col3: fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl', [0, 1], format_func=lambda x: 'False' if x == 0 else 'True')
            with col1: restecg = st.selectbox('Resting ECG Results', [0, 1, 2], help="0: Normal, 1: ST-T abnorm, 2: LV hypertrophy")
            with col2: thalach = st.text_input('Max Heart Rate Achieved')
            with col3: exang = st.selectbox('Exercise Induced Angina', [0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')
            with col1: oldpeak = st.text_input('ST Depression by Exercise')
            with col2: slope = st.selectbox('Slope of Peak Exercise ST Segment', [0, 1, 2])
            with col3: ca = st.selectbox('Major Vessels Colored (0-3)', [0, 1, 2, 3])
            with col1: thal = st.selectbox('Thalassemia (0-3)', [0, 1, 2, 3], help="0:NULL,1:Normal,2:Fixed,3:Reversible")

            submit_button_heart = st.form_submit_button('Get Heart Disease Prediction')

        if submit_button_heart:
            inputs = [age_heart, sex_heart, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
            if not all([age_heart, trestbps, chol, thalach, oldpeak]): # Basic check for some text inputs
                st.warning("Please fill all required text fields.")
            else:
                try:
                    diagnosis_raw = predict_heart_disease(inputs)
                    diagnosis_full = f"The prediction is: {diagnosis_raw}"
                    st.session_state.last_prediction_disease = "Heart Disease"
                    st.session_state.last_prediction_result_raw = diagnosis_raw
                    st.session_state.last_prediction_result = diagnosis_full
                    st.session_state.prediction_made = True
                    st.rerun()
                except ValueError: st.error("Invalid numerical input.")
                except Exception as e: st.error(f"Prediction error: {e}")

    # --- Parkinson's Prediction Inputs ---
    elif selected_disease == "Parkinson's Prediction":
        with st.form("parkinsons_form"):
            # ... (inputs as before)
            c1,c2,c3,c4 = st.columns(4)
            with c1: fo = st.text_input('MDVP:Fo(Hz)')
            with c2: fhi = st.text_input('MDVP:Fhi(Hz)')
            with c3: flo = st.text_input('MDVP:Flo(Hz)')
            with c4: Jitter_percent = st.text_input('MDVP:Jitter(%)')
            # ... Add all other Parkinson's inputs here, laid out as you prefer ...
            with c1: Jitter_Abs = st.text_input('MDVP:Jitter(Abs)')
            with c2: RAP = st.text_input('MDVP:RAP')
            with c3: PPQ = st.text_input('MDVP:PPQ')
            with c4: DDP = st.text_input('Jitter:DDP')
            with c1: Shimmer = st.text_input('MDVP:Shimmer')
            with c2: Shimmer_dB = st.text_input('MDVP:Shimmer(dB)')
            with c3: APQ3 = st.text_input('Shimmer:APQ3')
            with c4: APQ5 = st.text_input('Shimmer:APQ5')
            with c1: APQ = st.text_input('MDVP:APQ')
            with c2: DDA = st.text_input('Shimmer:DDA')
            with c3: NHR = st.text_input('NHR')
            with c4: HNR = st.text_input('HNR')
            with c1: RPDE = st.text_input('RPDE')
            with c2: DFA = st.text_input('DFA')
            with c3: spread1 = st.text_input('spread1')
            with c4: spread2 = st.text_input('spread2')
            with c1: D2 = st.text_input('D2')
            with c2: PPE = st.text_input('PPE')

            submit_button_parkinsons = st.form_submit_button("Get Parkinson's Prediction")

        if submit_button_parkinsons:
            inputs = [fo, fhi, flo, Jitter_percent, Jitter_Abs, RAP, PPQ, DDP,Shimmer, Shimmer_dB, APQ3, APQ5, APQ, DDA, NHR, HNR, RPDE, DFA, spread1, spread2, D2, PPE]
            if not all(inputs):
                st.warning("Please fill all fields.")
            else:
                try:
                    diagnosis_raw = predict_parkinsons(inputs)
                    diagnosis_full = f"The prediction is: {diagnosis_raw}"
                    st.session_state.last_prediction_disease = "Parkinson's Disease"
                    st.session_state.last_prediction_result_raw = diagnosis_raw
                    st.session_state.last_prediction_result = diagnosis_full
                    st.session_state.prediction_made = True
                    st.rerun()
                except ValueError: st.error("Invalid numerical input.")
                except Exception as e: st.error(f"Prediction error: {e}")

    if st.session_state.show_prediction_interface:
        if st.button("Back to Appointment Booking", key="back_to_appt"):
            st.session_state.show_prediction_interface = False
            st.session_state.appointment_booked = False # Allow re-booking
            st.session_state.prediction_made = False
            st.rerun()


# --- MODULE 4: THANK YOU & PRECAUTIONS PAGE ---
def display_thank_you_precautions_page():
    st.title("Prediction Result & Advice")
    st.balloons() # Fun!
    st.header(f"Thank you, {st.session_state.username}!")
    st.subheader(f"For your {st.session_state.last_prediction_disease.lower()} check:")
    st.success(st.session_state.last_prediction_result)

    st.markdown("---")
    st.subheader("Important Precautions & Next Steps:")

    disease_key_map = {
        "Diabetes": "Diabetes",
        "Heart Disease": "Heart Disease",
        "Parkinson's Disease": "Parkinson's Disease"
    }
    disease_precautions_key = disease_key_map.get(st.session_state.last_prediction_disease)
    
    # Use the raw prediction result (e.g., "Diabetic", "Not Diabetic") to fetch precautions
    prediction_status_key = st.session_state.last_prediction_result_raw

    if disease_precautions_key and prediction_status_key in PRECAUTIONS.get(disease_precautions_key, {}):
        precautions_list = PRECAUTIONS[disease_precautions_key][prediction_status_key]
        for i, precaution in enumerate(precautions_list):
            st.markdown(f"{i+1}. {precaution}")
    else:
        st.warning("Precautions for this specific result are not available. Please consult your doctor.")

    st.markdown("---")
    st.markdown("**Disclaimer:** This prediction is based on a machine learning model and is not a substitute for professional medical advice. Always consult with a qualified healthcare provider for any health concerns or before making any decisions related to your health or treatment.")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Make Another Prediction", key="predict_again"):
            st.session_state.prediction_made = False
            # Keep appointment details, just go back to prediction interface
            st.session_state.show_prediction_interface = True
            st.rerun()
    with col2:
        if st.button("Book New Appointment / Change Doctor", key="new_appt_from_precautions"):
            st.session_state.prediction_made = False
            st.session_state.show_prediction_interface = False
            st.session_state.appointment_booked = False
            # Patient details remain, go back to appointment booking
            st.rerun()


# --- Main App Logic (Flow Control) ---
if not st.session_state.logged_in:
    display_login_page()
elif not st.session_state.patient_details_submitted:
    display_patient_details_page()
else: # Logged in AND patient details submitted
    display_main_app()