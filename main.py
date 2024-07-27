from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()
# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Allow React app running on this port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
class AnswerRequest(BaseModel):
    answer: str
    responses: dict

# Load your model and rules
logistic_regression = joblib.load('logistic_regression.pkl')
rules = joblib.load('rules.pkl')
english_questions_with_evidence = joblib.load('english_questions_with_evidence.pkl')
feature_names = joblib.load('feature_names.pkl')
pathologies = joblib.load('pathologies.pkl')
feature_index = joblib.load('feature_index.pkl')
questions_with_evidence = joblib.load('questions_with_evidence.pkl')
label_encoder = joblib.load('label_encoder.pkl')

feature_to_index = {name: idx for idx, name in enumerate(feature_names)}

def get_department(pathology):
    pathology_to_department = {
        "Spontaneous pneumothorax": "Pulmonology",
        "Cluster headache": "Neurology",
        "Boerhaave": "Gastroenterology",
        "Spontaneous rib fracture": "Orthopedics",
        "GERD": "Gastroenterology",
        "HIV (initial infection)": "Infectious Diseases",
        "Anemia": "Hematology",
        "Viral pharyngitis": "Otolaryngology",
        "Inguinal hernia": "General Surgery",
        "Myasthenia gravis": "Neurology",
        "Whooping cough": "Infectious Diseases",
        "Anaphylaxis": "Emergency Medicine",
        "Epiglottitis": "Otolaryngology",
        "Guillain-Barré syndrome": "Neurology",
        "Acute laryngitis": "Otolaryngology",
        "Croup": "Pediatrics",
        "PSVT": "Cardiology",
        "Atrial fibrillation": "Cardiology",
        "Bronchiectasis": "Pulmonology",
        "Allergic sinusitis": "Otolaryngology",
        "Chagas": "Infectious Diseases",
        "Scombroid food poisoning": "Emergency Medicine",
        "Myocarditis": "Cardiology",
        "Laryngospasm": "Otolaryngology",
        "Acute dystonic reactions": "Neurology",
        "Localized edema": "General Medicine",
        "SLE": "Rheumatology",
        "Tuberculosis": "Infectious Diseases",
        "Unstable angina": "Cardiology",
        "Stable angina": "Cardiology",
        "Ebola": "Infectious Diseases",
        "Acute otitis media": "Otolaryngology",
        "Panic attack": "Psychiatry",
        "Bronchospasm / acute asthma exacerbation": "Pulmonology",
        "Bronchitis": "Pulmonology",
        "Acute COPD exacerbation / infection": "Pulmonology",
        "Pulmonary embolism": "Pulmonology",
        "URTI": "Otolaryngology",
        "Influenza": "Infectious Diseases",
        "Pneumonia": "Pulmonology",
        "Acute rhinosinusitis": "Otolaryngology",
        "Chronic rhinosinusitis": "Otolaryngology",
        "Bronchiolitis": "Pediatrics",
        "Pulmonary neoplasm": "Oncology",
        "Possible NSTEMI / STEMI": "Cardiology",
        "Sarcoidosis": "Pulmonology",
        "Pancreatic neoplasm": "Oncology",
        "Acute pulmonary edema": "Cardiology",
        "Pericarditis": "Cardiology"
    }
    return pathology_to_department.get(pathology, "Unknown Department")

@app.post("/question")
async def get_next_question(request: AnswerRequest):
    answer = request.answer
    responses = request.responses

    # Update responses with the new answer
    responses[feature_names[len(responses)]] = int(answer)

    # Prepare the sample with current responses
    sample = np.zeros(len(feature_names))
    for feature, value in responses.items():
        sample[feature_to_index[feature]] = value

    probas = model.predict_proba([sample])[0]
    prediction = np.argmax(probas)
    
    # Check if prediction confidence is high or max questions asked
    if max(probas) > 0.8 or len(responses) >= len(feature_names):
        result_label = label_encoder.inverse_transform([prediction])[0]
        department = get_department(result_label)
        return {"result": {"prediction": result_label, "department": department}}

    # Find next question based on rules
    for _, row in rules.iterrows():
        antecedents = row['antecedents']
        if all(feature_to_index[feat] in responses for feat in antecedents):
            continue

        for feature in antecedents:
            feature_index = feature_to_index[feature]
            if feature_index not in responses:
                question_text, _ = questions_with_evidence.get(feature, (None, None))
                return {"nextQuestion": question_text, "responses": responses}

    return {"nextQuestion": None, "responses": responses}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
