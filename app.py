from flask import Flask, request, session
from flask_restx import Api, Resource, fields
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import datetime

app = Flask(__name__)
app.secret_key = 'supersecretkey'
app.config['PERMANENT_SESSION_LIFETIME'] = datetime.timedelta(days=1)
CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}}, supports_credentials=True)

# Make session permanent
@app.before_request
def make_session_permanent():
    session.permanent = True

api = Api(app, version='1.0', title='Medical Diagnosis API',
          description='An API for diagnosing medical conditions based on user responses.')

# Load saved components
logistic_regression = joblib.load('logistic_regression.pkl')
rules = joblib.load('rules.pkl')
english_questions_with_evidence = joblib.load('english_questions_with_evidence.pkl')
feature_names = joblib.load('feature_names.pkl')
pathologies = joblib.load('pathologies.pkl')
label_encoder = joblib.load('label_encoder.pkl')
pathology_to_department = joblib.load('pathology_to_department.pkl')

# Create a mapping from feature name to index
feature_to_index = {name: idx for idx, name in enumerate(feature_names)}

# Utility functions
def get_department(pathology):
    return pathology_to_department.get(pathology, "Unknown Department")

def get_question_text(feature_index, questions_with_evidence, feature_names):
    for question_text, evidence_name in questions_with_evidence.items():
        if evidence_name in feature_to_index and feature_to_index[evidence_name] == feature_index:
            return question_text, evidence_name
    return None, None

def traverse_tree(log_reg, sample, feature_names):
    sample_df = pd.DataFrame([sample], columns=feature_names)
    probas = log_reg.predict_proba(sample_df)[0]
    prediction = np.argmax(probas)
    return prediction, probas

# Define API models
start_model = api.model('Start', {
    'questions': fields.List(fields.String, description='List of questions to ask at the start'),
    'question_indices': fields.List(fields.Integer, description='Indices of the initial questions')
})

answer_model = api.model('Answer', {
    'question_index': fields.Integer(description='Question index'),
    'answer': fields.Integer(description='Answer (0 or 1)')
})

ask_model = api.model('Ask', {
    'answers': fields.List(fields.Nested(answer_model), description='List of previous questions and answers'),
    'age': fields.Integer(description='Age of the user', required=True),
    'sex': fields.Integer(description='Sex of the user (female: 1, male: 0)', required=True)
})

response_model = api.model('Response', {
    'question': fields.String(description='Next question to ask'),
    'question_index': fields.Integer(description='Index of the next question'),
    'result': fields.String(description='Predicted result'),
    'department': fields.String(description='Department to visit'),
    'answers': fields.List(fields.Nested(answer_model), description='Updated list of questions and answers'),
    'age': fields.Integer(description='Age of the user')
})

@api.route('/start')
class Start(Resource):
    @api.response(200, 'Session started', start_model)
    def post(self):
        """Start a new session"""
        session.clear()
        sample = np.zeros(len(feature_names))
        session['sample'] = sample.tolist()

        questions = ["What is your sex? (female: 1, male: 0)", "How old are you?"]
        question_indices = [feature_to_index['SEX'], -1]  # -1 for age as it doesn't map directly

        return {'questions': questions, 'question_indices': question_indices}, 200

@api.route('/ask')
class Ask(Resource):
    @api.expect(ask_model)
    @api.response(200, 'Question or Result', response_model)
    def post(self):
        """Ask the next question or get the result"""
        data = request.json
        sample = np.array(session.get('sample', np.zeros(len(feature_names))))
        answers = data.get('answers', [])
        age = data.get('age')

        app.logger.debug(f"Session sample before update: {session.get('sample')}")
        app.logger.debug(f"Answers before update: {answers}")

        for item in answers:
            question_index = item['question_index']
            answer = item['answer']
            sample[question_index] = answer

        if 'sex' in data:
            sex_index = feature_to_index['SEX']
            sample[sex_index] = data['sex']
            answers.append({'question_index': sex_index, 'answer': data['sex']})

        if age is not None:
            age_ranges = ['0-12', '13-18', '19-30', '31-45', '46-60', '60<']
            for age_range in age_ranges:
                if age_range in feature_to_index:
                    age_index = feature_to_index[age_range]
                    if age_range == '0-12' and age <= 12:
                        sample[age_index] = 1
                    elif age_range == '13-18' and 13 <= age <= 18:
                        sample[age_index] = 1
                    elif age_range == '19-30' and 19 <= age <= 30:
                        sample[age_index] = 1
                    elif age_range == '31-45' and 31 <= age <= 45:
                        sample[age_index] = 1
                    elif age_range == '46-60' and 46 <= age <= 60:
                        sample[age_index] = 1
                    elif age_range == '60<' and age > 60:
                        sample[age_index] = 1
                    else:
                        sample[age_index] = 0
                    answers.append({'question_index': age_index, 'answer': sample[age_index]})

        session['sample'] = sample.tolist()

        prediction, probas = traverse_tree(logistic_regression, sample, feature_names)

        if max(probas) > 0.8 or len(answers) >= len(feature_names):
            result_label = label_encoder.inverse_transform([prediction])[0]
            department = get_department(result_label)
            session.clear()
            return {'result': result_label, 'department': department, 'answers': answers, 'age': age}, 200

        for _, row in rules.iterrows():
            antecedents = row['antecedents']
            consequents = row['consequents']

            # Check if all antecedents exist in feature_to_index
            if any(feat not in feature_to_index for feat in antecedents):
                continue

            # Check if any antecedent has already been asked
            if any(feature_to_index[feat] in [a['question_index'] for a in answers] for feat in antecedents):
                continue

            for feature in antecedents:
                if feature in feature_to_index:
                    feature_index = feature_to_index[feature]
                    if feature_index not in [a['question_index'] for a in answers]:
                        question_text, _ = get_question_text(feature_index, english_questions_with_evidence, feature_names)
                        session['sample'] = sample.tolist()
                        return {'question': question_text, 'question_index': feature_index, 'answers': answers, 'age': age}, 200

        return {'message': 'No more questions to ask', 'answers': answers, 'age': age}, 200

if __name__ == '__main__':
    app.run(debug=True)
