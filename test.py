import joblib
import pandas as pd
import numpy as np
from model import get_department, get_question_text, ask_questions_lg

# Load the logistic regression model
logistic_regression_model = joblib.load('logistic_regression_model.pkl')

# Load the label encoder
label_encoder = joblib.load('label_encoder.pkl')

# Load the association rules DataFrame
rules = joblib.load('association_rules.pkl')

# Load the dictionary mapping questions to evidence
questions_with_evidence = joblib.load('questions_with_evidence.pkl')

# Load the feature names
feature_names = joblib.load('feature_names.pkl')

def ask_questions_lg(log_reg, rules, questions_with_evidence, feature_names):
    def traverse_tree(sample):
        # Convert the sample to a DataFrame with the correct feature names
        sample_df = pd.DataFrame([sample], columns=feature_names)
        probas = log_reg.predict_proba(sample_df)[0]
        prediction = np.argmax(probas)
        return prediction, probas

    # Convert feature_names to a list
    feature_names = list(feature_names)

    # Create a mapping from feature name to index
    feature_to_index = {name: idx for idx, name in enumerate(feature_names)}

    sample = np.zeros(len(feature_names))  # Initialize sample with zeros
    asked_questions = set()

    # Ask about sex
    sex = int(input("What is your sex? (female: 1, male: 0): "))
    sex_index = feature_to_index['SEX']
    sample[sex_index] = sex
    asked_questions.add(sex_index)

    # Ask about age
    age = int(input("How old are you? "))
    age_ranges = ['0-12', '13-18', '19-30', '31-45', '46-60', '60<']
    for age_range in age_ranges:
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
        asked_questions.add(age_index)

    while True:
        # Predict the result based on current answers
        prediction, probas = traverse_tree(sample)

        # Check if we can stop asking questions
        if max(probas) > 0.8 or len(asked_questions) >= len(feature_names):
            break

        # Find the next question to ask based on the rules
        for _, row in rules.iterrows():
            antecedents = row['antecedents']
            consequents = row['consequents']

            # Skip if any antecedent has already been asked
            if any(feature_to_index[feat] in asked_questions for feat in antecedents):
                continue

            # Ask the next antecedent question
            for feature in antecedents:
                feature_index = feature_to_index[feature]
                if feature_index not in asked_questions:
                    question_text, _ = get_question_text(feature_index, questions_with_evidence, feature_names)
                    answer = int(input(f"{question_text} (0 or 1): "))
                    sample[feature_index] = answer
                    asked_questions.add(feature_index)
                    break

            # If we've asked a question, break the loop to re-evaluate
            if feature_index in asked_questions:
                break

    result_label = label_encoder.inverse_transform([prediction])[0]
    department = get_department(result_label)
    print(f"Predicted Result: {result_label} ... You should see the {department} department.")

# Example usage
ask_questions_lg(logistic_regression_model, rules, questions_with_evidence, feature_names)
