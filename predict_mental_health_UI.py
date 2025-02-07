import streamlit as st
import numpy as np
import joblib
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Load LLM model & tokenizer (T5 for text generation)
MODEL_NAME = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
llm_model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

# Load trained Random Forest model
@st.cache_resource
def load_model(model_path="mental_health_model.pkl"):
    return joblib.load(model_path)

model = load_model()

age_mapping = {
    "18-22": 0, "23-26": 1, "27-30": 2, "Above 30": 3, "Below 18": 4
}

gender_mapping = {
    "Female": 0, "Male": 1, "Prefer not to say": 2
}

cgpa_mapping = {
    "2.50 - 2.99": 0, "3.00 - 3.39": 1, "3.40 - 3.79": 2,
    "3.80 - 4.00": 3, "Below 2.50": 4, "Other": 5
}

scholarship_mapping = {
    "No": 0, "Yes": 1
}

depression_mapping = {
    0: "Moderate",
    1: "Minimal",
    2: "Mild",
    3: "No Depression",
    4: "Severe",
    5: "Very Severe"
}

# 🔹 Function to Predict Mental Health Condition
def predict_mental_health(symptom_inputs):
    symptom_inputs = np.array(symptom_inputs).reshape(1, -1)
    prediction = model.predict(symptom_inputs)[0]
    return depression_mapping.get(prediction)



# 🔹 Function to Generate LLM Explanation
def generate_explanation(condition):
    # Structured prompt to ensure detailed responses
    prompt = (
        f"You are a highly knowledgeable mental health expert. Write a comprehensive, well-structured explanation of {condition}. "
        "Ensure the response is at least 700-1000 words long and covers each section with in-depth analysis and real-life examples.\n\n"
        "### Overview\n"
        "Explain what the condition is and why it occurs. Provide historical background if relevant.\n\n"
        "### Common Symptoms\n"
        "List and explain at least 5-7 symptoms, including their impact on daily life and emotions.\n\n"
        "### Causes & Risk Factors\n"
        "Discuss in detail the biological, psychological, and environmental causes. Mention genetic links if any.\n\n"
        "### Effects on Daily Life\n"
        "Explain how this condition affects work, relationships, and emotions. Use real-life examples and case studies.\n\n"
        "### Coping Strategies\n"
        "Provide at least 5-7 practical coping strategies, including self-care techniques, therapy options, and lifestyle changes.\n\n"
        "### When to Seek Help\n"
        "List at least 3-5 warning signs that indicate professional help is needed.\n\n"
        "### Next Steps\n"
        "Suggest at least 3 immediate actions someone can take today to manage their condition effectively."
    )

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids

    # Tuning the LLM model
    output_ids = llm_model.generate(
        input_ids,
        max_length=2000,  
        min_length=700,   
        temperature=0.6,  
        do_sample=True,
        top_p=0.98, 
        repetition_penalty=1.2,  
        num_return_sequences=1,
        no_repeat_ngram_size=3  
    )

    explanation = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    return explanation.replace(". ", ".\n\n")

# ---- STREAMLIT UI ----
st.title("Mental Health Assessment & Explanation")

st.subheader("Enter Your Details")

# ✅ Age Input (Dropdown)
age_option = st.selectbox("Select Age Range:", list(age_mapping.keys()))
age_val = age_mapping[age_option]

# ✅ Gender Input (Radio Button)
gender_option = st.radio("Select Gender:", list(gender_mapping.keys()))
gender_val = gender_mapping[gender_option]

# ✅ CGPA Input (Dropdown)
cgpa_option = st.selectbox("Select CGPA Range:", list(cgpa_mapping.keys()))
cgpa_val = cgpa_mapping[cgpa_option]

# ✅ Scholarship Input (Radio Button)
scholarship_option = st.radio("Scholarship Received?", list(scholarship_mapping.keys()))
scholarship_val = scholarship_mapping[scholarship_option]

st.subheader("Rate Your Symptoms (0 - None, 5 - Very Severe)")
questions = [
    "Little Interest", "Feeling Down", "Sleep Issues", "Fatigue",
    "Appetite Issues", "Self Worth Issues", "Concentration Issues",
    "Restlessness", "Self-Harm Thoughts"
]
symptoms = [st.slider(q, 0, 5, 0) for q in questions]

# Submit Button
if st.button("Predict Mental Health Condition"):
    # Prepare input list
    user_inputs = [age_val, gender_val, cgpa_val, scholarship_val] + symptoms

    # Predict condition
    result = predict_mental_health(user_inputs)
    
    st.success(f"**Predicted Mental Health Condition: {result}**")

    # Generate LLM explanation
    with st.spinner("Generating detailed explanation..."):
        explanation = generate_explanation(result)

    st.subheader("Detailed Explanation")
    st.write(explanation)
