import streamlit as st
import torch
import numpy as np
from utils.model_loader import load_model
from utils.verifier import verify_news

st.set_page_config(page_title="DeepGuard AI", layout="wide")

st.title("🛡 DeepGuard: AI Fake News Detection")

model, tokenizer = load_model()

user_input = st.text_area("Enter News Text")

if st.button("Analyze News"):

    encoding = tokenizer(
        user_input,
        padding="max_length",
        truncation=True,
        max_length=256,
        return_tensors="pt"
    )

    with torch.no_grad():
        outputs = model(
            encoding["input_ids"],
            encoding["attention_mask"]
        )

    probability = torch.sigmoid(outputs).item()

    model_score = probability

    verification_score, articles = verify_news(user_input[:50])

    final_score = 0.7 * model_score + 0.3 * verification_score

    if final_score > 0.6:
        prediction = "REAL ✅"
    else:
        prediction = "FAKE ❌"

    st.subheader("Prediction:")
    st.write(prediction)
    st.write(f"Confidence Score: {round(final_score*100,2)}%")

    st.subheader("Verification Matches:")
    for article in articles:
        st.write(article["title"])
