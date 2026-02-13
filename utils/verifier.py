import requests
import streamlit as st

def verify_news(query):

    api_key = st.secrets["NEWS_API_KEY"]

    url = f"https://newsapi.org/v2/everything?q={query}&apiKey={api_key}"
    response = requests.get(url).json()

    articles = response.get("articles", [])

    match_count = len(articles)

    if match_count >= 3:
        score = 0.9
    elif match_count == 2:
        score = 0.7
    elif match_count == 1:
        score = 0.5
    else:
        score = 0.2

    return score, articles[:3]
