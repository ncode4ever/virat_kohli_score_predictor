import streamlit as st
import joblib
import numpy as np
import pandas as pd

columns = ["match", "innings", "balls_faced", "fours", "sixes"]
model = joblib.load('reg_model')
df = joblib.load('virat_score_data')

st.title("Virat Kohli ODI Score Prediction")
st.caption("Please provide the specification as listed below and click on Predict Score")
st.caption("This ML model is all the innings played by Virat Kohli in ODI matches, so might not give the most accurate predictions for other players")

no_of_matches = st.slider("Match Number", min_value=1, max_value=500, step=1, value=int(df['match'].max()))
innings = st.radio("Innings", [1,2], index=0, horizontal=True)
balls_faced = st.slider("Balls faced", min_value=0, max_value=200, step=1, value=int(round(df['balls_faced'].median())))
fours = st.slider("Fours", min_value=0, max_value=30, step=1, value=int(round(df['fours'].median())))
sixes = st.slider("Sixes", min_value=0, max_value=15, step=1, value=int(round(df['sixes'].median())))

if st.button("PREDICT SCORE"):

    query = pd.DataFrame(
        [[no_of_matches, innings, balls_faced, fours, sixes]],
        columns=["match", "innings", "balls_faced", "fours", "sixes"])

    op = model.predict(query)

    st.subheader(
        f"The estimated score match {no_of_matches} with innings {innings} is {int(op[0])}"
    )
