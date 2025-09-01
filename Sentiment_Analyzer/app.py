import streamlit as st
import requests

st.title("Analyze the sentiment")

text_input=st.text_area("Enter your text:")

if(st.button("predict")):
    if (text_input.strip()==" "):
        st.warning("enter some text")

    else:
        response=requests.post(
        "http://127.0.0.1:5000/predict",
        json={"text":text_input}
        )

        if response.status_code == 200:
            result = response.json()
            st.success(f"Predicted Sentiment: **{result['sentiment']}**")
            # st.write("Class ID:", result["class_id"])
        else:
            st.error("Failed to get a response from the API.")