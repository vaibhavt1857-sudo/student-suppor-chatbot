import streamlit as st
from run import chatbot_response   # import your function from run.py

st.title("🎓 Student Support Chatbot")

user_input = st.text_input("Ask me something:")

if user_input:
    response = chatbot_response(user_input)
    st.write(response)
