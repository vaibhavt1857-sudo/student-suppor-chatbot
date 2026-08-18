import streamlit as st
from run import chatbot_response

# Page setup
st.set_page_config(page_title="Student Support Chatbot", page_icon="🎓")

st.title("🎓 Student Support Chatbot")
st.write("Ask me anything about campus, academics, or services!")

# Cache chatbot so heavy models don't reload every time
@st.cache_resource
def get_chatbot():
    return chatbot_response

chatbot = get_chatbot()

# Keep chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Display past messages
for msg in st.session_state["messages"]:
    if msg["role"] == "user":
        st.markdown(f"<div style='background-color:#e6f2ff;padding:8px;border-radius:5px'><b>You:</b> {msg['content']}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div style='background-color:#e6ffe6;padding:8px;border-radius:5px'><b>Bot:</b> {msg['content']}</div>", unsafe_allow_html=True)

# Input box
user_input = st.text_input("Type your question here:")

if st.button("Send") and user_input:
    # Save user message
    st.session_state["messages"].append({"role": "user", "content": user_input})

    # Get bot reply (cached)
    reply = chatbot(user_input)
    st.session_state["messages"].append({"role": "bot", "content": reply})

    # Refresh page to show new messages
    st.experimental_rerun()
