import streamlit as st
from run import chatbot_response

st.set_page_config(page_title="Student Support Chatbot", page_icon="🎓")

st.title("🎓 Student Support Chatbot")
st.write("Ask me anything about campus, academics, or services!")

# Keep chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Display past messages
for msg in st.session_state["messages"]:
    if msg["role"] == "user":
        st.markdown(f"**You:** {msg['content']}")
    else:
        st.markdown(f"**Bot:** {msg['content']}")

# Input box
user_input = st.text_input("Type your question here:")

if st.button("Send") and user_input:
    # Save user message
    st.session_state["messages"].append({"role": "user", "content": user_input})

    # Get bot reply
    reply = chatbot_response(user_input)
    st.session_state["messages"].append({"role": "bot", "content": reply})

    # Refresh page to show new messages
    st.experimental_rerun()
