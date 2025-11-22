import streamlit as st
from chatbot import HRChatbot

st.set_page_config(page_title='HR Chatbot Demo', layout='centered')

st.title('Intelligent HR Assistant — Demo')
st.write('Ask payroll or HR-related questions (leave balance, employee details, payroll queries).')

# Initialize Chatbot
bot = HRChatbot()

# Store chat history
if 'history' not in st.session_state:
    st.session_state.history = []

# Chat input form
with st.form('chat_form', clear_on_submit=True):
    user_input = st.text_input('You:', '')
    submitted = st.form_submit_button('Send')

# Process message
if submitted and user_input:
    resp = bot.retrieve(user_input)
    st.session_state.history.append(('You', user_input))
    st.session_state.history.append(('Bot', resp))

# Display chat history
for sender, text in st.session_state.history:
    if sender == 'You':
        st.markdown(f"**🧑‍💼 You:** {text}")
    else:
        st.markdown(f"**🤖 Bot:** {text}")

st.markdown('---')
st.write('Try: "Check leaves for EMP10234", "Show employee details EMP56789", "How to update bank details?"')
