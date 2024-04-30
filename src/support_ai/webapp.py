import streamlit as st
import time
import requests

WEBSERVER_URL = 'http://127.0.0.1:5000/'

st.title('Support AI')

query = st.text_area(label='Query',
                     placeholder='Please describe the symptom you experienced.',
                     label_visibility='hidden')

if query:
    url = WEBSERVER_URL + 'ask_ai'
    data = {'query': query}

    st.session_state.content = ''
    try:
        response = requests.post(url, data=data)
        with st.empty():
            for token in response.iter_content():
                st.session_state.content += "  \n" if token == b'\n' else token.decode('utf-8')
                st.write(st.session_state.content)
                time.sleep(1/100)
    except requests.exceptions.RequestException as e:
        st.error(f"An error occurred: {e}")
