import os
import time
import requests
import streamlit as st


st.title('Support AI')

api_svc_url = os.getenv('API_SVC_URL', '')
case_number = st.text_area(label='CaseNumber',
                     placeholder='Please provide the case number.',
                     label_visibility='hidden')

if case_number:
    URL = f'{api_svc_url}/salesforce/{case_number}/summary'

    st.session_state.content = ''
    try:
        response = requests.get(URL, stream=True)
        with st.empty():
            for token in response.iter_content():
                st.session_state.content += "  \n" if token == b'\n' else token.decode('utf-8')
                st.write(st.session_state.content)
                time.sleep(1/100)
    except requests.exceptions.RequestException as e:
        st.error(f"An error occurred: {e}")
