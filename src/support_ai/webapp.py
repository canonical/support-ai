import os
import streamlit as st
import time
import requests


st.title('Support AI')

api_svc_url = os.getenv('API_SVC_URL', '')
case_number = st.text_area(label='CaseNumber',
                     placeholder='Please provide the case number.',
                     label_visibility='hidden')

if case_number:
    url = api_svc_url + 'summarize_case'
    data = {'case_number': case_number}

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
