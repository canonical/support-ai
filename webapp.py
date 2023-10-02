import streamlit as st
import requests

WEBSERVER_URL = 'http://127.0.0.1:5000/'

# Define the Streamlit app
st.title('Avalokitesvara')

# Create a text input field for user input
query = st.text_area(label='Query',
                     placeholder='Please describe the symptom you experienced.',
                     label_visibility='hidden')

# Create a button to trigger the POST request
if query:
    # Define the URL to which you want to send the POST request
    sf_url = WEBSERVER_URL + 'salesforce'

    # Define the data to be sent in the POST request (you can modify this as needed)
    data = {'query': query}

    # Send the POST request
    try:
        response = requests.post(sf_url, data=data)
        st.write(response.text)
    except requests.exceptions.RequestException as e:
        st.error(f"An error occurred: {e}")
