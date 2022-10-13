import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import requests
import json
import time
import traceback
import streamlit as st


def call_api(path: str,
             method: str = 'GET',
             data: json = None,
             stream: bool = False):
    """API call function.
    Args:
        path (str): URL path.
        method (str, optional): Http request method. Defaults to 'GET'.
        data (json, optional): Json data in post method. Defaults to None.
        stream (bool, optional): True if StreamingResponse. Defaults to False.
    Returns:
        _type_: _description_
    """

    if(method.upper()=='GET'):
        res = requests.get(PATH + path, stream=stream)
    else:
        res = requests.post(PATH + path, json=data, stream=stream)
    res.raise_for_status()

    return res



################################ main ################################

## Lending page
st.title('Federated Learning')
with st.expander('Please input your server url path.', expanded=True):
    # https://jsonplaceholder.typicode.com/todos/1  http://localhost:8000
    PATH = st.text_input('e.g) https://jsonplaceholder.typicode.com/todos/1')
    if(not PATH):
        st.stop()
    st.write('Status:', call_api('/'))
    st.success('Confirmed')


'\n'
'\n'
'\n'


## menu tabs
tab_list = ['/root', '/psi', '/define-model']
tab1, tab2, tab3 = st.tabs(tab_list)

with tab1:
    st.header('Root')
    st.write('''
        _some docs..._
    ''')
    btn = st.button('Run', key='tab1')
    st.write('Status:')

    key = 'root'
    with st.empty():
        if(key in st.session_state):
            st.write(st.session_state[key])
        if(btn):
            res = call_api('/').json()
            st.write(res)
            st.session_state[key] = res

with tab2:
    st.header("PSI")
    st.write('''
        _some docs..._
    ''')
    btn = st.button('Run', key='tab2')

    key = 'psi'
    with st.empty():
        if(key in st.session_state):
            st.write(st.session_state[key])
        if(btn):
            res = ''
            with call_api('/psi', stream=True) as response:
                for chunk in response.iter_lines(chunk_size=1):
                    res += chunk.decode('utf-8') + '\n\n'
                    st.write(res)
            st.session_state[key] = res
                

with tab3:
    st.header("Defind Model")
    st.image("https://static.streamlit.io/examples/owl.jpg", width=200)









# st.error('Wrong URL.', icon='')

# btn = st.button('Simple status')
# if(btn):
#   res = requests.get(PATH)
#   if(res.status_code==200):
#     st.write('Status:', res.json())

# "---"

# ## form
# with st.form("my_form"):
#    st.write("Inside the form")
#    slider_val = st.slider("Form slider")
#    checkbox_val = st.checkbox("Form checkbox")
#    # Every form must have a submit button.
#    submitted = st.form_submit_button("Submit")
#    if submitted:
#        st.write("slider", slider_val, "checkbox", checkbox_val)
# st.write("Outside the form")

# ## inputs
# submit_btn = st.button('Submit')
# download_btn = st.download_button('Download', 'boo!')
# selected = st.checkbox('I agree')
# # print(selected)
# choice = st.selectbox('Pick one', ['Cat', 'Dog'])
# text_input = st.text_input('Save path')

# ## dataframe & plot
# data = { k: [ x+k for x in range(5) ] for k in range(10) }
# st.dataframe(data)
# st.line_chart(data)
# st.area_chart(data)

# st.balloons()

# import time
# 'Starting a long computation...'

# # Add a placeholder
# latest_iteration = st.empty()
# bar = st.progress(100)

# for i in range(100):
#   # Update the progress bar with each iteration.
#   latest_iteration.text(f'Iteration {i+1}')
#   bar.progress(i + 1)
#   time.sleep(0.1)

# '...and now we\'re done!'

# import time
# with st.spinner('Wait for it...'):
#     time.sleep(5)
# st.success('Done!')