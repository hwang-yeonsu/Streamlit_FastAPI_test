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
    """ API call function.

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


def text2list(raw: str,
              dtype: callable):
    """ Convert text_input to List[int].
        Using in /define-model.

    Args:
        raw (str): Raw text input
        dtype function: Data type.

    Returns:
        List[[int]]: Nested List[int]
    """

    if(raw.lower() in ['', 'false']):
        return [0.0]

    res = []
    for x in raw.strip('[], ').split('],'):
        tmp = []
        for val in x.split(','):
            tmp.append(dtype(val.strip('[], ')))
        res.append(tmp)

    return res



################################ main ################################

## Lending page
st.title('Federated Learning')
with st.expander('Input your server url path.', expanded=True):
    # https://jsonplaceholder.typicode.com/todos/1  http://localhost:8000
    with st.empty():
        PATH = st.text_input('e.g) http://localhost:8000')
        if(not PATH):
            st.stop()
        if('start_status' not in st.session_state):
            st.session_state['start_status'] = call_api('/')
        st.success('Confirmed')
    st.write(f'''
        Connect to Server: _{st.session_state['start_status']}_  
        Docs: {PATH}/docs
    ''')
'\n'
'\n'
'\n'


## menu tabs
tab_list = ['Root', 'PSI', 'Define Model']
tab1, tab2, tab3 = st.tabs(tab_list)

with tab1:
    st.header("Root")
    st.write('''
        _Simple status.  
        API: /(root) [GET]_  
    ''');'\n'
    btn = st.button('Run', key='tab1')

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
        _Private Set Intersection  
        API: /psi [GET]_  
    ''');'\n'
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
    st.header("Define Model")
    st.write('''
        _Define your NeuralNet Models  
        API: /define-model [POST]_
    ''');'\n'
    btn = st.button('Run', key='tab3')

    with st.form('define-model_form'):
        #### --- #### --- ####
        '#### Layers'
        inner_layer = text2list(
            st.text_input('Inner model :', '[95,190,95,47]'), int
        )
        outer_layer = text2list(
            st.text_input('Outer model :', '[47, 1]'), int
        )
        layers = [inner_layer, outer_layer]

        #### --- #### --- ####
        cols_1 = st.columns(2)
        with cols_1[0]:
            '#### Dropout'
            inner_dropout = text2list(
                st.text_input('Inner dropout :', '[0.1, 0.3, 0]'), float
            )
            outer_dropout = text2list(
                st.text_input('Outer dropout :', '[0]'), float
            )
            dropout = [inner_dropout, outer_dropout]

            '#### BatchNorm'
            batchnorm =  st.select_slider(
                'batchnorm', label_visibility='collapsed',
                options = ['False', 'True'], value='True',
            )
        with cols_1[1]:
            '#### Activation'
            feature_activation = st.selectbox(
                'feature_activation', label_visibility='visible',
                options = ['ReLU', 'SeLU', 'Sigmoid'],
            )
            target_activation = st.selectbox(
                'target_activation', label_visibility='visible',
                options = ['Sigmoid', 'Softmax', 'LogSoftmax'],
            )
            
            '#### Load Path'
            load_path = st.text_input(
                'Path if the model exists', 'None',
            )
            if(load_path.lower()=='none'):
                load_path = None

        #### --- #### --- ####
        cols_2 = st.columns(3)
        with cols_2[0]:
            '#### Optimizer'
            otimizer = st.selectbox(
                'optimizer', label_visibility='collapsed',
                options = ['SGD', 'Adam'],
            )
        with cols_2[1]:
            '#### Learning Rate'
            lr = st.number_input(
                'lr', value=0.03, label_visibility='collapsed',
            )
        with cols_2[2]:
            '#### Weight Decay'
            weight_decay = st.number_input(
                'weight_decay', value=0.05, label_visibility='collapsed',
            )
        
        #### --- #### --- ####
        cols_3 = st.columns(2)
        with cols_3[0]:
            '#### Criterion'
            criterion = st.selectbox(
                'criterion', label_visibility='hidden',
                options = ['BCE', 'NLL', 'CrossEntropy'],
            )
        with cols_3[1]:
            '#### Differential Privacy'
            dp = st.number_input(
                'Noise ∝ 1/DP', value=1000, label_visibility='visible',
            )



        '---'
        submitted = st.form_submit_button("Submit")
        if(submitted):
            st.write(f'''
                {inner_layer, outer_layer, type(inner_layer[-1][-1])}  
                {inner_dropout, outer_dropout, type(inner_dropout[-1][-1])}
            ''')
            # call_api('/define-model')









    key = 'define-model'
    with st.empty():
        if(key in st.session_state):
            st.write(st.session_state[key])

        if(btn):
            data = json.dumps(inputs)
            res = call_api('/define-model', method='POST', data=data).json()
            st.write(res)
            st.session_state[key] = res


# res = requests.post(PATH + '/define-model', json=json.dumps(input_req))

# n = 95
# input_req = {
#     # <Model Build>
#     # inner models: feature holder model
#     # outer model: target holder model(*반드시 리스트의 마지막 순서*)
#     # e.g)
#         # inner_1: nn.Linear( 10, 5)
#         # inner_2: nn.Linear(  8, 4)
#         # outer  : nn.Linear(5+4, 2)
#     "layers": [[n, n*2, n, n//2], [n//2, 1]],
#     "dropout": [[0.1, 0.3, 0], [0]],
#     "batchnorm": True,
#     "feature_activation": "ReLU",        # [Sigmoid, ReLU, SeLU]
#     "target_activation": "Sigmoid",   # [Sigmoid, Softmax, LogSoftmax]

#     # <Training>
#     "optimizer": "Adam",                 # [SGD, Adam]
#     "criterion": "BCE",                  # [BCE, NLL, CrossEntropy]
#     "learning_rate": 0.03,
#     "weight_decay": 0,
#     "batch_size": 8192,
#     "epochs": 75,
#     "test_size": 0,
#     "seed": 123,

#     # <differential privacy>
#         # laplace noise dp epsilon (if false or 0, turns off)
#     "differential_privacy": 10000,

#     # <etc>
#     "target_holder": "bob",               # target holder id
#     "load_path": None,                    # 기 학습된 모델 import하는 경우
#     "print_term": 1,                      # training output print term
# }









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