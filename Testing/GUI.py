import os
from pathlib import Path
import streamlit as st
from metric_learn import *
import pandas as pd
import plotly.express as px
# from algorithms import *
import json
import copy
import matplotlib.pyplot as plt
from dataSetTester import *

DATA_PATH = f"datasets"
DATASETS_PATH = [f"cobras-paper/UCI", f"cobras-paper/CMU_faces", 
                f"cobras-paper/newsgroups" , f"cobras-paper/UCI", f"created" ]


st.set_page_config(page_title="Thesis GUI", page_icon=":snake:")
st.set_option('deprecation.showPyplotGlobalUse', False)

# Initalize
@st.cache
def getDatasets():
    data = {}
    for datasetType in DATASETS_PATH:
        path_data = Path(f'{DATA_PATH}/{datasetType}').absolute()
        dir_list = os.listdir(path_data)
        datasets = []
        for i in range(len(dir_list)):
            datasets.append(dir_list[i][:len(dir_list[i]) - 5])
        data[datasetType] = datasets
    return data
  
@st.cache
def getMetricLearners():
    with open('settings/metricLearners.json') as json_file:
        return json.load(json_file)

datasets = getDatasets()
metrics = getMetricLearners()

if 'results' not in st.session_state:
    st.session_state.results = []

if 'count' not in st.session_state:
    st.session_state.count = {}


def newMetric():
    st.session_state.metric = dict(metrics[typeMetric][st.session_state.selectedMetric])

# ---- SIDEBAR ----
st.sidebar.selectbox(
    'Select a dataset',
    datasets["created"],
    key = "selectedData")
st.sidebar.markdown('---')
isSupervised = st.sidebar.checkbox("Supervised", value = False)
typeMetric = "supervised" if isSupervised else "semisupervised"
st.sidebar.markdown('---')
st.sidebar.selectbox(
    'Select a metric learner',
    metrics[typeMetric],
    key = "selectedMetric",
    on_change=newMetric)
st.sidebar.markdown('---')

if st.sidebar.button('Remember results'):
    learnMetric(None, None, None)

st.sidebar.markdown('---')

st.sidebar.selectbox(
    'Saved results',
    [],
    key = "savedResult",
    on_change=newMetric)


# The chosen metric -> dit principe doen voor de gehele setting
if 'metric' not in st.session_state:
    st.session_state.metric = dict(metrics[typeMetric][st.session_state.selectedMetric])


# ---- Main ----
# slider for COBRAS animation and option for animation
# Container for the results
with st.container():
    col1, col2= st.columns(2)
    with col1:
        st.write("Hier komen de resultaten")
        st.line_chart({"welcome": [1,2,3,4,5,6,7,4,3,2,1]})
    with col2:
        st.write("Hier komt de transformed dataset")
        st.line_chart({"welcome": [1,2,3,4,5,6,7,4,3,2,1]})
st.markdown("""---""")
with st.container():
    cola, colb, colc, cold = st.columns(4)
    with cola:
        if st.button('Learn metric on original'):
                learnMetric(None, None, None)
    with colb:
        if typeMetric == "semisupervised":
                if st.button('Generate new constraints'):
                    getConstraints()
    with colc:
        if st.checkbox('Execture Cobras'):
            executeCobras()
            # execute the plot function once
    with cold:
        if st.button('Learn metric on the transformed dataset'):
                learnMetric(None, None, None)
st.markdown("""---""")
# Container for the selection of the metric learner
with st.container():
    col1, col2 = st.columns(2)
    with col1:
        for key, value in st.session_state.metric.items():
            st.write(f"{key}: {value}")

    with col2:
        st.selectbox(
            'Which parameter do you wanna change',
            st.session_state.metric.keys(),
            key = "parameterToChange")

        parameter = st.session_state.metric[st.session_state.parameterToChange]
        typeParameter = type(parameter)
        
        newParameter = None
        
        if typeParameter == float:
            newParameter = st.number_input("Change to", value = parameter)
        elif typeParameter == int:
            newParameter = st.number_input("Change to", value = parameter)
        elif typeParameter == str:
            newParameter = st.text_input("Change to", value = parameter)
        elif typeParameter == bool:
            newParameter = st.checkbox(st.session_state.parameterToChange, value = parameter)
        elif parameter == None:
            new = st.text_input("Change to", value = parameter)
            if not new == "None" and new:
                ntype = st.selectbox(
                'Choose type to convert to',
                (int, float, str, bool))
                newParameter = ntype(new)
        else:
            st.write("Can't handle this")

        if st.button('Change parameter'):
            st.session_state.metric[st.session_state.parameterToChange] = newParameter
            st.experimental_rerun()


st.write(st.session_state)