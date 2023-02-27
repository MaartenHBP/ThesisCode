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
from settings import *
from scipy.spatial import ConvexHull
import time
from matplotlib.animation import FuncAnimation
import functools
import streamlit.components.v1 as components

plt.style.use("default")

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

if 'current' not in st.session_state:
    st.session_state.current = "0"

if 'changedType' not in st.session_state:
    st.session_state.changedType = False

if 'newM' not in st.session_state:
    st.session_state.newM = 0

if 'newS' not in st.session_state:
    st.session_state.newS = 0



# st.session_state.Rerun = False

def newType():
    st.session_state.typeMetric = "supervised" if st.session_state.isSupervised else "semisupervised"
    st.session_state.changedType = True

def newMetric():
    newParameters = dict(metrics[st.session_state.typeMetric][st.session_state.selectedMetric])
    st.session_state.settings[st.session_state.current].changeMertic(st.session_state.selectedMetric, newParameters, st.session_state.typeMetric)

def newData():
    newData = st.session_state.selectedData
    st.session_state.settings[st.session_state.current].changeData(newData)
    path = Path(f'{DATA_PATH}/created/{newData}.data').absolute()
    data = np.loadtxt(path, delimiter=',')
    st.session_state.original = data

def newName():
    newName = st.session_state.selectedName
    st.session_state.settings[newName] = st.session_state.settings[st.session_state.current].changeName(newName)
    del(st.session_state.settings[st.session_state.current])
    st.session_state.current = newName

def newK():
    newK = st.session_state.clustK
    st.session_state.settings[st.session_state.current].changeK(newK)

def loadSetting():
    st.session_state.current = st.session_state.Selectedsetting
    oldT = st.session_state.typeMetric
    st.session_state.isSupervised = st.session_state.settings[st.session_state.current].typeMetric == "supervised" 
    st.session_state.typeMetric = st.session_state.settings[st.session_state.current].typeMetric
    st.session_state.selectedData = st.session_state.settings[st.session_state.current].data
    st.session_state.clustK = st.session_state.settings[st.session_state.current].k
    if oldT == st.session_state.typeMetric:
        st.session_state.selectedMetric = st.session_state.settings[st.session_state.current].metric
    else:
        st.session_state.newM = list(metrics[st.session_state.typeMetric].keys()).index(st.session_state.settings[st.session_state.current].metric)

    # st.session_state.Rerun = True

    path = Path(f'{DATA_PATH}/created/{st.session_state.settings[st.session_state.current].data}.data').absolute()
    data = np.loadtxt(path, delimiter=',')
    st.session_state.original = data



# ---- SIDEBAR ----
st.sidebar.selectbox(
    'Select a dataset',
    datasets["created"],
    key = "selectedData",
    on_change=newData)

if 'original' not in st.session_state:
    path = Path(f'{DATA_PATH}/created/{st.session_state.selectedData}.data').absolute()
    data = np.loadtxt(path, delimiter=',')
    st.session_state.original = data

st.sidebar.markdown('---')
st.sidebar.checkbox("Supervised", value = False, on_change=newType, key = "isSupervised")
if 'typeMetric' not in st.session_state:
    newType()
    st.session_state.changedType = False
st.sidebar.markdown('---')
st.sidebar.selectbox(
    'Select a metric learner',
    metrics[st.session_state.typeMetric],
    key = "selectedMetric",
    index = st.session_state.newM,
    on_change=newMetric)

if st.session_state.changedType:
    newMetric()
    st.session_state.changedType = False

st.sidebar.markdown('---')

# Hier nog voor de k-means
st.sidebar.number_input("Number of clusters (do not spam)", key = "clustK", step = 1, on_change=newK, value=2)


st.sidebar.markdown('---')
new = st.text_input("Name of this setting", value = "0", on_change=newName, key = "selectedName")

if 'settings' not in st.session_state:
    st.session_state.settings = {st.session_state.selectedName: 
    metricSettings(
        st.session_state.selectedMetric,
        st.session_state.selectedData, 
        dict(metrics[st.session_state.typeMetric][st.session_state.selectedMetric]),
        st.session_state.selectedName,
        st.session_state.typeMetric,
        st.session_state.clustK

    )}

st.sidebar.selectbox(
    'Settings',
    st.session_state.settings.keys(),
    key = "Selectedsetting",
    index = st.session_state.newS,
    on_change=loadSetting)

if st.sidebar.button("New/copy"):
    name = str(len(st.session_state.settings.keys()))
    st.session_state.settings[name] = st.session_state.settings[st.session_state.current].copy(name)
    st.session_state.newS = len(st.session_state.settings.keys()) - 1
    st.session_state.current = name
    st.experimental_rerun()


# ---- Main ----
# slider for COBRAS animation and option for animation
# Container for the results
st.markdown("""---""")
with st.container():
    col1, col2, col3, col4= st.columns(4)
    with col1:
        st.checkbox('Show constraints', key = "showConstraints")
    with col2:
        st.checkbox('Show clustering', key = "showSuper")
with st.container():
    col1, col2= st.columns(2)
    with col1:
        st.write("Original")
        fig, ax = plt.subplots()
        c = st.session_state.original[:, 0]
        if st.session_state.showSuper and not st.session_state.settings[st.session_state.current].clustTransformed is None:
            c = st.session_state.settings[st.session_state.current].clustTransformed
        ax.scatter(x = st.session_state.original[:, 1], y = st.session_state.original[:, 2], c = c)

        if st.session_state.showConstraints and st.session_state.typeMetric == "semisupervised":
            lines = st.session_state.settings[st.session_state.current].pairs
            labels = st.session_state.settings[st.session_state.current].constraints

            colors = {-1: "r", 1: "g"}

            for i in range(len(lines)):
                if labels[i] == -1 and st.session_state.original[:, 0][lines[i,0]] == 1:
                    point1 = st.session_state.original[:, 1:][lines[i,0]]
                    point2 = st.session_state.original[:, 1:][lines[i,1]]
                    x_values = [point1[0], point2[0]]
                    y_values = [point1[1], point2[1]]
                    ax.plot(x_values, y_values, c = colors[labels[i]])

        st.pyplot(fig)

    with col2:
        st.write("Transformed")
        if not st.session_state.settings[st.session_state.current].transformed is None:
            fig, ax = plt.subplots()
            c = st.session_state.original[:, 0]
            if st.session_state.showSuper and not st.session_state.settings[st.session_state.current].clustTransformed is None:
                c = st.session_state.settings[st.session_state.current].clustTransformed
            ax.scatter(x = st.session_state.settings[st.session_state.current].transformed, 
                        y = np.zeros(len(st.session_state.settings[st.session_state.current].transformed)), c = c)
            st.pyplot(fig)
st.markdown("""---""")
with st.container(): # container as
    cola, colb, colc, cold = st.columns(4)
    with cola:
        if st.button('Learn metric on original'):
            st.session_state.settings[st.session_state.current].learnMetric(st.session_state.original[:,1:])
            st.experimental_rerun()
    with colb:
        if st.button('Generate new constraints'):
            st.session_state.settings[st.session_state.current].newConstraints(st.session_state.original, 
            st.session_state.nbconst)
            st.experimental_rerun()
    with colc:
        if st.button('Exectute k-means'):
            st.session_state.settings[st.session_state.current].executeClustering(st.session_state.original[:,1:])
            st.experimental_rerun()
            # execute the plot function once
    with cold:
        if st.button('Learn metric on the transformed dataset'):
            st.session_state.settings[st.session_state.current].learnMetric(st.session_state.original[:,1:], onOrig=False)
            st.experimental_rerun()
with st.container(): # container as
    cola, colb, colc, cold = st.columns(4)
    with colb:
            if st.button('Exectute spectral'):
                st.session_state.settings[st.session_state.current].exectuteSpectral(st.session_state.original[:,1:])
                st.experimental_rerun()
            if st.button('Exectute special spectral'):
                st.session_state.settings[st.session_state.current].executeSpectralSpecial(st.session_state.original[:,1:])
                st.experimental_rerun()
            st.number_input('Number of new constraints', key = 'nbconst', value=5)
st.markdown("""---""")


# Container for the selection of the metric learner
with st.container():
    param = st.session_state.settings[st.session_state.current].parameters
    col1, col2 = st.columns(2)
    with col1:
        for key, value in param.items():
            st.write(f"{key}: {value}")

        if st.button('Reset parameters'):
            newMetric()
            st.experimental_rerun()

    with col2:
        st.selectbox(
            'Which parameter do you wanna change',
            param.keys(),
            key = "parameterToChange")

        parameter = param[st.session_state.parameterToChange]
        typeParameter = type(parameter)
        
        newParameter = None

        newParameter = st.text_input("Change to", value = parameter)
        types = [int, float, str, bool, type(None)]
        ntype = st.selectbox(
                'Choose type to convert to',
                types,
                index = types.index(typeParameter))
        try:
            newParameter = ntype(newParameter)
        except:
            newParameter = None

        if st.button('Change parameter'):
            st.session_state.settings[st.session_state.current].parameters[st.session_state.parameterToChange] = newParameter
            st.experimental_rerun()

        if st.button('Reset parameter'):
            newParameter = metrics[st.session_state.typeMetric][st.session_state.selectedMetric][st.session_state.parameterToChange]
            st.session_state.settings[st.session_state.current].parameters[st.session_state.parameterToChange] = newParameter
            st.experimental_rerun()

st.write(st.session_state)
