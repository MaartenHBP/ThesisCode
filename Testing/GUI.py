# import PySimpleGUI as sg

# sg.theme('DarkTeal9')
import os
from pathlib import Path
import streamlit as st
from metric_learn import *
import pandas as pd
import plotly.express as px
from algorithms import *
from metricalgos import *

st.set_page_config(page_title="Experiment Runner", page_icon=":bar_chart:")
st.set_option('deprecation.showPyplotGlobalUse', False)

def aligned(dfs, algos):
    df = (
    pd.concat(dfs)
    # replace 0 values with nan to exclude them from mean calculation
    # group by the row within the original dataframe
    .groupby(level=0)
    # calculate the mean
    .mean())
    for frames in df:   
        print("yeet")

def average(dfs, algos):
    plot = pd.DataFrame()
    for i in range(len(algos)):
        plot[algos[i]] = dfs[i].mean(axis=1)[0:amount]

    st.session_state.plots = plot

#  The datasets
@st.cache
def getInitialValues():
    path_data = Path('datasets/cobras-paper/').absolute()
    dir_list = os.listdir(path_data)
    datasets = []
    for i in range(len(dir_list)):
        datasets.append(dir_list[i][:len(dir_list[i]) - 5])

    # The algos
    algos = Algorithm.getAlgos()
    metrics = Algorithm.needsMetric(algos)
    return algos, datasets, metrics

algos, datasets, metrics = getInitialValues()

# ge kunt werken met columns en states, maar is ff te advanced

if 'options' not in st.session_state:
    options = {}
    for i in algos:
        if (metrics[i]):
            op = MetricAlgos.getAlg(metrics[i])
            opN = [a.__name__ for a in op]
            options[i] = [opN[0]]
        else: options[i] = [""]

    st.session_state.options = options

if 'plots' not in st.session_state:
    st.session_state.plots = pd.DataFrame({"welcome": [1,2,3,4,5,6,7,4,3,2,1]})

if 'notRunned' not in st.session_state:
    st.session_state.notRunned = []


# ---- SIDEBAR ----
st.sidebar.header("Please Filter Here:")
algo = st.sidebar.multiselect(
    "Select the algorithms:",
    options=algos,
    default=algos
)
st.sidebar.markdown('---')
edit = st.sidebar.checkbox("Options for algorithms")

if edit:
    for i in algo:
        if (metrics[i]):
            st.sidebar.markdown('---')
            op = MetricAlgos.getAlg(metrics[i])
            opN = [a.__name__ for a in op]
            options = st.sidebar.multiselect(
                f"Select metric learn for {i}:",
                options=opN,
                default = st.session_state.options[i])
            st.session_state.options[i] = options

st.sidebar.markdown('---')
plot_type = st.sidebar.multiselect(
    "Kind of plot:",
    options=["Average", "Aligned rank"],
    default=["Average"]
)

plot_kind = st.sidebar.radio(
    "What to plot", 
    options = ["ARI", "Time"]
)

st.sidebar.markdown('---')
amount = st.sidebar.slider(
    "Select number of queries:", 1, 200, value = 200
)

st.sidebar.markdown('---')
crossfold = st.sidebar.checkbox("Crossfold", value = True)
runsPQ = 10
fold = 0
crossStr = ""

if not crossfold:
    runsPQ = st.sidebar.slider(
    "How many runs:", 1, 200
    )
else:
    fold = st.sidebar.slider(
    "Which fold:", 1, 10
    )
    crossStr = "crossfold"

st.sidebar.markdown('---')

allData = st.sidebar.checkbox("All the data", value = True)
data = datasets
if not allData:
    data = st.sidebar.multiselect(
    "Select the datasets:",
    options=datasets,
    default=datasets
)

st.sidebar.markdown('---')
allData = st.sidebar.checkbox("Add to next run when needed", value = False)

st.sidebar.markdown('---')
update = st.sidebar.checkbox("Update automatically", value = False)

st.sidebar.markdown('---')

# if st.sidebar.button('Show result'):
if st.sidebar.button('Show result') or update:
    not_availabe_data = []
    dfs = []
    available_algos = []
    for i in algo:
        for metr in st.session_state.options[i]:
            path = Path(f'batches/{plot_kind}/{i}{metr}_10_crossfold_{fold}').absolute()
            if not os.path.exists(path):
                st.text(f"Missing results of {i}{metr} in this setting")
                continue
            available_algos.append(i)
            df = pd.read_csv(path)
            dfs.append(df)
            for d in data:
                if not d in df:
                    not_availabe_data.append(d)
                    st.text(f"Missing results for {d} with algorithm {i}{metr} in this setting")
    available_data = []
    for d in data:
        if d not in not_availabe_data:
            available_data.append(d)
    dfs = [df[available_data] for df in dfs]
    
    if "Average" in plot_type:
        average(dfs, available_algos)

if "Average" in plot_type:
    st.session_state.plots.plot(title="Average Aligned rank", xlabel="Number of queries", ylabel="Aligned rank")
    st.pyplot()


if st.button('Save figure'):
    print("figure saved")


# import json
# s = "{'muffin' : 'lolz', 'foo' : 'kitty'}"
# json_acceptable_string = s.replace("'", "\"")
# d = json.loads(json_acceptable_string)

    


