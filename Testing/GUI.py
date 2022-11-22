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
import json

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
    with open('settings/algorithms.json') as json_file:
        algos = json.load(json_file)

    with open('settings/metricLearners.json') as json_file:
        metrics = json.load(json_file)

    ask = whatToAsk(algos)
    return algos, datasets, metrics, ask

@st.cache
def whatToAsk(algos):
    dictio = {}
    for key,values in algos.items():
        path, what = findAsk(values)
        dictio[key] = {"path": path, "what": what}
    return dictio

def findAsk(dictio):
    if "type" in dictio.keys():
        if "parameters" in dictio.keys():
            path, dicts = findAsk(dictio["parameters"])
            for i in path:
                i.insert(0, "parameters")
            return path, dicts
        else:
            return [[]], [dictio] 

    bigpath, bigdict = [], []
    for key,value in dictio.items():
        if type(value) is dict:
            path, dicts = findAsk(value)
            for i in path:
                i.insert(0, key)
            bigpath += path
            bigdict += dicts
    return bigpath, bigdict

def get_nested(data, *args):
    if args and data:
        element  = args[0]
        if element:
            value = data.get(element)
            return value if len(args) == 1 else get_nested(value, *args[1:])



algos, datasets, metrics, ask = getInitialValues()

# ge kunt werken met columns en states, maar is ff te advanced

# if 'options' not in st.session_state:
#     options = {}
#     for i in algos:
#         if (metrics[i]):
#             op = MetricAlgos.getAlg(metrics[i])
#             opN = [a.__name__ for a in op]
#             options[i] = [opN[0]]
#         else: options[i] = [""]

#     st.session_state.options = options

if 'plots' not in st.session_state:
    st.session_state.plots = pd.DataFrame({"welcome": [1,2,3,4,5,6,7,4,3,2,1]})

if 'algorithms' not in st.session_state:
    st.session_state.algorithms = {}

if 'nbalgo' not in st.session_state:
    st.session_state.nbalgo = 0

if 'adding' not in st.session_state:
    st.session_state.adding = False


# ---- SIDEBAR ----
st.sidebar.header("Please Filter Here:")
# algo = st.sidebar.multiselect(
#     "Select the algorithms:",
#     options=algos.keys(),
#     default=algos.keys()
# )
# edit = st.sidebar.checkbox("Options for algorithms")

# with st.sidebar.expander("Extra options for algorithms"):
#     for i in algo:
#         if (metrics[i]):
#             op = MetricAlgos.getAlg(metrics[i])
#             opN = [a.__name__ for a in op]
#             options = st.multiselect(
#                 f"Select metric learn for {i}:",
#                 options=opN,
#                 default = opN[0], key = i + "options")





# if edit:
#     for i in algo:
#         if (metrics[i]):
#             st.sidebar.markdown('---')
#             op = MetricAlgos.getAlg(metrics[i])
#             opN = [a.__name__ for a in op]
#             st.session_state.options[i] = st.sidebar.multiselect(
#                 f"Select metric learn for {i}:",
#                 options=opN,
#                 default = st.session_state.options[i])
#             st.write(st.session_state)


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

# if st.sidebar.button('Show result'):
if st.sidebar.button('Show result'):
    not_availabe_data = []
    dfs = []
    available_algos = []
    # for i in algo:
    #     for metr in st.session_state.options[i]:
    #         path = Path(f'batches/{plot_kind}/{i}{metr}_10_crossfold_{fold}').absolute() # format gaat nog veranderen
    #         if not os.path.exists(path):
    #             st.text(f"Missing results of {i}{metr} in this setting")
    #             continue
    #         available_algos.append(i)
    #         df = pd.read_csv(path)
    #         dfs.append(df)
    #         for d in data:
    #             if not d in df:
    #                 not_availabe_data.append(d)
    #                 st.text(f"Missing results for {d} with algorithm {i}{metr} in this setting")
    # available_data = []
    # for d in data:
    #     if d not in not_availabe_data:
    #         available_data.append(d)
    # dfs = [df[available_data] for df in dfs]
    
    # if "Average" in plot_type:
    #     average(dfs, available_algos)


tab1, tab2, tab3 = st.tabs(["Average ARI", "Aligned rank", "Time"])
with tab1:
   st.header("Average ARI")
   st.line_chart(st.session_state.plots)

with tab2:
   st.header("Aligned rank")
   st.image("https://static.streamlit.io/examples/dog.jpg", width=200)

with tab3:
   st.header("Time")
   st.image("https://static.streamlit.io/examples/owl.jpg", width=200)


for key, value in st.session_state.algorithms.items():    
    with st.expander(str(key),expanded = True):
        if not value:
            nameAlgo = st.selectbox(
                "Choose an algorithm",
                options = algos
            )   
            if st.button('Next'):
                st.session_state.algorithms[key]["name algo"] = nameAlgo
                st.session_state.adding = False
                st.experimental_rerun()
        else:
            st.write(value["name algo"])
            # now display all the qustions
            for i in range(len(ask[value["name algo"]]["path"])):
                what = ask[value["name algo"]]["what"][i]["type"]
                string = str(ask[value["name algo"]]["path"][i])
                if what == "supervised" or what == "semisupervised":
                    data = st.multiselect(
                    string,
                    options=metrics[what],
                    key = value["name algo"] + string + str(key)
                )
                if what == "int":
                    st.slider(
                    string, ask[value["name algo"]]["what"][i]["min"], ask[value["name algo"]]["what"][i]["max"],
                    key = value["name algo"] + string + str(key)
                    )




if not st.session_state.adding:
    if st.button('Add algorithm'):
        st.session_state.algorithms[st.session_state.nbalgo] = {}
        st.session_state.nbalgo += 1
        st.session_state.adding = True
        st.experimental_rerun()

st.write(st.session_state)



# if st.button('Save figure'):
#     print("figure saved")


# import json
# s = "{'muffin' : 'lolz', 'foo' : 'kitty'}"
# json_acceptable_string = s.replace("'", "\"")
# d = json.loads(json_acceptable_string)

    


