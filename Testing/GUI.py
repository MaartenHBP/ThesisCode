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
import copy

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

def addToTheQueueu(algo, settings, foldnb, crossfold, data, runsPQ, string):
    # ask[value["name algo"]]["path"]
    nal = st.session_state.algorithms[algo]
    with open('settings/algorithms.json') as json_file:
        sett = json.load(json_file)[nal["name algo"]]
    for i in range(len(ask[nal["name algo"]]["path"])):
        print(settings[i])
        get_nested(sett, *ask[nal["name algo"]]["path"][i], sett = settings[i])
    output = {
        "foldnb": foldnb,
        "crossfold": crossfold,
        "runsPQ": runsPQ,
        "data": data,
        "settings": sett,
        "string": string

    }
    with open(f"queue/{string}.json", "w") as outfile:
        json.dump(output, outfile, indent=4)

    

#  The datasets
@st.cache
def getInitialValues():
    data = {}
    for j in ["cobras-paper/UCI","cobras-paper/CMU_faces", "cobras-paper/newsgroups" , "cobras-paper/UCI", "drawn" ]: # ff hardgecoded
        path_data = Path(f'datasets/{j}').absolute()
        dir_list = os.listdir(path_data)
        datasets = []
        for i in range(len(dir_list)):
            datasets.append(dir_list[i][:len(dir_list[i]) - 5])
        data[j] = datasets

    # The algos
    with open('settings/algorithms.json') as json_file:
        algos = json.load(json_file)

    with open('settings/metricLearners.json') as json_file:
        metrics = json.load(json_file)

    ask = whatToAsk(algos)
    return algos, data, metrics, ask

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

def get_nested(data, *args, sett):
    if args and data:
        element  = args[0]
        if element:
            value = data.get(element)
            if len(args) == 1:
                data[element] = sett 
            else: get_nested(value, *args[1:], sett = sett)



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

if 'metrics' not in st.session_state:
    st.session_state.metrics = {"supervised" : [], "semisupervised" : []}

if 'metricsoptions' not in st.session_state:
    st.session_state.metricsoptions = {}    

if 'nbalgo' not in st.session_state:
    st.session_state.nbalgo = 0

if 'adding' not in st.session_state:
    st.session_state.adding = False

if 'addingm' not in st.session_state:
    st.session_state.addingm = False

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
    crossStr = "crossfold_"

st.sidebar.markdown('---')

# allData = st.sidebar.checkbox("All the data", value = True)

st.sidebar.write("Choose datasets")
optda = st.sidebar.multiselect(
    "Choose datasets folders",
    options=datasets.keys(),
    default=list(datasets.keys())[0])
data = {}
for key in optda:
    st.sidebar.markdown('---')
    st.sidebar.write(key)
    data[key] = []
    options=datasets[key]
    for i in options:
        agree = st.sidebar.checkbox(i, value = True)
        if agree:
            data[key].append(i)
st.sidebar.markdown('---')


addToQueue = st.sidebar.checkbox("Add runs to queue when needed", value = True)

st.sidebar.markdown('---')
# if st.sidebar.button('Show result'):
if st.sidebar.button('Show result'):
    not_availabe_data = []
    dfs = []
    available_algos = []
    for key, value in st.session_state.algorithms.items(): # momenteel alleen voor ARI en aligned rank
        algs = [""]
        settings = [[]]
        # path = Path(f'batches/ARI/{value["name algo"]}{str}_10_crossfold_{fold}').absolute()
        for i in range(len(ask[value["name algo"]]["path"])):
            string = str(ask[value["name algo"]]["path"][i])
            pl = st.session_state[value["name algo"] + string + str(key)]
            what = ask[value["name algo"]]["what"][i]["type"]
            extra = [pl]
            extrasetting=[pl]
            if hasattr(pl, "__len__"):
                if len(pl) == 0:
                    print("Wel iets invullen he")
                    break
                extra = [v for v in pl]
                extrasetting = [v for v in pl]
                if what == "supervised" or what == "semisupervised":
                    extra = []
                    extrasetting = []
                    for k in pl:
                        for option in st.session_state[value["name algo"] + string + str(key) + str(k) + "options"]:
                            # extra = [v + "_" + str(metrics[what][v].values()) for v in pl]
                            # extrasetting = [{"type": "class", "value": v, "parameters": copy.deepcopy(metrics[what][v])} for v in pl]
                            dictPar = {"type": "class", "value": k, "parameters": copy.deepcopy(metrics[what][k])}

                            for paramKey, paramValue in dictPar["parameters"].items():
                                if type(paramValue) is dict: 
                                    dictPar["parameters"][paramKey] = st.session_state[k + str(option) + paramKey]

                            print(dictPar)
                            extra.append(k + "_" + str(dictPar["parameters"].values()))
                            extrasetting.append(dictPar)


            newalgs = []
            newsettings = []
            for i in range(len(algs)):
                for j in range(len(extra)):
                    newalgs.append(str(algs[i])+"_"+str(extra[j]))
                    newS = settings[i].copy()
                    newS.append(extrasetting[j])
                    newsettings.append(newS)
            settings = newsettings
            algs = newalgs
        for al in range(len(algs)):
            a = algs[al].replace(" ", "").replace("[", "").replace("]","").replace(",", "_").replace("'","")
            p = f'{value["name algo"]}{a}_{runsPQ}_{crossStr}{fold}'
            path = Path(f'batches/ARI/{p}')
            if not os.path.exists(path):
                print(f"Missing results of {p} in this setting")
                if (addToQueue):
                    addToTheQueueu(key, settings[al], fold, crossfold, data, runsPQ, p)
                continue
            available_algos.append(a)
            df = pd.read_csv(path)
            dfs.append(df)
            for d in data:
                if not d in df:
                    not_availabe_data.append(d)
                    print(f"Missing results for {d} with algorithm {p} in this setting")
    available_data = []
    for d in data:
        if d not in not_availabe_data:
            available_data.append(d)
    dfs = [df[available_data] for df in dfs]

        # make the string of the data
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

col1, col2 = st.columns(2)

with col1:
    st.write("Algorithms")
    if st.session_state.algorithms:
        tabs = st.tabs([str(i) for i in st.session_state.algorithms.keys()])
        for nbTab in range(len(tabs)): 
            with tabs[nbTab]:
                key = nbTab
                value = st.session_state.algorithms[nbTab]
                # with st.expander(str(key),expanded = True):
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
                            st.multiselect(
                            string,
                            options=st.session_state.metrics[what],
                            key = value["name algo"] + string + str(key))

                            for k in st.session_state[value["name algo"] + string + str(key)]:
                                st.multiselect(
                                f"Configurations for {k}",
                                options=st.session_state.metricsoptions[k],
                                key = value["name algo"] + string + str(key) + str(k) + "options")
                            # with st.expander("metrics"):
                            #     for metricOptions in st.session_state[value["name algo"] + string + str(key)]:
                            #         st.write(metricOptions)
                            #         # st.write(metrics[what][metricOptions])
                            #         for keyd,valued in metrics[what][metricOptions].items():
                            #             if type(valued) is dict:
                            #                 if valued["type"] == "int":
                            #                         st.slider(
                            #                         str(keyd), valued["min"], valued["max"], value = valued["standard"],
                            #                         key = value["name algo"] + string + str(key) + str(metricOptions) + str(keyd)
                            #                         )
                            #                 if valued["type"] == "number":
                            #                     st.number_input(
                            #                         str(keyd), value = -1,
                            #                         key = value["name algo"] + string + str(key) + str(metricOptions) + str(keyd) 
                            #                     )

                        if what == "int":
                            st.slider(
                            string, ask[value["name algo"]]["what"][i]["min"], ask[value["name algo"]]["what"][i]["max"],
                            key = value["name algo"] + string + str(key)
                        )
                        if what == "Boolean":
                            st.multiselect(
                            string,
                            options = [True, False],
                            default=[True, False],
                            key = value["name algo"] + string + str(key)
                        )
    if not st.session_state.adding:
        if st.button('Add algorithm'):
            st.session_state.algorithms[st.session_state.nbalgo] = {}
            st.session_state.nbalgo += 1
            st.session_state.adding = True
            st.experimental_rerun()

with col2:
    st.write("Metric learners")
    if st.session_state.addingm: 
        what = st.selectbox(
                        "Choose an algorithm",
                        options = ["semisupervised", "supervised"],
                    )  

        name = st.selectbox(
                        "Choose an algorithm",
                        options = list(set(metrics[what].keys()) - set(st.session_state.metrics[what]))
                    )   
        if st.button('Next step'):
            if not name is None:
                st.session_state.metrics[what].append(name)
                if name not in st.session_state.metricsoptions:
                    st.session_state.metricsoptions[name] = []
            st.session_state.addingm = False
            st.experimental_rerun()
    else:
        op = ["semisupervised", "supervised"]
        tbs = st.tabs(op)
        for tb in range(len(tbs)):
            with tbs[tb]:
                opt = op[tb]
                if st.session_state.metrics[opt]:
                    tabs = st.tabs([str(i) for i in st.session_state.metrics[opt]])
                    for nbTab in range(len(tabs)): 
                        with tabs[nbTab]:
                            nameMetric = st.session_state.metrics[opt][nbTab]
                            for i in st.session_state.metricsoptions[nameMetric]:
                                with st.expander(str(i)):
                                    for key,value in metrics[opt][nameMetric].items():
                                        if type(value) is dict: 
                                            if value["type"] == "int":
                                                st.slider(
                                                    str(key), value["min"], value["max"], value = value["standard"],
                                                    key = nameMetric + str(i) + key
                                                    )
                                            if value["type"] == "number":
                                                st.number_input(
                                                str(key), step = 1, value = 42,
                                                    key = nameMetric + str(i) + key
                                                )
                            if st.button('Add configuration to ' + nameMetric):
                                st.session_state.metricsoptions[st.session_state.metrics[opt][nbTab]].append(len(st.session_state.metricsoptions[st.session_state.metrics[opt][nbTab]]))
                                st.experimental_rerun()
                        

    if not st.session_state.addingm:        
        if st.button('Add metric algo'):
            st.session_state.addingm = True
            st.experimental_rerun()
# if st.session_state.algorithms:
#     tabs = st.tabs([str(i) for i in st.session_state.algorithms.keys()])
#     for nbTab in range(len(tabs)): 
#         with tabs[nbTab]:
#             key = nbTab
#             value = st.session_state.algorithms[nbTab]
#             # with st.expander(str(key),expanded = True):
#             if not value:
#                 nameAlgo = st.selectbox(
#                     "Choose an algorithm",
#                     options = algos
#                 )   
#                 if st.button('Next'):
#                     st.session_state.algorithms[key]["name algo"] = nameAlgo
#                     st.session_state.adding = False
#                     st.experimental_rerun()
#             else:
#                 st.write(value["name algo"])
#                 # now display all the qustions
#                 for i in range(len(ask[value["name algo"]]["path"])):
#                     what = ask[value["name algo"]]["what"][i]["type"]
#                     string = str(ask[value["name algo"]]["path"][i])
#                     if what == "supervised" or what == "semisupervised":
#                         st.multiselect(
#                         string,
#                         options=metrics[what],
#                         default=metrics[what],
#                         key = value["name algo"] + string + str(key))
#                         with st.expander("metrics"):
#                             for metricOptions in st.session_state[value["name algo"] + string + str(key)]:
#                                 st.write(metricOptions)
#                                 # st.write(metrics[what][metricOptions])
#                                 for keyd,valued in metrics[what][metricOptions].items():
#                                     if type(valued) is dict:
#                                         if valued["type"] == "int":
#                                              st.slider(
#                                                 str(keyd), valued["min"], valued["max"], value = valued["standard"],
#                                                 key = value["name algo"] + string + str(key) + str(metricOptions) + str(keyd)
#                                                 )
#                                         if valued["type"] == "number":
#                                             st.number_input(
#                                                str(keyd), value = -1,
#                                                 key = value["name algo"] + string + str(key) + str(metricOptions) + str(keyd) 
#                                             )

#                     if what == "int":
#                         st.slider(
#                         string, ask[value["name algo"]]["what"][i]["min"], ask[value["name algo"]]["what"][i]["max"],
#                         key = value["name algo"] + string + str(key)
#                     )
#                     if what == "Boolean":
#                         st.multiselect(
#                         string,
#                         options = [True, False],
#                         default=[True, False],
#                         key = value["name algo"] + string + str(key)
#                     )

st.write(st.session_state)



# if st.button('Save figure'):
#     print("figure saved")


# import json
# s = "{'muffin' : 'lolz', 'foo' : 'kitty'}"
# json_acceptable_string = s.replace("'", "\"")
# d = json.loads(json_acceptable_string)

    


