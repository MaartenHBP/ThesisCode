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

def anim(i, superinstances, clusterIteration, data):
    cluster = i%len(superinstances)
    fig.clear()
    plt.text(0.15,0.3,str(cluster), fontsize = 22)
    plt.scatter(data[:,0], data[:,1], c = clusterIteration[cluster])

    for j in np.unique(superinstances[cluster]):
    # get the convex hull
        points = data[superinstances[cluster] == j]
        if len(points) < 3:
            continue
        hull = ConvexHull(points)
        x_hull = np.append(points[hull.vertices,0],
                        points[hull.vertices,0][0])
        y_hull = np.append(points[hull.vertices,1],
                        points[hull.vertices,1][0])
        
        # interpolate
        # dist = np.sqrt((x_hull[:-1] - x_hull[1:])**2 + (y_hull[:-1] - y_hull[1:])**2)
        # dist_along = np.concatenate(([0], dist.cumsum()))
        # spline, u = interpolate.splprep([x_hull, y_hull], 
        #                                 u=dist_along, s=0, per=1)
        # interp_d = np.linspace(dist_along[0], dist_along[-1], 50)
        # interp_x, interp_y = interpolate.splev(interp_d, spline)
        # plot shape
        plt.fill(x_hull, y_hull, '--', alpha=0.2)


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

if 'animation' not in st.session_state:
    st.session_state.animation = False

if 'animationI' not in st.session_state:
    st.session_state.animationI = 0

st.session_state.Rerun = False

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

def loadSetting():
    st.session_state.current = st.session_state.Selectedsetting
    st.session_state.isSupervised = st.session_state.settings[st.session_state.current].typeMetric == "supervised" 
    st.session_state.typeMetric = st.session_state.settings[st.session_state.current].typeMetric
    st.session_state.selectedData = st.session_state.settings[st.session_state.current].data
    st.session_state.newM = list(metrics[st.session_state.typeMetric].keys()).index(st.session_state.settings[st.session_state.current].metric)

    st.session_state.Rerun = True

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
new = st.text_input("Name of this setting", value = "0", on_change=newName, key = "selectedName")

if 'settings' not in st.session_state:
    st.session_state.settings = {st.session_state.selectedName: 
    metricSettings(
        st.session_state.selectedMetric,
        st.session_state.selectedData, 
        dict(metrics[st.session_state.typeMetric][st.session_state.selectedMetric]),
        st.session_state.selectedName,
        st.session_state.typeMetric

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
    st.experimental_rerun()


# ---- Main ----
# slider for COBRAS animation and option for animation
# Container for the results
st.markdown("""---""")
with st.container():
    col1, col2, col3, col4= st.columns(4)
    with col2:
        st.checkbox('Show constraints', key = "showConstraints")
    with col3:
        st.checkbox('Show superinstances', key = "showSuper")
    with col4:
        if st.button("Start/stop animation"):
            st.session_state.animation = not st.session_state.animation
    with col1:
        st.slider('Iteration of Cobras', 0, 200, key = "i")
    st.markdown("""---""")
with st.container():
    # col1, col2= st.columns(2)
    # with col1:
        st.write("Original")
        fig, ax = plt.subplots()

        ### THE CONVEX HULL PROCEDURE ###
        if st.session_state.showSuper and st.session_state.settings[st.session_state.current].cobrasOriginal:
            # i = st.session_state.i if not st.session_state.animation else st.session_state.animationI
            superinstances = np.copy(st.session_state.settings[st.session_state.current].cobrasOriginal)[0]
            # cluster = i%len(superinstances)
            # fig.clear()
            # ax.text(0.15,0.3,str(cluster), fontsize = 22)
            data = np.copy(st.session_state.original[:, 1:])

            clusterIteration = np.copy(st.session_state.settings[st.session_state.current].cobrasOriginal)[1]

            test = functools.partial(anim, superinstances=superinstances, clusterIteration=clusterIteration,
            data = data)
            animation = FuncAnimation(fig, test, interval = 1000)
            components.html(animation.to_jshtml(), height = 1000)
            # animation.save(r'Animation.mp4')

            # with open("myvideo.html","w") as f:
            #     print(line_ani.to_html5_video(), file=f)

            # HtmlFile = open("myvideo.html", "r")
            # #HtmlFile="myvideo.html"
            # source_code = HtmlFile.read() 
            # components.html(source_code, height = 900,width=900)
            # ax.scatter(data[:,0], data[:,1], c = np.copy(st.session_state.settings[st.session_state.current].cobrasOriginal)[1][cluster])

            # for j in np.unique(superinstances[cluster]):
            # # get the convex hull
            #     points = data[superinstances[cluster] == j]
            #     if len(points) < 3:
            #         continue
            #     hull = ConvexHull(points)
            #     x_hull = np.append(points[hull.vertices,0],
            #                     points[hull.vertices,0][0])
            #     y_hull = np.append(points[hull.vertices,1],
            #                     points[hull.vertices,1][0])
                
            #     # interpolate
            #     # dist = np.sqrt((x_hull[:-1] - x_hull[1:])**2 + (y_hull[:-1] - y_hull[1:])**2)
            #     # dist_along = np.concatenate(([0], dist.cumsum()))
            #     # spline, u = interpolate.splprep([x_hull, y_hull], 
            #     #                                 u=dist_along, s=0, per=1)
            #     # interp_d = np.linspace(dist_along[0], dist_along[-1], 50)
            #     # interp_x, interp_y = interpolate.splev(interp_d, spline)
            #     # plot shape
            #     ax.fill(x_hull, y_hull, '--', alpha=0.2)
        else:
            ax.scatter(x = st.session_state.original[:, 1], y = st.session_state.original[:, 2], c = st.session_state.original[:, 0])
            st.pyplot(fig)

    # with col2:
    #     st.write("Transformed")
    #     if st.session_state.settings[st.session_state.current].transformed:
    #         st.line_chart({"welcome": [1,2,3,4,5,6,7,4,3,2,1]})
st.markdown("""---""")
with st.container():
    cola, colb, colc, cold = st.columns(4)
    with cola:
        if st.button('Learn metric on original'):
                st.session_state.settings[st.session_state.current].learnMetric()
    with colb:
        if st.button('Generate new constraints'):
            st.session_state.settings[st.session_state.current].newConstraints()
    with colc:
        if st.button('Execture Cobras'):
            st.session_state.settings[st.session_state.current].executeCOBRAS(st.session_state.original)
            # execute the plot function once
    with cold:
        if st.button('Learn metric on the transformed dataset'):
            st.session_state.settings[st.session_state.current].learnMetric(onOrig=False)
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

        
        # if typeParameter == float:
        #     newParameter = st.number_input("Change to", value = parameter)
        # elif typeParameter == int:
        #     newParameter = st.number_input("Change to", value = parameter)
        # elif typeParameter == str:
        #     newParameter = st.text_input("Change to", value = parameter)
        # elif typeParameter == bool:
        #     newParameter = st.checkbox(st.session_state.parameterToChange, value = parameter)
        # elif parameter == None:
        #     new = st.text_input("Change to", value = parameter)
        #     if not new == "None" and new:
        #         ntype = st.selectbox(
        #         'Choose type to convert to',
        #         (int, float, str, bool))
        #         newParameter = ntype(new)
        # else:
        #     st.write("Can't handle this")

        if st.button('Change parameter'):
            st.session_state.settings[st.session_state.current].parameters[st.session_state.parameterToChange] = newParameter
            st.experimental_rerun()

        if st.button('Reset parameter'):
            newParameter = metrics[st.session_state.typeMetric][st.session_state.selectedMetric][st.session_state.parameterToChange]
            st.session_state.settings[st.session_state.current].parameters[st.session_state.parameterToChange] = newParameter
            st.experimental_rerun()

st.write(st.session_state)

# if (st.session_state.animation):
#     if st.session_state.animationI == 10:
#         st.session_state.animation = False
#     st.session_state.animationI += 1
#     time.sleep(1)
#     st.experimental_rerun()