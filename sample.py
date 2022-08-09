import configparser
import pickle
import time

from utils import *
from Anomaly_transformer.model.AnomalyTransformer import *
from Anomaly_transformer.main import *
from Anomaly_transformer.solver import *

import streamlit as st
import numpy as np
from streamlit_option_menu import option_menu
from pathlib import Path
import csv

import altair as alt
import pandas as pd

st.set_page_config(layout="wide")

with st.sidebar:
    selected = option_menu(
        menu_title="Menu",
        options = ["Home", "Example"],
        icons=["house", "book", "envelope"],
        menu_icon="cast",
        default_index=1,
        # orientation="horizontal"
    )

def read_markdown_file(markdown_file):
    return Path(markdown_file).read_text()


if selected == "Home":
    intro_markdown = read_markdown_file("./static/introduction.md")
    st.markdown(intro_markdown, unsafe_allow_html=True)

if selected == "Example":
    if "SMD" not in st.session_state:
        st.session_state.SMD = []

    st.title("Anomaly Detection")
    st.caption("* Train anomaly transformer(deep learning based anomlay detection)")
    col1, col2 = st.columns([1, 3])

    with col1:
        st.subheader("Training")

        dataset = st.radio("Dataset", ("SMD", "PSM"))
        btn1 = st.button(label="Start training")

        if btn1:
            ########################
            # Train the model
            ########################
            if dataset == "SMD":
                st.session_state.SMD = []
                rec_loss = np.load("./Anomaly_transformer/demo/SMD_rec_list.npy").reshape(-1,1)
                assdis_loss = np.load("./Anomaly_transformer/demo/SMD_assdis_list.npy").reshape(-1,1)
                data = create_dataframe([rec_loss, assdis_loss], ['Reconstruction error', 'Association discrepancy'], ['Step', 'Label', 'Value'], 0, rec_loss.shape[0])
                chart = get_single_chart(data, "Training Loss")
                st.session_state.SMD.append(chart)

                with st.spinner(text="In progress... it takes little time..."):
                    st.write(f"Training {dataset} dataset using default parameter")
                    my_bar = st.progress(0)
                    for i in range(1):
                        time.sleep(0.1)
                        my_bar.progress(i + 1)
                    time.sleep(1)
                my_bar.empty()

            else:
                st.write("Not implemented")

        ########################
        # Show train metrics
        ########################
        if len(st.session_state.SMD) != 0:
            st.success("Training finished!")
            chart = st.session_state.SMD[0]
            st.altair_chart(
                (chart).interactive(),
                use_container_width=True
            )
    with col2:
        st.subheader("Test")
        feature_names = [
            'cpu_r',  #cpu
            'load_1', #cpu load
            'load_5',
            'load_15',
            'mem_shmem',
            'mem_u',
            'mem_u_e',
            'total_mem',
            'disk_q',
            'disk_r',
            'disk_rb',
            'disk_svc',
            'disk_u',
            'disk_w',
            'disk_wa',
            'disk_wb',
            'si',
            'so',
            'eth1_fi',
            'eth1_fo',
            'eth1_pi',
            'eth1_po',
            'tcp_tw',
            'tcp_use',
            'active_opens',
            'curr_estab',
            'in_errs',
            'in_segs',
            'listen_overflows',
            'out_rsts',
            'out_segs',
            'passive_opens',
            'retransegs',
            'tcp_timeouts',
            'udp_in_dg',
            'udp_out_dg',
            'udp_rcv_buf_errs',
            'udp_snd_buf_errs'
        ]

        if len(st.session_state.SMD) != 0:
            if dataset == "SMD":
                start, end = 15700,16500

                test_energy = np.load("./Anomaly_transformer/demo/test_energy.npy")
                train_energy = np.load("./Anomaly_transformer/demo/SMD_train_energy.npy")
                threshold = np.percentile(np.concatenate([train_energy, test_energy], axis=0), 100-0.5)

                test_energy = test_energy[start:end]

                feature_value = np.load("./Anomaly_transformer/demo/SMD_test.npy")[start:end]
                interpret = pickle.load(open("./Anomaly_transformer/demo/SMD_interpret_v2.p", "rb"))

            else:
                st.write("Not implemented")

            pred = (test_energy > threshold).astype(int)
            anomal_point = np.where(pred == 1)[0]

            # Plot feature and window
            slider = int(st.slider("Slider", start, end-100))
            source1 = create_dataframe([feature_value[:1000, [0,5,10,15,20,25,35]]], [feature_names[_] for _ in [0,5,10,15,20,25,35]], ['Step', 'Label', 'Value'], start, end)
            line1 = get_single_chart(source1, "Feature")
            cutoff = pd.DataFrame({
                'index': ['blue'],
                'start': slider,
                'stop': slider + 100
            })
            areas = alt.Chart(
                cutoff
            ).mark_rect(
                opacity=0.2
            ).encode(
                x='start',
                x2='stop',
                y=alt.value(0),  # pixels from top
                y2=alt.value(300),  # pixels from top
                color=alt.Color('index:N', scale=None)
            )
            st.altair_chart(
                line1 + areas,
                use_container_width=True
            )

            # Plot anomaly score in window
            select_anomaly = st.selectbox(
                "Detected anomaly points",
                options=["---"] + [f"{__ + start}" for _, __ in enumerate(anomal_point)],
                index=0,
            )

            if select_anomaly == "---":
                source2 = create_dataframe([test_energy[slider - start : slider + 100 - start], threshold], ['Anomaly score', 'Threshold'], ['Step', 'Label', "Value"], slider, slider + test_energy[slider - start : slider + 100 - start].shape[0])
                line2 = get_single_chart(source2, "Anomaly score")

                start_ = []
                end_ = []
                gt = np.load("./Anomaly_transformer/demo/test_labels.npy")[start:end][slider - start : slider + 100 - start]

                for idx,i in enumerate(np.where(gt == 1)[0]):
                    if idx == 0:
                        current_num = i
                        start_.append(i+slider)
                    else:
                        if i - current_num == 1:
                            current_num = i
                            if (idx+1) == len(np.where(gt == 1)[0]):
                                end_.append(i+slider)
                            continue
                        else:
                            end_.append(current_num+slider)
                            start_.append(i+slider)
                            current_num = i

                cutoff = pd.DataFrame({
                    'index': ['red' for _ in range(len(start_))],
                    'start': start_,
                    'stop': end_
                })

                areas = alt.Chart(
                    cutoff
                ).mark_rect(
                    opacity=0.2
                ).encode(
                    x='start',
                    x2='stop',
                    y=alt.value(0),  # pixels from top
                    y2=alt.value(300),  # pixels from top
                    color=alt.Color('index:N', scale=None)
                )

                st.altair_chart(
                    line2 + areas,
                    use_container_width=True
                )
            else:
                slider = int(select_anomaly) - 100
                source2 = create_dataframe([test_energy[slider - start : slider + 200 - start], threshold], ['Anomaly score', 'Threshold'], ['Step', 'Label', "Value"], slider, slider + test_energy[slider - start : slider + 200 - start].shape[0])
                line2 = get_single_chart(source2, "Anomaly score")
                rules = alt.Chart(pd.DataFrame({'Step': [int(select_anomaly)]})).mark_rule(color='red').encode(
                    x='Step:Q',
                )

                start_ = []
                end_ = []
                gt = np.load("./Anomaly_transformer/demo/test_labels.npy")[start:end][slider - start : slider + 200 - start]

                for idx,i in enumerate(np.where(gt == 1)[0]):
                    if idx == 0:
                        current_num = i
                        start_.append(i+slider)
                    else:
                        if i - current_num == 1:
                            current_num = i
                            if (idx+1) == len(np.where(gt == 1)[0]):
                                end_.append(i+slider)
                            continue
                        else:
                            end_.append(current_num+slider)
                            start_.append(i+slider)
                            current_num = i
                cutoff = pd.DataFrame({
                    'index': ['red' for _ in range(len(start_))],
                    'start': start_,
                    'stop': end_
                })

                areas = alt.Chart(
                    cutoff
                ).mark_rect(
                    opacity=0.2
                ).encode(
                    x='start',
                    x2='stop',
                    y=alt.value(0),  # pixels from top
                    y2=alt.value(300),  # pixels from top
                    color=alt.Color('index:N', scale=None)
                )

                st.altair_chart(
                    line2 + rules + areas,
                    use_container_width=True
                )

                # Plot interpretation
                anomaly_feature = None
                if int(select_anomaly) in interpret:
                    anomaly_feature = interpret[int(select_anomaly)]
                else:
                    for _ in range(slider, slider+101):
                        if _ in interpret:
                            anomaly_feature = interpret[_]
                            break

    st.caption("* Cause of abnormality")
    try:
        col1_, col2_, col3_ = st.columns([1,1,1])
        for idx, _ in enumerate(anomaly_feature):
            if idx & 3 == 0:
                with col1_:
                    source = create_dataframe([feature_value[slider - start : slider + 200 - start, _]], [feature_names[_]], ['Step', 'Label', 'Value'], slider, slider + test_energy[slider - start : slider + 200 - start].shape[0])
                    line = get_single_chart(source, f"{feature_names[_]}")
                    st.altair_chart(
                        line + rules,
                        use_container_width=True
                    )
            elif idx & 3 == 1:
                with col2_:
                    source = create_dataframe([feature_value[slider - start : slider + 200 - start, _]], [feature_names[_]], ['Step', 'Label', 'Value'], slider, slider + test_energy[slider - start : slider + 200 - start].shape[0])
                    line = get_single_chart(source, f"{feature_names[_]}")
                    st.altair_chart(
                        line + rules,
                        use_container_width=True
                    )
            elif idx & 3 == 2:
                with col3_:
                    source = create_dataframe([feature_value[slider - start : slider + 200 - start, _]], [feature_names[_]], ['Step', 'Label', 'Value'], slider, slider + test_energy[slider - start : slider + 200 - start].shape[0])
                    line = get_single_chart(source, f"{feature_names[_]}")
                    st.altair_chart(
                        line + rules,
                        use_container_width=True
                    )
    except:
        st.write("There is no anomaly feature")


    ##################################
    #       ONLINE DETECTING         #
    ##################################
    st.markdown("***")
    st.subheader("Online detecting")





