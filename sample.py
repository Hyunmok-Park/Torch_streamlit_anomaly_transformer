import configparser
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
                    for i in range(100):
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
        st.write("Window")

        start = st.number_input(label="Start time step", value=15000)
        end = st.number_input(label="End time step", value=17000)

        choices = st.multiselect(
            label="Select feature",
            options = ["cpu", "memory", "disk", "eth", "tcp"],
            default= "cpu"
        )

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

        feature_list = []
        for _ in choices:
            if _ == "cpu":
                feature_list += [0,1,2,3]
            elif _ == "memory":
                feature_list += [4,5,6,7]
            elif _ == "disk":
                feature_list += [8,9,10,11,12,13,14,15]
            elif _ == "eth":
                feature_list += [18,19,20,21]
            elif _ == "tcp":
                feature_list += [22,23,33]


        if len(st.session_state.SMD) != 0:
            if dataset == "SMD":
                test_energy = np.load("./Anomaly_transformer/demo/test_energy.npy")[start:end]
                threshold = 0.12837149113416307

                # test_energy[np.where(test_energy > threshold)] = threshold * 5

                pred = (test_energy > threshold).astype(int)
                gt = np.load("./Anomaly_transformer/demo/test_labels.npy")[start:end]
                feature_value = np.load("./Anomaly_transformer/dataset/SMD/origin/SMD_test.npy")[start:end, feature_list]
                feature_label = [feature_names[_] for _ in feature_list]

            else:
                st.write("Not implemented")

            source1 = create_dataframe([test_energy, threshold], ['Anomaly score', 'Threshold'], ['Step', 'Label', "Anomaly score"], start, end)
            source2 = create_dataframe([feature_value], feature_label, ['Step', 'Label', 'Anomaly score'], start, end)
            source3 = pd.DataFrame(
                np.concatenate([
                    np.array([_ for _ in range(start, end)]).reshape(-1,1),
                    test_energy.reshape(-1,1),
                    np.array([threshold for _ in range(start, end)]).reshape(-1,1),
                    pred.reshape(-1,1),
                    feature_value.reshape(-1, len(feature_list)),
                ], axis=-1),
                columns=['Step', 'Anomaly score','Threshold', 'Prediction'] + list(np.array(feature_names)[feature_list])
            )

            new_chart = get_double_chart(source1, source2, source3, "Anomaly score", "Feature", feature_names, feature_list, gt, start, width=1300, height=200)

            st.altair_chart(
                new_chart,
                use_container_width=True
            )

    ##################################
    #              TEST              #
    ##################################
    st.markdown("***")
    st.subheader("Inference")

    if "inference" not in st.session_state:
        st.session_state.inference = False

    st.caption("* Detect anomaly with your own data")

    # Upload data
    file = st.file_uploader("Upload file", type=['csv', 'npy'])
    if file is not None:
        data_path = os.path.join("./Anomaly_transformer/dataset/test", file.name)
        with open(data_path, "wb") as f:
            f.write((file).getbuffer())

        with st.expander("Preview your dataset"):
            if "csv" in file.name:
                test_dataset = pd.read_csv(file).to_numpy()
                st.write(test_dataset)
                st.line_chart(test_dataset[:1000, 1:])
            else:
                test_dataset = np.load(file)
                st.write(test_dataset)
                st.line_chart(test_dataset[:1000, 1:])
    else:
        st.session_state.inference = False

    # Hyper-parameter setting
    st.caption("* Hyperparameter")
    col1, col2, col3, col4, col5 = st.columns([1,1,1,1,1])
    with col1:
        anomaly_ratio = st.number_input(
            "Anomaly ratio",
            value=0.5
        )
        batch_size = st.number_input(
            "Batch size",
            value=32,
            step=1
        )
        lr = st.number_input(
            "Learning rate",
            value=0.0001,
        )
    with col2:
        epoch = st.number_input(
            "Epoch",
            value=10,
            step=1
        )
        win_size = st.number_input(
            "Window size",
            value=100,
            step=1
        )
        model_path = st.selectbox(
            "Model",
            options=[_ for _ in os.listdir('./Anomaly_transformer/result') if "DS" not in _]
        )
    with col3:
        elayers = st.number_input(
            "Encoder layer",
            value=3,
            step=1
        )
        dmodel = st.number_input(
            "Model dimension",
            value=512,
            step=1
        )
    with col4:
        dff = st.number_input(
            "Hidden dimension",
            value=512,
            step=1
        )
        patience = st.number_input(
            "Early stop",
            value=3,
            step=1
        )
    with col5:
        input_c = st.number_input(
            "Input channel",
            value=38
        )
        k = st.number_input(
            "lambda",
            value=5,
            step=1
        )

    # Test
    btn2 = st.button("Test your data")
    if btn2 and file is not None:
        config = make_config(model_path, "./Anomaly_transformer/dataset/test", anomaly_ratio, epoch, elayers, dff, input_c, batch_size, win_size, dmodel, patience, k, lr)
        st.caption("* Test result")
        solver = Solver(config)
        solver.test()
        st.session_state.inference = True

    if st.session_state.inference:
        start = 0
        end = 200
        test_energy = np.load(os.path.join("./Anomaly_transformer/result", model_path, "test_energy.npy"))[start:end]
        combined_energy = np.load(os.path.join("./Anomaly_transformer/result", model_path, "combined_energy.npy"))
        thresh = np.percentile(combined_energy, 100 - anomaly_ratio)
        data = create_dataframe([test_energy, thresh], ['Anomal Score', 'Threshold'], ['Step', 'Label', 'Value'], start, end)
        chart = get_single_chart(data, "Anomaly Score")

        rules = alt.Chart(pd.DataFrame({'Step': np.where(test_energy > thresh)[0]})).mark_rule(color='red').encode(
            x='Step',
        )

        st.altair_chart(
            chart + rules,
            use_container_width=True
        )

        # st.altair_chart(
        #     alt.vconcat(
        #         chart.interactive(),
        #         test_chart.interactive()
        #     ),
        #     use_container_width=True
        # )


        col1, col2 = st.columns([1,1])
        with col1:
            result = pd.DataFrame(
                np.concatenate([np.array([_ for _ in range(test_energy.shape[0])]).reshape(-1,1), test_energy.reshape(-1,1)], axis=-1),
                columns=['Time points', 'Anomaly score']
            )

            st.write(result)

        with col2:
            anomaly_point = np.where(test_energy > thresh)[0].reshape(-1,1)
            anomaly_score = test_energy[anomaly_point].reshape(-1,1)
            result = pd.DataFrame(
                np.concatenate([anomaly_point, anomaly_score], axis=-1),
                columns=['Anomaly time points', 'Anomaly score']
            )
            st.write(result)

    elif btn2 and file is None:
        st.write("Upload your data")







