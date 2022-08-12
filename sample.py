import os
import shutil

from utils import *
from Anomaly_transformer.solver import *

import streamlit as st
from streamlit_option_menu import option_menu

import numpy as np
from pathlib import Path
import altair as alt
import pandas as pd
import pickle
import time

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

def reset(option=False):
    begin, end = 17800, 21300
    if "b1_train_chart" not in st.session_state:
        st.session_state.b1_train_chart = None     # train matrics figure
        st.session_state.b1_anomaly_feature = None # selected anomaly
        st.session_state.b1_accuracy = 0.9913
        st.session_state.b1_precision = 0.8603
        st.session_state.b1_recall = 0.9459
        st.session_state.b1_f_score = 0.9011
        st.session_state.b1_rec_loss = np.load("./Anomaly_transformer/demo/SMD_2022_08_12_12_03_28/SMD_rec_list.npy").reshape(-1,1)
        st.session_state.b1_assdis_loss = np.load("./Anomaly_transformer/demo/SMD_2022_08_12_12_03_28/SMD_assdis_list.npy").reshape(-1,1)

        st.session_state.b1_test_energy = np.load("./Anomaly_transformer/demo/SMD_2022_08_12_12_03_28/SMD_2022_08_12_15_40_40/test_energy.npy")
        st.session_state.b1_test_label = np.load("./Anomaly_transformer/demo/SMD_2022_08_12_12_03_28/SMD_2022_08_12_15_40_40/test_labels.npy")
        st.session_state.b1_train_energy = np.load("./Anomaly_transformer/demo/SMD_2022_08_12_12_03_28/train_energy.npy")
        st.session_state.b1_feature_value = np.load("./Anomaly_transformer/demo/SMD_test.npy")
        st.session_state.b1_interpret_v2 = pickle.load(open("./Anomaly_transformer/demo/SMD_interpret_v2.p", "rb"))
        st.session_state.b1_threshold = np.percentile(np.concatenate([st.session_state.b1_train_energy, st.session_state.b1_test_energy], axis=0), 100-0.5)

        st.session_state.simul_test_label = st.session_state.b1_test_label[begin:end]
        st.session_state.simul_feature_value = st.session_state.b1_feature_value[begin:end]
        st.session_state.simul_win_size = end - begin

        np.save("./Anomaly_transformer/demo/online/SMD_test.npy", st.session_state.simul_feature_value)
        np.save("./Anomaly_transformer/demo/online/SMD_test_label.npy", st.session_state.b1_test_label[begin: end])

        st.session_state.simul_online_chart = None # online detecting figure
        st.session_state.simul_anomaly_list = {}   # online detected anomalies
        st.session_state.simul_unique_anomaly_list = [] # unique online detected anomalies
        st.session_state.simul_time_list = []      # online detecting inference time

    if option:
        st.session_state.simul_test_label = st.session_state.b1_test_label[begin:end]
        st.session_state.simul_feature_value = st.session_state.b1_feature_value[begin:end]
        st.session_state.simul_win_size = end - begin

        np.save("./Anomaly_transformer/demo/online/SMD_test.npy", st.session_state.simul_feature_value)
        np.save("./Anomaly_transformer/demo/online/SMD_test_label.npy", st.session_state.b1_test_label[begin: end])

        st.session_state.simul_online_chart = None
        st.session_state.simul_anomaly_list = {}
        st.session_state.simul_unique_anomaly_list = []
        st.session_state.simul_time_list = []

if selected == "Home":
    intro_markdown = read_markdown_file("./static/introduction.md")
    st.markdown(intro_markdown, unsafe_allow_html=True)

if selected == "Example":
    reset()
    model_win_size = 50

    feature_names = [
        'cpu_r',  #cpu  'cpu_r',
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

    st.title("Anomaly Detection")
    st.caption("* Train anomaly transformer(deep learning based anomlay detection)")
    st.subheader("Hyperparameter")
    hyper_cont = st.container()
    col1, col2 = st.columns([1, 2])

    ### 1. Training
    with hyper_cont:
        hyper_col1, hyper_col2, hyper_col3, hyper_col4 = st.columns([1,1,1,1])
        with hyper_col1:
            dataset = st.radio("Dataset", ("SMD", "PSM"), key=0)
            layers = st.radio("Layers", ("3", "4", "5"), key=1, index=2)
        with hyper_col2:
            batch = st.radio("Batch size", ("32", "64"), key=3)
            anomaly_ratio = st.radio("Anomaly ratio", ("0.1", "0.5"), key=4, index=1)
        with hyper_col3:
            lr = st.radio("Learning rate", ("0.001", "0.0001"), key=2, index=1)
            epoch = st.radio("Epoch", ("10", "20"), key=3)
        with hyper_col4:
            dmodel = st.radio("Model dimension", ("512", "1024"), key=4, index=0)
        btn1 = st.button(label="Start training")
        if btn1:
            if dataset == "SMD":
                data = create_dataframe([st.session_state.b1_rec_loss, st.session_state.b1_assdis_loss], ['Reconstruction error', 'Association discrepancy'], ['Step', 'Label', 'Value'], 0, st.session_state.b1_rec_loss.shape[0])
                chart = get_single_chart(data, "Training Loss")
                st.session_state.b1_train_chart = chart
                with st.spinner(text="In progress... it takes little time..."):
                    st.write(f"Training Anomaly Transformer with {dataset} dataset")
                    my_bar = st.progress(0)
                    for i in range(100):
                        time.sleep(0.05)
                        my_bar.progress(i + 1)
                    time.sleep(1)
                my_bar.empty()
            else:
                st.write("Not implemented")

    with col1:
        if st.session_state.b1_train_chart is not None:
            st.subheader("Training")
            hyper_cont.success("Training finished!")
            st.altair_chart((st.session_state.b1_train_chart).interactive(), use_container_width=True)
            st.metric("Accuracy", st.session_state.b1_accuracy)
            st.metric("Precision", st.session_state.b1_precision)
            st.metric("Recall", st.session_state.b1_recall)
            st.metric("F1 score", st.session_state.b1_f_score)

    ### 2. Test
    with col2:
        if st.session_state.b1_train_chart is not None:
            st.subheader("Test")
            if dataset == "SMD":
                start, end = 15400,17400
                test_energy = st.session_state.b1_test_energy[start:end]
                test_label = st.session_state.b1_test_label[start:end]
                feature_value = st.session_state.b1_feature_value[start:end]
                interpret_v2 = st.session_state.b1_interpret_v2
                threshold = st.session_state.b1_threshold
            else:
                st.write("Not implemented")
            pred = (test_energy > threshold).astype(int)
            anomal_point = np.where(pred == 1)[0]

            # Plot feature and window
            slider = int(st.slider("Detection window", start, end-model_win_size))
            source1 = create_dataframe([feature_value[:end-start, [0,5,10,15,20,25,35]].T], [feature_names[_] for _ in [0,5,10,15,20,25,35]], ['Step', 'Label', 'Value'], start, end)
            line1 = get_single_chart(source1, "Feature")
            cutoff = pd.DataFrame({ 'index': ['blue'], 'start': slider, 'stop': slider + model_win_size})
            areas = alt.Chart(cutoff).mark_rect(opacity=0.2).encode(x='start',x2='stop',y=alt.value(0), color=alt.Color('index:N', scale=None))
            ano_areas = make_areas(test_label, start)
            st.altair_chart(line1 + areas + ano_areas, use_container_width = True)

            # Plot anomaly score in window
            select_anomaly = st.selectbox(
                f"Detected anomaly points \n (Total {len(anomal_point)} anomaly detected)",
                options=["---"] + [f"{__ + start} sec" for _, __ in enumerate(anomal_point)],
                index=0,
            )

            # Choose anomaly
            if select_anomaly == "---":
                source2 = create_dataframe([test_energy[slider - start : slider + model_win_size - start], threshold], ['Anomaly score', 'Threshold'], ['Step', 'Label', "Value"], slider, slider + test_energy[slider - start : slider + model_win_size - start].shape[0])
                line2 = get_single_chart(source2, "Anomaly score")
                gt = st.session_state.b1_test_label[start:end][slider - start : slider + model_win_size - start]
                areas = make_areas(gt, slider)
                st.altair_chart(line2 + areas, use_container_width = True)
            else:
                select_anomaly = select_anomaly.split(" ")[0]
                slider = int(select_anomaly) - model_win_size
                source2 = create_dataframe([test_energy[slider - start : slider + 200 - start], threshold], ['Anomaly score', 'Threshold'], ['Step', 'Label', "Value"], slider, slider + test_energy[slider - start : slider + 200 - start].shape[0])
                line2 = get_single_chart(source2, "Anomaly score")
                rules = alt.Chart(pd.DataFrame({'Step': [int(select_anomaly)]})).mark_rule(color='red').encode(x='Step:Q',)
                gt = st.session_state.b1_test_label[start:end][slider - start : slider + 200 - start]
                areas = make_areas(gt, slider, 'red')
                st.altair_chart(line2 + rules + areas,use_container_width=True)
                if int(select_anomaly) in interpret_v2:
                    st.session_state.b1_anomaly_feature = interpret_v2[int(select_anomaly)]
                else:
                    for _ in range(slider, slider+101):
                        if _ in interpret_v2:
                            st.session_state.b1_aanomaly_feature = interpret_v2[_]
                            break
    # Plot interpretation
    if st.session_state.b1_anomaly_feature is not None and select_anomaly != "---":
        st.caption("* Cause of abnormality")
        col1_, col2_, col3_ = st.columns([1,1,1])
        for idx, _ in enumerate(st.session_state.b1_anomaly_feature):
            with [col1_, col2_, col3_][idx & 3 == 0]:
                source = create_dataframe([feature_value[slider - start : slider + 200 - start, _]], [feature_names[_]], ['Step', 'Label', 'Value'], slider, slider + test_energy[slider - start : slider + 200 - start].shape[0])
                line = get_single_chart(source, f"{feature_names[_]}")
                st.altair_chart(line + rules, use_container_width=True)


    ##################################
    #       ONLINE DETECTING         #
    ##################################
    st.markdown("***")
    st.subheader("Online detecting")
    st.video("./static/online_detecting_32.mp4")

    # Sliding window for detection area
    col1, col2 = st.columns([2,1])
    with col1:
        infer_cont = st.container()
    with col2:
        st.markdown(" ")
        st.markdown(" ")
        st.markdown(" ")
        st.markdown(" ")
        st.markdown(" ")
        st.markdown(" ")
        st.markdown(" ")
        st.markdown(" ")
        st.markdown(" ")
        st.markdown(" ")
        st.markdown(" ")
        st.markdown(" ")
        response_cont = st.expander("Response", expanded=False)
    info_cont = st.container()
    expand_cont = st.container()
    infer_cont.caption("* Simulation")
    st.session_state.simul_expand_feature_list = []

    ### 3. Simulate
    with infer_cont:
        detect_interval = st.number_input("Detecting interval",value=15)
        simul_btn = st.button("Simulate")
        if simul_btn:
            reset(True)

        if simul_btn and st.session_state.simul_online_chart is None:
            response_cont.write("")
            st.session_state.simul_anomaly_score = np.zeros(st.session_state.simul_win_size)
            model_path = "./Anomaly_transformer/demo/SMD_2022_08_12_12_03_28"
            config = pickle.load(open(os.path.join(model_path, "config.p"), "rb"))
            config['model_save_path'] = model_path
            solver = Solver(config)
            ano_areas = make_areas(st.session_state.simul_test_label, 0, "red")
            line2 = create_dataframe([np.array([1 for _ in range(st.session_state.simul_win_size)])], ['None'], ['Step', 'Label', 'Value'], 0, st.session_state.simul_win_size)
            line2 = alt.Chart(line2).mark_line(interpolate='basis').encode(x='Step:Q',opacity=alt.value(0)).properties(width=1200, height=360)
            st.session_state.simul_accuracy, st.session_state.simul_precision, st.session_state.simul_recall, st.session_state.simul_f_score, test_energy, thresh, t = solver.test(config, data_path="./Anomaly_transformer/demo/online", th=st.session_state.b1_threshold, detect_interval=detect_interval)
            st.session_state.simul_time_list = t
            num_ano = 0
            for idx, slider in enumerate(range(model_win_size, st.session_state.simul_win_size, detect_interval)):
                line = create_dataframe([st.session_state.simul_feature_value[:slider+1,[3*_ for _ in range(10)]].T], [feature_names[_] for _ in [3*_ for _ in range(10)]], ['Step', 'Label', 'Value'], 0, slider+1)
                line = get_single_chart(line, "Test")
                cutoff = pd.DataFrame({'index': ['blue'], 'start': slider - model_win_size, 'stop': slider})
                areas = alt.Chart(cutoff).mark_rect(opacity=0.2).encode(
                    x='start',
                    x2='stop',
                    y=alt.value(0),  # pixels from top
                    y2=alt.value(360),  # pixels from top
                    color=alt.Color('index:N', scale=None)
                )

                st.session_state.simul_anomaly_score[slider-model_win_size:slider] = test_energy[idx*model_win_size: (idx+1)*model_win_size]
                source = create_dataframe([st.session_state.simul_anomaly_score, thresh], ['Anomaly score', 'Threshold'], ['Step', 'Label', 'Value'], 0, st.session_state.simul_win_size)
                source = get_single_chart(source, "Anomaly score", 1200, 360)

                # try:
                #     fig2 = fig2.altair_chart(alt.vconcat((line + areas + line2 + ano_areas), (source + areas + ano_areas)), use_container_width=True)
                # except:
                #     fig2 = st.altair_chart(alt.vconcat((line + areas + line2 + ano_areas), (source + areas + ano_areas)), use_container_width=True)

                anomaly = np.where(st.session_state.simul_anomaly_score[slider-model_win_size:slider] > thresh)[0]
                if len(anomaly) > 0:
                    for i in anomaly:
                        st.session_state.simul_anomaly_list[f'Anomaly_{slider-model_win_size+i} s / ( window : {slider-model_win_size} s ~ {slider} s)'] = {
                            "anomaly point": i,
                            "time": str(datetime.now()),
                            "score": test_energy[idx*model_win_size: (idx+1)*model_win_size],
                            "thres": thresh,
                            "feature": st.session_state.simul_feature_value[slider-model_win_size:slider],
                            "p_score": test_energy[idx*model_win_size: (idx+1)*model_win_size][i],
                            "p_feature": st.session_state.simul_feature_value[slider-model_win_size:slider][i],
                            "intensity": ((test_energy[idx*model_win_size: (idx+1)*model_win_size][i] - thresh) / thresh)
                        }
                        response_cont.write(st.session_state.simul_anomaly_list[f'Anomaly_{slider-model_win_size+i} s / ( window : {slider-model_win_size} s ~ {slider} s)'])

                        if slider-model_win_size+i in np.where(st.session_state.simul_test_label == 1)[0]:
                            st.session_state.simul_true_positive += 1
                        else:
                            st.session_state.simul_false_positive += 1
            fig2 = st.altair_chart(alt.vconcat((line + areas + line2 + ano_areas), (source + areas + ano_areas)), use_container_width=True)
            st.session_state.simul_online_chart = alt.vconcat((line + areas + line2 + ano_areas), (source + areas + ano_areas))

        elif st.session_state.simul_online_chart is not None:
            response_cont.write(st.session_state.simul_anomaly_list)
            fig2 = st.altair_chart(st.session_state.simul_online_chart,use_container_width=True)

    ### 4. Simulation result
    with info_cont:
        if st.session_state.simul_online_chart is not None:
            st.caption("* Result")
            unique_anomaly = [_ for _ in st.session_state.simul_anomaly_list.keys()]
            _1, _2, _3, _4 = st.columns([1,1,1,1])
            _1.metric("Average inference time(CPU)", np.mean(st.session_state.simul_time_list).round(4))
            _2.metric("Number of alert", len(st.session_state.simul_anomaly_list))
            _3.metric("Accuracy", st.session_state.simul_accuracy.round(4))
            _3.metric("Precision", st.session_state.simul_precision.round(4))
            _4.metric("Recall", st.session_state.simul_recall.round(4))
            _4.metric("F1 score", st.session_state.simul_f_score.round(4))

            st.caption("* Result(700,000 time points)")
            _1, _2, _3, _4 = st.columns([1,1,1,1])
            _3.metric("Accuracy", 0.9918)
            _3.metric("Precision", 0.9335)
            _4.metric("Recall", 0.9048)
            _4.metric("F1 score", 0.9189)

            key = st.selectbox("Anomaly", options=list(st.session_state.simul_anomaly_list.keys()), index=0)
            thresh = st.session_state.simul_anomaly_list[key]['thres']
            score = st.session_state.simul_anomaly_list[key]['score']
            score = create_dataframe([score, thresh], ['Anomaly score', 'Threshold'], ['Step', 'Label', 'Value'], 0, model_win_size)
            score = get_single_chart(score, "Anomaly score")
            rules = alt.Chart(pd.DataFrame({"Step": [st.session_state.simul_anomaly_list[key]['anomaly point']]})).mark_rule(color='red').encode( x='Step:Q',)
            st.altair_chart(score + rules, use_container_width=True)

            col1, col2, col3 = st.columns([1,1,1])
            col4, col5, _ = st.columns([1,1,1])
            feature_value = st.session_state.simul_anomaly_list[key]['feature']
            with col1:
                draw_online_feature("CPU", "", [0,1,2,3], feature_value, feature_names, rules, model_win_size)
                if st.checkbox( "Show detail", key=1):
                    st.session_state.simul_expand_feature_list += [0,1,2,3]
            with col2:
                draw_online_feature("DISK", "", [8,9,10,11,12,13,14,15], feature_value, feature_names, rules, model_win_size)
                if st.checkbox( "Show detail", key=2):
                    st.session_state.simul_expand_feature_list += [8,9,10,11,12,13,14,15]
            with col3:
                draw_online_feature("MEMORY", "", [4,5,6,7], feature_value, feature_names, rules, model_win_size)
                if st.checkbox( "Show detail", key=3):
                    st.session_state.simul_expand_feature_list += [4,5,6,7,]
            with col4:
                draw_online_feature("TCP/UDP/ETH", "", [18,19,20,21,22,23,33,34,35,36,37], feature_value, feature_names, rules, model_win_size)
                if st.checkbox( "Show detail", key=4):
                    st.session_state.simul_expand_feature_list += [18,19,20,21,22,23,33,34,35,36,37]
            with col5:
                draw_online_feature("etc", "", [24,25,26,27,28,29,30,31,32], feature_value, feature_names, rules, model_win_size)
                if st.checkbox( "Show detail", key=5):
                    st.session_state.simul_expand_feature_list += [24,25,26,27,28,29,30,31,32]

    ### 5. Expand
    with expand_cont:
        if st.session_state.simul_online_chart is not None:
            st.markdown("---")
            col1, col2, col3 = st.columns([1,1,1])
            for idx, i in enumerate(st.session_state.simul_expand_feature_list):
                with [col1, col2, col3][idx % 3]:
                    source = create_dataframe([feature_value[:, i]], [feature_names[i]], ['Step', 'Label', 'Value'], 0, model_win_size)
                    line = get_single_chart(source, feature_names[i])
                    st.altair_chart(line + rules,use_container_width=True)
