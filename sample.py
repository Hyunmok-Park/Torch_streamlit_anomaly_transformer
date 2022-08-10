from utils import *
from Anomaly_transformer.model.AnomalyTransformer import *
from Anomaly_transformer.main import *
from Anomaly_transformer.solver import *

import streamlit as st
from streamlit_option_menu import option_menu

import csv
import numpy as np
from pathlib import Path
import altair as alt
import pandas as pd
import configparser
import pickle
import time

from pyecharts.charts import Bar
from pyecharts.charts import Line
from pyecharts import options as opts

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

    if "SMD" not in st.session_state:
        st.session_state.SMD = []
        st.session_state.anomaly_feature = None

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
        st.subheader("Inference")

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
            slider = int(st.slider("Detection window", start, end-100))
            source1 = create_dataframe([feature_value[:1000, [0,5,10,15,20,25,35]].T], [feature_names[_] for _ in [0,5,10,15,20,25,35]], ['Step', 'Label', 'Value'], start, end)
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
                f"Detected anomaly points \n (Total {len(anomal_point)} anomaly detected)",
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
                if int(select_anomaly) in interpret:
                    st.session_state.anomaly_feature = interpret[int(select_anomaly)]
                else:
                    for _ in range(slider, slider+101):
                        if _ in interpret:
                            st.session_state.anomaly_feature = interpret[_]
                            break

    if st.session_state.anomaly_feature is not None:
        st.caption("* Cause of abnormality")
        col1_, col2_, col3_ = st.columns([1,1,1])
        for idx, _ in enumerate(st.session_state.anomaly_feature):
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
    # else:
    #     st.write("There is no anomaly feature")


    ##################################
    #       ONLINE DETECTING         #
    ##################################
    st.markdown("***")
    st.subheader("Online detecting")
    if "simul_test_energy" not in st.session_state:
        st.session_state.simul_test_energy = np.load("./Anomaly_transformer/demo/test_energy.npy")
        st.session_state.simul_train_energy = np.load("./Anomaly_transformer/demo/SMD_train_energy.npy")
        st.session_state.simul_threshold = np.percentile(np.concatenate([st.session_state.simul_train_energy, st.session_state.simul_test_energy], axis=0), 100-0.5)
        st.session_state.simul_test_energy = np.load("./Anomaly_transformer/demo/test_energy.npy")[18000:19000]
        st.session_state.simul_feature_value = np.load("./Anomaly_transformer/demo/SMD_test.npy")[18000:19000]

        st.session_state.simul_online_chart = None
        st.session_state.simul_anomaly_list = {}
        st.session_state.simul_time_list = []

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
        infer_cont_ = st.expander("Response")
    info_cont = st.container()
    expand_cont = st.container()
    infer_cont.caption("* Simulation")

    st.session_state.simul_expand_feature_list = []

    with infer_cont:
        if st.button("Simulate") and st.session_state.simul_online_chart is None:
            st.session_state.simul_anomaly_score = np.zeros(1000)
            for i in range(100,1000,20):
                slider = i
                line = create_dataframe([st.session_state.simul_feature_value[:slider+1,[0,5,10,15]].T], [feature_names[_] for _ in [0,5,10,15]], ['Step', 'Label', 'Value'], 0, slider+1)
                line = get_single_chart(line, "Test")
                line2 = create_dataframe([np.array([1 for _ in range(1000)])], ['None'], ['Step', 'Label', 'Value'], 0, 1000)
                line2 = alt.Chart(line2).mark_line(interpolate='basis').encode(
                    x='Step:Q',
                    opacity=alt.value(0)
                ).properties(width=1200, height=300)
                cutoff = pd.DataFrame({
                    'index': ['blue'],
                    'start': slider - 100,
                    'stop': slider
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

                # Online anomlay score
                # try:
                #     mkdir(f"./Anomaly_transformer/dataset/test/online/{slider}")
                # except:
                #     pass
                # np.save(f"./Anomaly_transformer/dataset/test/online/{slider}/SMD_test.npy", st.session_state.simul_feature_value[slider-100:slider])

                model_path = "SMD_2022_08_08_15_38_41"
                config = make_config(model_path, f"./Anomaly_transformer/dataset/test/online/{slider}", 0.5, 10, 3, 512, 38, 1, 100, 512, 3, 5, 0.001)
                solver = Solver(config)
                tik = time.time()
                accuracy, precision, recall, f_score, test_energy, thresh = solver.test()
                tok = time.time()
                st.session_state.simul_time_list.append(tok - tik)
                st.session_state.simul_anomaly_score[slider-100:slider] = test_energy

                source = create_dataframe([st.session_state.simul_anomaly_score, thresh], ['Anomaly score', 'Threshold'], ['Step', 'Label', 'Value'], 0, 1000)
                source = get_single_chart(source, "Anomaly score", 1200, 300)

                try:
                    fig2 = fig2.altair_chart(
                        alt.vconcat((line + areas + line2), (source + areas)),
                        use_container_width=True
                    )
                except:
                    fig2 = st.altair_chart(
                        alt.vconcat((line + areas + line2), (source + areas)),
                        use_container_width=True
                    )
                import copy
                anomaly = np.where(st.session_state.simul_anomaly_score[slider-100:slider] > thresh)[0]
                if len(anomaly) > 0:
                    for i in anomaly:
                        st.session_state.simul_anomaly_list[f'Anomaly_{slider-100+i} s / ( window : {slider-100} s ~ {slider} s)'] = {
                            "anomaly point": i,
                            "time": str(datetime.now()),
                            "score": test_energy,
                            "thres": thresh,
                            "feature": st.session_state.simul_feature_value[slider-100:slider],
                            "p_score": test_energy[i],
                            "p_feature": st.session_state.simul_feature_value[slider-100:slider][i],
                            "intensity": ((test_energy[i] - thresh) / thresh)
                        }

                        infer_cont_.write(st.session_state.simul_anomaly_list[f'Anomaly_{slider-100+i} s / ( window : {slider-100} s ~ {slider} s)'])

            st.session_state.simul_online_chart = alt.vconcat((line + areas + line2), (source + areas))

        elif st.session_state.simul_online_chart is not None:
            infer_cont_.write(st.session_state.simul_anomaly_list)
            fig2 = st.altair_chart(
                st.session_state.simul_online_chart,
                use_container_width=True
            )

    with info_cont:
        if st.session_state.simul_online_chart is not None:
            st.caption("* Result")
            metric = pd.DataFrame({
                "Average inference time(CPU)": [np.mean(st.session_state.simul_time_list)],
                "Total detected anomaly": [len(st.session_state.simul_anomaly_list)]
            })
            st.write(metric)

            k = st.selectbox(
                "Anomaly",
                options=list(st.session_state.simul_anomaly_list.keys()),
                index=0
            )

            thresh = st.session_state.simul_anomaly_list[k]['thres']
            score = st.session_state.simul_anomaly_list[k]['score']
            score = create_dataframe([score, thresh], ['Anomaly score', 'Threshold'], ['Step', 'Label', 'Value'], 0, 100)
            score = get_single_chart(score, "Anomaly score")
            rules = alt.Chart(pd.DataFrame({"Step": [st.session_state.simul_anomaly_list[k]['anomaly point']]})).mark_rule(color='red').encode(
                x='Step:Q',
            )

            st.altair_chart(
                score + rules,
                use_container_width=True
            )

            col1, col2, col3, col4, col5 = st.columns([1,1,1,1,1])
            feature_value = st.session_state.simul_anomaly_list[k]['feature']

            with col1:
                st.subheader("cpu") #0,1,2,3,
                feature_list = [0,1,2,3]
                source = create_dataframe([feature_value[:,feature_list].T], [feature_names[_] for _ in feature_list], ['Step', 'Label', 'Value'], 0,100)
                line = get_single_chart(source, "cpu_r")
                st.altair_chart(
                    line + rules,
                    use_container_width=True
                )
                if st.checkbox(
                    "Show detail",
                    key=1
                ):
                    st.session_state.simul_expand_feature_list += feature_list

            with col2:
                st.subheader("disk") #8,9,10,11,12,13,14,15
                feature_list = [8,9,10,11,12,13,14,15]
                source = create_dataframe([feature_value[:,feature_list].T], [feature_names[_] for _ in feature_list], ['Step', 'Label', 'Value'], 0,100)
                line = get_single_chart(source, "disk_rb")
                st.altair_chart(
                    line + rules,
                    use_container_width=True
                )
                if st.checkbox(
                        "Show detail",
                        key=2
                ):
                    st.session_state.simul_expand_feature_list += feature_list

            with col3:
                st.subheader("memory/eth") #4,5,6,7,18,19,20,21
                feature_list = [4,5,6,7,18,19,20,21]
                source = create_dataframe([feature_value[:,feature_list].T], [feature_names[_] for _ in feature_list], ['Step', 'Label', 'Value'], 0,100)
                line = get_single_chart(source, "disk_wb")
                st.altair_chart(
                    line + rules,
                    use_container_width=True
                )
                if st.checkbox(
                        "Show detail",
                        key=3
                ):
                    st.session_state.simul_expand_feature_list += feature_list
            with col4:
                st.subheader("TCP/UDP") # 22,23,33,34,35,36,37
                feature_list = [22,23,33,34,35,36,37]
                source = create_dataframe([feature_value[:,feature_list].T], [feature_names[_] for _ in feature_list], ['Step', 'Label', 'Value'], 0,100)
                line = get_single_chart(source, "mem_u")
                st.altair_chart(
                    line + rules,
                    use_container_width=True
                )
                if st.checkbox(
                        "Show detail",
                        key=4
                ):
                    st.session_state.simul_expand_feature_list += feature_list

            with col5:
                st.subheader("etc") # 24,25,26,27,28,29,30,31,32
                feature_list = [24,25,26,27,28,29,30,31,32]
                source = create_dataframe([feature_value[:,feature_list].T], [feature_names[_] for _ in feature_list], ['Step', 'Label', 'Value'], 0,100)
                line = get_single_chart(source, "mem_u")
                st.altair_chart(
                    line + rules,
                    use_container_width=True
                )
                if st.checkbox(
                        "Show detail",
                        key=5
                ):
                    st.session_state.simul_expand_feature_list += feature_list

    with expand_cont:
        if st.session_state.simul_online_chart is not None:
            st.markdown("---")
            col1, col2, col3 = st.columns([1,1,1])
            for idx, i in enumerate(st.session_state.simul_expand_feature_list):
                if idx % 3 == 0:
                    with col1:
                        source = create_dataframe([feature_value[:, i]], [feature_names[i]], ['Step', 'Label', 'Value'], 0, 100)
                        line = get_single_chart(source, feature_names[i])
                        st.altair_chart(
                            line + rules,
                            use_container_width=True
                        )
                elif idx % 3 == 1:
                    with col2:
                        source = create_dataframe([feature_value[:, i]], [feature_names[i]], ['Step', 'Label', 'Value'], 0, 100)
                        line = get_single_chart(source, feature_names[i])
                        st.altair_chart(
                            line + rules,
                            use_container_width=True
                        )
                elif idx % 3 == 2:
                    with col3:
                        source = create_dataframe([feature_value[:, i]], [feature_names[i]], ['Step', 'Label', 'Value'], 0, 100)
                        line = get_single_chart(source, feature_names[i])
                        st.altair_chart(
                            line + rules,
                            use_container_width=True
                        )
