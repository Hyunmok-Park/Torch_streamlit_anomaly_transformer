import altair as alt
import pandas as pd
import numpy as np

def create_dataframe(source_list, label_list, column_list, start=None, end=None):

    preprocessed = []

    for s, l in zip(source_list, label_list):
        if l == "Threshold":
            preprocessed.append(np.array([s for _ in range(start, end)]).reshape(-1,1))
        else:
            preprocessed.append(s.reshape(-1,1))

    preprocessed = np.array(preprocessed)

    step = np.array([_ for _ in range(start, end)] * len(label_list)).reshape(-1,1)
    label = np.array([[_ for i in range(start, end)] for _ in label_list])
    label = np.concatenate(label, axis=0).reshape(-1,1)
    value = np.concatenate(preprocessed, axis=0).reshape(-1,1)

    data = np.concatenate([step, label, value], axis=-1)

    df = pd.DataFrame(
        data,
        columns=column_list
    )

    return df

def get_single_chart(source, title, width=None, height=None):
    # Create a selection that chooses the nearest point & selects based on x-value

    nearest = alt.selection(type='single', nearest=True, on='mouseover',
                            fields=['Step'], empty='none')
    # The basic line
    if width != None:
        line = alt.Chart(source, title=title).mark_line(interpolate='basis').encode(
            x='Step:Q',
            y=alt.Y('Value:Q', scale=alt.Scale(domain=(0,1))),
            color='Label:N'
        ).properties(width=width, height=height)
    else:
        line = alt.Chart(source, title=title).mark_line(interpolate='basis').encode(
            x='Step:Q',
            y='Value:Q',
            color='Label:N'
        )

    # Transparent selectors across the chart. This is what tells us
    # the x-value of the cursor
    selectors = alt.Chart(source).mark_point().encode(
        x='Step:Q',
        opacity=alt.value(0),
    ).add_selection(
        nearest
    )

    # Draw points on the line, and highlight based on selection
    points = line.mark_point().encode(
        opacity=alt.condition(nearest, alt.value(1), alt.value(0))
    )

    # Draw text labels near the points, and highlight based on selection
    text = line.mark_text(align='left', dx=5, dy=-5).encode(
        text=alt.condition(nearest, 'Value:Q', alt.value(' '))
    )

    # Draw a rule at the location of the selection
    rules = alt.Chart(source).mark_rule(color='gray').encode(
        x='Step:Q',
    ).transform_filter(
        nearest
    )

    return (line + selectors + points + text + rules).interactive()

def get_double_chart(source1, source2, source3, title1, title2, feature_names, feature_list, gt, inc_num=0, width=None, height=None):

    nearest = alt.selection(type='single', nearest=True, on='mouseover',
                            fields=['Step'], empty='none')

    ############
    #   Upper  #
    ############
    # Upper basic line
    line = alt.Chart(source1, title=title1)\
        .mark_line(interpolate='basis')\
        .encode(x='Step:Q',y='Anomaly score:Q',color='Label:N')\
        .properties(width=width, height=height)
    line_ = alt.Chart(source2).mark_line(interpolate='basis').encode(x='Step:Q',y='Anomaly score:Q',color='Label:N',opacity=alt.value(0))

    # Upper selection
    selector = alt.Chart(source1).mark_point().encode(
        x='Step:Q',
        opacity=alt.value(0),
    ).add_selection(
        nearest
    )

    text = line.mark_text(align='left', dx=5, dy=-5).encode(
        text=
        alt.condition(
            nearest,
            'Anomaly score:Q',
            alt.value(' '))
    )

    points = line.mark_point().encode(
        opacity=alt.condition(nearest, alt.value(1), alt.value(0))
    )

    source3 = source3.replace({'Prediction': 0}, {'Prediction': 'Normal'})
    source3 = source3.replace({'Prediction': 1}, {'Prediction': 'Anomaly'})

    tp = [
             alt.Tooltip("Step", title="step"),
             alt.Tooltip("Anomaly score", title="anomaly score"),
             alt.Tooltip("Prediction", title="anomaly"),
             alt.Tooltip("Threshold", title="threshold")] + [alt.Tooltip(f"{_}", title=f"{_}") for _ in list(np.array(feature_names)[feature_list])]

    tooltips = alt.Chart(source3).mark_rule(color='grey').encode(
        x="Step:Q",
        opacity=alt.condition(nearest, alt.value(1), alt.value(0)),
        tooltip=tp
    )

    ############
    #   Lower  #
    ############

    source1 = source1.rename(columns={"Anomaly score":"Value"})
    source2 = source2.rename(columns={"Anomaly score":"Value"})

    line2 = alt.Chart(source2, title=title2).mark_line(interpolate='basis').encode(x='Step:Q',y='Value:Q',color='Label:N').properties(width=width, height=height)
    line2_ = alt.Chart(source1).mark_line(interpolate='basis').encode(x='Step:Q',y='Value:Q',color='Label:N',opacity=alt.value(0))

    selector2 = alt.Chart(source2).mark_point().encode(
        x='Step:Q',
        opacity=alt.value(0),
    ).add_selection(
        nearest
    )

    text2 = line2.mark_text(align='left', dx=5, dy=-5).encode(
        text=
        alt.condition(
            nearest,
            'Step:Q',
            alt.value(' '))
    )

    points2 = line2.mark_point().encode(
        opacity=alt.condition(nearest, alt.value(1), alt.value(0))
    )

    rules2 = alt.Chart(source2).mark_rule(color='gray').encode(
        x='Step:Q',
    ).transform_filter(
        nearest
    )

    tooltips2 = alt.Chart(source3).mark_rule(color='grey').encode(
        x="Step:Q",
        opacity=alt.condition(nearest, alt.value(1), alt.value(0)),
        tooltip=tp
    )

    start = []
    end = []
    for idx,i in enumerate(np.where(gt == 1)[0]):
        if idx == 0:
            current_num = i
            start.append(i+inc_num)
        else:
            if i - current_num == 1:
                current_num = i
                if (idx+1) == len(np.where(gt == 1)[0]):
                    end.append(i+inc_num)
                continue
            else:
                end.append(current_num+inc_num)
                start.append(i+inc_num)
                current_num = i

    cutoff = pd.DataFrame({
        'index': ['red' for _ in range(len(start))],
        'start': start,
        'stop': end
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

    return alt.vconcat(
        (line + line_ + selector + points + tooltips + areas).interactive(),
        (line2 + line2_ + selector2 + points2 + tooltips2 + areas).interactive()
    )


