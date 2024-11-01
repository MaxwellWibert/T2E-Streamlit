# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 13:58:05 2024

@author: VHAWASShaoY
"""

from dataclasses import dataclass


import torch
from torch import nn
import numpy as np


import matplotlib.pyplot as plt
from matplotlib import ticker
import plotly.graph_objects as go

import streamlit as st


dimIn, dimOut = 29, 77


race_map = {"White": 0, "Black": 1, "Other": 2, "Unkown": 3}
eth_map = {"Non-Hispanic": 0, "Hispanic": 1, "Other": 2, "Unkown": 3}


def initialize_state():
    if "profile" not in st.session_state:
        st.session_state.profile = None
        st.session_state.alt_profile = None


@dataclass
class RiskProfile:
    age: int
    sex: int
    race: int
    ethnicity: int

    # fitness
    fitness: float
    bmi: float

    # comorbidities
    afib: bool = False
    anemia: bool = False
    arthritis: bool = False
    asthma: bool = False
    cancer: bool = False
    ckd: bool = False
    copd: bool = False
    depression: bool = False
    diabetes: bool = False
    heart_failure: bool = False
    hypertension: bool = False
    hyperlipidemia: bool = False
    ischemic_heart_disease: bool = False
    stroke: bool = False
    tbi: bool = False
    statin: bool = False
    dmrx: bool = False
    cvdrx: bool = False
    asthma_copd: bool = False
    cvd: bool = False

    @classmethod
    def from_form(
        cls,
        *comorbidities,
        age: int,
        sex: str,
        race: str,
        ethnicity: str,
        fitness: float,
        bmi: float,
        cvdrx: bool,
        dmrx: bool,
    ):
        sex_int = 0 if sex == "M" else 1
        race_int = race_map[race]
        eth_int = eth_map[ethnicity]
        comorbid_dict = {opt: True for opt in comorbidities}

        return cls(
            age=age,
            sex=sex_int,
            race=race_int,
            ethnicity=eth_int,
            fitness=fitness,
            bmi=bmi,
            **comorbid_dict,
        )

    def to_vec(self):
        X = torch.zeros(dimIn)
        num_mean = np.array([60.687679, 29.320632, 7.543494])
        num_std = np.array([10.996245, 5.382613, 3.141214])
        # Age, BMI, Fitness
        age_bmi_fit = np.array([self.age, self.bmi, self.fitness])
        X[:3] = torch.from_numpy((age_bmi_fit - num_mean) / num_std)
        # Sex M = 0, F = 1
        X[3] = float(self.sex)
        # Race white=0, black=1, other=2, unkown=3
        X[4] = float(self.race == 1)
        X[5] = float(self.race == 2)
        X[6] = float(self.race == 3)
        # Ethnicity: Non-hisp=0, Hisp =1, Unkown =2
        X[7] = float(self.ethnicity == 1)
        X[8] = float(self.ethnicity == 2)

        # Comorbidities
        comorbid_vec = np.array(
            [
                self.afib,
                self.anemia,
                self.arthritis,
                self.asthma,
                self.cancer,
                self.ckd,
                self.copd,
                self.depression,
                self.diabetes,
                self.heart_failure,
                self.hypertension,
                self.hyperlipidemia,
                self.ischemic_heart_disease,
                self.stroke,
                self.tbi,
                self.statin,
                self.dmrx,
                self.cvdrx,
                self.asthma_copd,
                self.cvd,
            ]
        ).astype(float)
        X[9:] = torch.from_numpy(comorbid_vec)

        return X


def compute_bmi(height: float, weight: float):
    return 703 * weight / (height**2)


class ResFFNN(nn.Module):
    def __init__(self, d_in, d_hid, d_out):
        super(ResFFNN, self).__init__()
        self.fc1 = nn.Linear(d_in, d_out)
        self.dropout1 = nn.Dropout(p=0.1)
        self.norm1 = nn.LayerNorm(d_out)
        self.fc2 = nn.Linear(d_out, d_hid)
        self.dropout2 = nn.Dropout(p=0.1)
        self.fc3 = nn.Linear(d_hid, d_out)
        self.dropout3 = nn.Dropout(p=0.1)
        self.norm2 = nn.LayerNorm(d_out)

    def forward(self, input):
        output = self.norm1(self.dropout1(self.fc1(input)))
        output2 = self.dropout2(nn.functional.relu(self.fc2(output)))
        output = output + self.dropout3(self.fc3(output2))
        output = self.norm2(output)
        return output


class Model(nn.Module):
    def __init__(self, in_dim):
        super(Model, self).__init__()
        self.ffnn = ResFFNN(in_dim, 32, 32)
        self.ffnn1 = ResFFNN(32, 32, 32)
        self.ffnn2 = ResFFNN(32, 32, 32)
        self.linear1 = nn.Linear(32, dimOut)
        self.linear2 = nn.Linear(32, dimOut)

    def forward(self, input):
        output = self.ffnn(input)
        # output = self.ffnn0(output)
        output1 = self.linear1(self.ffnn1(output))
        output2 = self.linear2(self.ffnn2(output))
        output = torch.cat([output1, output2], dim=-1)
        return output


@st.cache_data
def load_model(pt_file):
    model = Model(in_dim=dimIn)
    model.load_state_dict(torch.load(pt_file))
    return model


def load_data():
    data = np.zeros(26)
    """
load data from user input
26 variables:
    Age
    Sex
    Race
    Ethnicity
    Fitness (METs)
    BMI
    AFib
    Anemia
    Arthritis
    Asthma
    Cancer
    CKD
    COPD
    Depression
    Diabetes
    Heart Failure
    Hypertension
    Hyperlipidemia
    Ischemic Heart Disease
    Stroke
    TBI
    Statin
    DmRx
    CVDRx
    Asthma_COPD
    CVD
    """

    X = torch.zeros(dimIn)
    num_mean = np.array([60.687679, 29.320632, 7.543494])
    num_std = np.array([10.996245, 5.382613, 3.141214])
    # Age, BMI, Fitness
    X[:3] = (data[[0, 4, 3]] - num_mean) / num_std
    # Sex: M=0, F=1
    X[3] = float(data[1] == 1)  # Sex=='F'
    # Race: White=0, Black=1, Other=2, Unknown=3
    X[4] = float(data[2] == 1)  # Race=='Black'
    X[5] = float(data[2] == 2)  # Race=='Other'
    X[6] = float(data[2] == 3)  # Race=='Unknown'
    # Ethnicity: Non-Hisp=0, Hisp=1, Unknown=2
    X[7] = float(data[3] == 1)  # Ethnicity=='Hisp'
    X[8] = float(data[3] == 2)  # Ethnicity=='Unknown'
    X[9:] = data[6:]
    """
    X is a vector of dim = dimIn = 29
    """
    return X


def run_model(X):
    model.eval()
    with torch.no_grad():
        output = model(X.view(1, -1))
    pmfs = output.softmax(dim=1).view(-1, 2, dimOut)[0]
    cifs = np.zeros((2, dimOut + 4))
    cifs[:, 5:] = pmfs.cumsum(1).numpy()[:, :-1]
    return cifs


# def plot_result(cifs):
#     plt.stackplot(
#         np.arange(dimOut + 4) / 4.0, 100 * cifs, labels=["ADRD", "Death w/o ADRD"]
#     )
#     ax = plt.gca()
#     yticks = ax.get_yticks()
#     ndec = max(0, -np.floor(np.log10(yticks[1])))
#     ax.yaxis.set_major_formatter(ticker.PercentFormatter(decimals=ndec))
#     plt.legend(loc="upper left")
#     plt.xlabel("Elapsed Time (Years)")
#     plt.ylabel("Cumulative Incidence")
#     plt.title("Cumulative Incidence Functions")
#     return plt.gcf()


def hover_texts(y, baseline_y=None):
    texts = [f"{val:+.2%}" for val in y]

    if baseline_y is not None:
        n = len(y)
        delta_y = y - baseline_y
        perc_delta_y = delta_y / baseline_y
        prepositions = ["under" if delta < 0 else "over" for delta in delta_y]
        texts = [
            f"{texts[i]} ({perc_delta_y[i]:+.2%} {prepositions[i]} baseline)"
            for i in range(n)
        ]

    return texts


# Plotly version
def create_plot(cifs, cifs_alt):
    x = np.arange(dimOut + 4) / 4.0
    fig = go.Figure(
        layout=dict(
            title=dict(
                text="Cumulative ADRD Risk and Mortality Over Time",
                xanchor="left",
                yanchor="top",
            ),
            legend=dict(entrywidth=0),
        )
    )

    y_adrd = cifs[0]
    y_death = cifs[1]

    stacked_y_adrd = cifs[0]
    stacked_y_death = cifs[0] + cifs[1]

    if cifs_alt is not None:
        y_adrd_alt = cifs_alt[0]
        y_death_alt = cifs_alt[1]
        stacked_y_adrd_alt = cifs_alt[0]
        stacked_y_death_alt = cifs_alt[0] + cifs_alt[1]

        fig.add_trace(
            go.Scatter(
                x=x,
                y=stacked_y_adrd_alt,
                fill="tozeroy",
                name="CF ADRD Risk",
                hovertext=hover_texts(y=y_adrd_alt, baseline_y=y_adrd),
                hovertemplate="%{hovertext}",
                line=dict(color="darkviolet", dash="dash"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=stacked_y_adrd,
                fill="tonexty",
                name="ADRD Risk",
                hovertext=hover_texts(y=y_adrd),
                hovertemplate="%{hovertext}",
                line=dict(color="blueviolet"),
            )
        )
    else:
        fig.add_trace(
            go.Scatter(
                x=x,
                y=stacked_y_adrd,
                fill="tozeroy",
                name="ADRD Risk",
                hovertext=hover_texts(y=y_adrd),
                hovertemplate="%{hovertext}",
                line=dict(color="blueviolet"),
            )
        )

    if cifs_alt is not None:
        fig.add_trace(
            go.Scatter(
                x=x,
                y=stacked_y_death_alt,
                fill="tonexty",
                name="CF Mortality (w/o ADRD)",
                hovertext=hover_texts(y=y_death_alt, baseline_y=y_death),
                hovertemplate="%{hovertext}",
                line=dict(color="red", dash="dash"),
            ),
        )

    fig.add_trace(
        go.Scatter(
            x=x,
            y=stacked_y_death,
            fill="tonexty",
            name="Mortality (w/o ADRD)",
            hovertext=hover_texts(y=y_death),
            hovertemplate="%{hovertext}",
            line=dict(color="red"),
        )
    )
    fig.update_layout(
        hovermode="x",
        xaxis=dict(ticksuffix=" years", title="Time Elapsed from Present (Years)"),
        yaxis=dict(tickformat=",.0%", title="Cumulative Risk"),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    )

    return fig


def plotly_result(df):
    """
    df should have columns t(time) adrd, adrd_alt, death, death_alt
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.X, y=df.death_alt, fill="tozeroy"))
    fig.add_trace(go.Scatter(x=df.X, y=df.death, fill="tonexty"))
    fig.add_trace(go.Scatter(x=df.X, y=df.adrd_alt, fill="tonexty"))
    fig.add_trace(go.Scatter(x=df.X, y=df.adrd, fill="tonexty"))


def show_vars():
    if "profile" in st.session_state and st.session_state.profile is not None:
        debug_show(st.session_state.profile, "profile", wrap_lines=True)
        X = st.session_state.profile.to_vec()
        debug_show(X, "input")
        y = run_model(X)
        debug_show(torch.from_numpy(y), "output")
        fig = create_plot(y)
        st.pyplot(fig)


def display_opt(raw_opt: str):
    return raw_opt.replace("_", " ").title()


def on_submit(*args, **kwargs):
    profile = RiskProfile.from_form(*args, **kwargs)
    st.session_state.profile = profile


@st.fragment
def render_form():
    # show_vars()
    wrap = st.container()
    with wrap.form(key="risk_factor"):
        st.markdown("## Risk Factors")
        # Demographics
        st.markdown("### Demographics")
        d1, d2, d3, d4 = st.columns(4)

        age = d1.number_input("Age", 30, 95, value=61)
        sex = d2.radio("Sex", ["M", "F"])
        race = d3.radio("Race", ["White", "Black", "Other", "Unkown"])
        eth = d4.radio("Ethnicity", ["Non-Hispanic", "Hispanic", "Unkown"])

        # Comorbidities
        st.markdown("### Comorbidities")
        # c1, c2, c3, c4
        option_names = [
            "afib",
            "anemia",
            "arthritis",
            "asthma",
            "cancer",
            "ckd",
            "copd",
            "depression",
            "diabetes",
            "heart_failure",
            "hypertension",
            "hyperlipidemia",
            "ischemic_heart_disease",
            "stroke",
            "tbi",
            "statin",
            "asthma copd",
            "cvd",
        ]

        comorbidities = st.multiselect(
            "comorbidities",
            options=option_names,
            format_func=display_opt,
            label_visibility="hidden",
        )

        st.markdown("### Medication")

        med_1, med_2 = st.columns(2)
        dmrx = med_1.checkbox("Diabetes Mellitus RX")
        cvdrx = med_2.checkbox("Cardivascular Disease RX")

        # TODO Link to explainers for BMI and Fitness

        # Fitness & BMI scores
        st.markdown("### Fitness & Weight")
        # Default val = 7.0 (median from stats)
        fw1, fw2 = st.columns(2)
        fit = fw1.number_input(
            "Fitness (METs)",
            min_value=2.0,
            max_value=24.0,
            step=0.25,
            value=7.0,
        )
        # Default val = 28.7 (median from stats)
        bmi_exp = fw2.number_input(
            "Body Mass Index",
            min_value=18.5,
            max_value=74.5,
            step=0.25,
            value=29.0,
        )

        bmi_toggle = fw2.toggle("Use Weight & Height to Compute BMI")
        weight = fw2.number_input(
            "Weight (Pounds)",
            min_value=100.0,
            max_value=400.0,
            step=1.0,
            value=200.0,
        )
        height = fw2.number_input(
            "Height (Inches)",
            min_value=48.0,
            max_value=84.0,
            step=1.0,
            value=70.0,
        )

        with st.expander(
            label="Counterfactual", expanded=False, icon=":material/balance:"
        ):
            include_counter = st.toggle("Include Counterfactual (CF)")
            counter_1, counter_2 = st.columns(2)
            counter_fit = counter_1.number_input(
                "CF Fitness (METs)",
                min_value=2.0,
                max_value=24.0,
                step=0.25,
                value=7.0,
            )
            # Default val = 28.7 (median from stats)

            counter_bmi_exp = counter_2.number_input(
                "CF Body Mass Index",
                min_value=18.5,
                max_value=74.5,
                step=0.25,
                value=29.0,
            )

            counter_bmi_toggle = counter_2.toggle(
                "Use Weight and Height to Compute BMI"
            )

            counter_weight = counter_2.number_input(
                "CF Weight (Pounds)",
                min_value=100.0,
                max_value=400.0,
                step=1.0,
                value=200.0,
            )
            counter_height = counter_2.number_input(
                "CF Height (Inches)",
                min_value=48.0,
                max_value=84.0,
                step=1.0,
                value=73.0,
            )

        submit = st.form_submit_button("Compute Risk Scores", type="primary")

    if submit:
        if bmi_toggle:
            bmi = compute_bmi(weight=weight, height=height)
        else:
            bmi = bmi_exp

        profile = RiskProfile.from_form(
            *tuple(comorbidities),
            age=age,
            sex=sex,
            race=race,
            ethnicity=eth,
            fitness=fit,
            bmi=bmi,
            cvdrx=cvdrx,
            dmrx=dmrx,
        )
        st.session_state.profile = profile

        if include_counter:
            if counter_bmi_toggle:
                counter_bmi = compute_bmi(weight=counter_weight, height=counter_height)
            else:
                counter_bmi = counter_bmi_exp

            alt_profile = RiskProfile.from_form(
                *tuple(comorbidities),
                age=age,
                sex=sex,
                race=race,
                ethnicity=eth,
                fitness=counter_fit,
                bmi=counter_bmi,
                cvdrx=cvdrx,
                dmrx=dmrx,
            )
            st.session_state.alt_profile = alt_profile
        else:
            st.session_state.alt_profile = None
        render_plot()


def render_plot():
    if "profile" in st.session_state and st.session_state.profile is not None:
        X = st.session_state.profile.to_vec()
        if st.session_state.alt_profile is not None:
            X_alt = st.session_state.alt_profile.to_vec()
            y_alt = run_model(X_alt)
        else:
            y_alt = None
        y = run_model(X)

        fig = create_plot(y, y_alt)
        st.plotly_chart(fig)


def debug_show(thing, name="", wrap_lines=False):
    if DEBUG:
        st.caption(name)
        st.code(thing, wrap_lines=wrap_lines)


if __name__ == "__main__":
    DEBUG = True
    initialize_state()
    model = load_model("model.pt")
    st.markdown("# T2E")

    if DEBUG:
        # show_vars()
        pass

    render_form()
    render_plot()

    # st.code(model)
