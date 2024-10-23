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
import streamlit as st


dimIn, dimOut = 29, 77


race_map = {"White": 0, "Black": 1, "Other": 2, "Unkown": 3}
eth_map = {"Non-Hispanic": 0, "Hispanic": 1, "Other": 2, "Unkown": 3}


def initialize_state():
    if "profile" not in st.session_state:
        st.session_state.profile = None


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


def plot_result(cifs):
    plt.stackplot(
        np.arange(dimOut + 4) / 4.0, 100 * cifs, labels=["ADRD", "Death w/o ADRD"]
    )
    ax = plt.gca()
    yticks = ax.get_yticks()
    ndec = max(0, -np.floor(np.log10(yticks[1])))
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(decimals=ndec))
    plt.legend(loc="upper left")
    plt.xlabel("Elapsed Time (Years)")
    plt.ylabel("Cumulative Incidence")
    plt.title("Cumulative Incidence Functions")
    return plt.gcf()


def show_vars():
    if "profile" in st.session_state and st.session_state.profile is not None:
        debug_show(st.session_state.profile, "profile", wrap_lines=True)
        X = st.session_state.profile.to_vec()
        debug_show(X, "input")
        y = run_model(X)
        debug_show(torch.from_numpy(y), "output")
        fig = plot_result(y)
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

        # Fitness & BMI scores
        st.markdown("### Fitness & Weight")
        # Default val = 7.0 (median from stats)
        fw1, fw2 = st.columns(2)
        fit = fw1.number_input(
            "Fitness", min_value=2.0, max_value=24.0, step=0.25, value=7.0
        )
        # Default val = 28.7 (median from stats)
        bmi = fw2.number_input(
            "Body Mass Index", min_value=18.5, max_value=74.5, step=0.25, value=29.0
        )

        # Comorbidities
        ## TODO maybe reformat as multiselect
        st.markdown("### Comorbidities")
        # c1, c2, c3, c4 = st.columns(4)
        #
        # # first col
        # afib = c1.checkbox("afib")
        # anemia = c1.checkbox("anemia")
        # arthritis = c1.checkbox("arthritis")
        # asthma = c1.checkbox("asthma")
        # cancer = c1.checkbox("cancer")
        #
        # # 2nd col
        # ckd = c2.checkbox("ckd")
        # copd = c2.checkbox("copd")
        # depression = c2.checkbox("depression")
        # diabetes = c2.checkbox("diabetes")
        # heart_failure = c2.checkbox("heart failure")
        #
        # # 3rd col
        # hypertension = c3.checkbox("hypertension")
        # hyperlipidemia = c3.checkbox("hyperlipidemia")
        # isch_heart = c3.checkbox("ischemic heart disease")
        # stroke = c3.checkbox("stroke")
        # tbi = c3.checkbox("tbi")
        #
        # # 4th col
        # statin = c4.checkbox("statin")
        # dmrx = c4.checkbox("DmRx")
        # cvdrx = c4.checkbox("CVDRx")
        # asma_copd = c4.checkbox("asma_copd")
        # cvd = c4.checkbox("CVD")

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
            "dmrx",
            "cvdrx",
            "asthma copd",
            "cvd",
        ]
        comorbidities = st.multiselect(
            "comorbidities",
            options=option_names,
            format_func=display_opt,
            label_visibility="hidden",
        )

        submit = st.form_submit_button(
            "Compute Risk Scores",
            type="primary",
            # on_click=on_submit,
            # args=tuple(comorbidities),
            # kwargs={
            #     "age": age,
            #     "sex": sex,
            #     "race": race,
            #     "ethnicity": eth,
            #     "fitness": fit,
            #     "bmi": bmi,
            # },
        )

    if submit:
        profile = RiskProfile.from_form(
            *tuple(comorbidities),
            age=age,
            sex=sex,
            race=race,
            ethnicity=eth,
            fitness=fit,
            bmi=bmi,
        )
        st.session_state.profile = profile
        render_plot()


def render_plot():
    if "profile" in st.session_state and st.session_state.profile is not None:
        X = st.session_state.profile.to_vec()
        y = run_model(X)
        fig = plot_result(y)
        st.pyplot(fig)


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
