"""Streamlit interface for the locked model in analysis/MELA_inference.ipynb.

Run from the repository root with ``streamlit run webapp/mela_new.py``.
Keep the repository-level models/ directory alongside webapp/. The saved
artifacts use scikit-learn 1.7.1, XGBoost 3.0.5, CatBoost 1.2.8 and NumPy 2.2.6
(see the repository-level requirements.txt); webapp/requirements.txt describes
the older model environment.
"""

import streamlit as st
import json
from pathlib import Path

import numpy as np
import pandas
import joblib

APP_DIR = Path(__file__).resolve().parent
MODELS_DIR = APP_DIR.parent / "models"

glob_title = "MELA: Prediction of acute aortic syndrome"
st.set_page_config(page_title=glob_title)


@st.cache_resource
def load_models():
    with (MODELS_DIR / "manifest.json").open() as fh:
        manifest = json.load(fh)
    pipelines = {
        name: joblib.load(MODELS_DIR / f"pipeline_{name}.joblib")
        for name in manifest["ensemble_members"]
    }
    return dict(
        manifest=manifest,
        preproc=dict(
            imputer=joblib.load(MODELS_DIR / "imputer_final.joblib"),
            scaler=joblib.load(MODELS_DIR / "scaler_final.joblib"),
        ),
        thresholds={
            name: manifest["thresholds"][name]
            for name in manifest["ensemble_members"]
        },
        models={name: pipe.named_steps["clf"] for name, pipe in pipelines.items()},
        inputs=pandas.read_excel(APP_DIR / "interface_data.xlsx", sheet_name="features"),
    )


def add_feature(feat):
    str2bool = {"True": True, "False": False}

    feat_mapping = None
    if feat["TYPE"] == "bool":
        widget = st.toggle
        kwargs = dict(value=str2bool[feat["INFO"]])
    elif feat["TYPE"] in {"int", "logint"}:
        kwargs = dict(
            zip(
                ["min_value", "max_value", "value", "step"],
                map(int, feat["INFO"].split("-")),
            )
        )
        widget = st.number_input
    elif feat["TYPE"] == "cat":
        cats = feat["INFO"].split(";")
        kwargs = dict(
            options=cats,
        )
        widget = st.radio
        feat_mapping = lambda cat: {cat: i for i, cat in enumerate(cats)}[cat]
    else:
        raise ValueError(
            f"bad type {feat['TYPE']} for feature {feat['NAME']} in feature table"
        )

    # widget creation
    feat_value = widget(label=feat["LABEL"], **kwargs)
    if feat_mapping is not None:
        feat_value = feat_mapping(feat_value)
    return feat_value

def add_features(feat_values, feature_df):
    for n, feat in feature_df.iterrows():
        feat_values[feat["NAME"]] = add_feature(feat)

icons = {
    'DEMOGRAPHICS': 'demography',
    'MEDICAL HISTORY': 'medical_information',
    'PAIN': 'sentiment_dissatisfied',
    'OTHER SIGNS/SYMPTOMS': 'stethoscope',
    'D-DIMER TEST RESULT': 'bloodtype',
}

def create_input_widgets_groups_hardcoded(feature_df, container_func=st.columns):
    groups = dict(iter(feature_df.groupby("CONTAINER", sort=False)))
    try:
        containers = container_func(groups.keys())
        add_title = False
    except TypeError:
        containers = container_func(len(groups))
        add_title = True
        add_title = False

    feat_values = {}
    for container, (title, container_feature_df) in zip(containers, groups.items()):
        with container:
            if add_title:
                st.markdown(f'#### {title}')
            for group, group_feature_df in container_feature_df.groupby('GROUP', sort=False, dropna=False):
                if pandas.isnull(group):
                    add_features(feat_values, group_feature_df)
                else:
                    with st.expander(
                        group, 
                        #expanded=len(group_feature_df) < 5, 
                        expanded=False,
                        icon=f':material/{icons[group]}:',
                    ):
                        #st.markdown(f'##### {group}')
                        add_features(feat_values, group_feature_df)
    return feat_values


def input_to_dataframe(input_values):
    expected = set(feature_names)
    if expected != input_values.keys():
        raise ValueError(
            f"Input features differ from the manifest: "
            f"{expected.symmetric_difference(input_values.keys())}"
        )
    return pandas.DataFrame(input_values, columns=feature_names, index=[0])


def preprocess(input_df):
    """Apply the deterministic preprocessing from MELA_inference.ipynb."""
    df = input_df.copy()
    mapping_smoke = {
        "never smoked": 0, "former smoker": 1, "unknown": 0,
        "active smoker": 1, 0: 0, "0": 0,
    }
    if "smoke" in df.columns and df["smoke"].dtype == object:
        df["smoke"] = df["smoke"].map(mapping_smoke).fillna(0).astype(int)

    for column in models["manifest"]["fixed_zero_columns"]:
        if column in df.columns:
            df[column] = df[column].fillna(0.0)

    # The interface accepts only positive D-dimer values and has no shock input.
    for column in models["manifest"]["log2_transform"]:
        if column not in df.columns:
            raise ValueError(f"'{column}' is required and was not found")
        if (df[column].isna() | (df[column] <= 0)).any():
            raise ValueError(f"'{column}' must contain positive, non-missing values")
        df[column] = np.log2(df[column])

    return df.reindex(columns=feature_names)


def predict_voting(input_df):
    processed_input_df = preprocess(input_df)
    imputed_input_df = models["preproc"]["imputer"].transform(processed_input_df)
    scaled_input_df = models["preproc"]["scaler"].transform(imputed_input_df)

    scores = pandas.DataFrame(
        {
            name: model.predict_proba(scaled_input_df)[:, 1]
            for name, model in models["models"].items()
        },
        index=input_df.index,
    )
    thresholds = pandas.Series(models["thresholds"])
    preds = scores.ge(thresholds, axis=1)
    pred_sum = preds.sum(axis=1)
    return pred_sum, scores, preds


outcomes = pandas.DataFrame(
    [
        # last version: pooled data from 5,571 patients in the derivation and validation cohorts
        ["I", 0.0010, 1000, "low risk (AAS unlikely: consider rule-out and alternative diagnosis)", "#50ff50"], # 0 or 1 votes
        ["II", 0.1036, 10, "high risk (AAS likely: consider advanced aortic imaging for conclusive diagnosis)", "#ff5050"], # 2 or 3 votes

    ],
    columns=["Class", "RISK", "RISK_DEN", "DESC", "COLOR"],
).set_index("Class")
outcomes["Risk (%)"] = outcomes["RISK"].map("{:.2%}".format)
outcomes["Risk (1/N)"] = outcomes["RISK_DEN"].map("one in {}".format)
outcomes["Desc"] = outcomes["DESC"]

models = load_models()
feature_names = models["manifest"]["feature_order"]

st.title(glob_title)
st.markdown(
    "This prediction tool has been developed as a decision-support tool to aid in "
    "the clinical assessment of adult patients who are evaluated for suspected acute "
    "aortic syndrome (aortic dissection, intramural aortic hematoma, penetrating "
    "aortic ulcer, spontaneous aortic rupture). At present, this application may be "
    "used for *educational and research purposes only*. For further information, please refer to the manuscript..."
)
st.subheader("Model input variables")
feat_values = create_input_widgets_groups_hardcoded(models["inputs"], st.columns)
input_df = input_to_dataframe(feat_values)
st.subheader(f"Predicted risk of acute aortic syndrome")
pred_sum, scores, preds = predict_voting(input_df)
assert len(pred_sum) == 1
pred_sum = pred_sum.loc[0]
outtable = outcomes.loc[:, ["Risk (%)", "Risk (1/N)", "Desc"]]
risk_class = "II" if pred_sum >= models["manifest"]["n_votes_positive"] else "I"

def highlight_row(row):
    color = outcomes.at[risk_class, "COLOR"]
    if row.name == risk_class:  # Change the row index you want to highlight
        return ["font-weight: bold; background-color: " + color] * len(row)
    else:
        return [""] * len(row)


outtable = outtable.style.apply(highlight_row, axis=1)
st.dataframe(outtable)