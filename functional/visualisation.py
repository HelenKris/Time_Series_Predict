import streamlit as st
# import pandas as pd
# import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error,mean_absolute_percentage_error
import plotly.graph_objects as go
import json
from streamlit_echarts import st_echarts
from streamlit_echarts import JsCode
with open("E:\IT_academy\Time_Series_Predict\HISTORICAL_DATA\df.json") as f:
    raw_data = json.load(f)
models = ["XGBoost Prediction","GUN prediction","Prophet prediction","Historical value"]

datasetWithFilters = [
    {
        "id": f"dataset_{model}",
        "fromDatasetId": "dataset_raw",
        "transform": {
            "type": "filter",
            "config": {
                "and": [
                    # {"dimension": "Data", "gte": "2018-01-12"},
                    # {"dimension": "Data"},
                    {"dimension": "Model", "=": model},
                ]
            },
        },
    }
    for model in models
]

seriesList = [
    {
        "type": "line",
        "datasetId": f"dataset_{model}",
        "showSymbol": False,
        "name": model,
        "endLabel": {
            "show": False,
            "formatter": JsCode(
                "function (params) { return params.value[3] + ': ' + params.value[0];}"
            ).js_code,
        },
        "labelLayout": {"moveOverlap": "shiftY"},
        "emphasis": {"focus": "series"},
        "encode": {
            "x": "Data",
            "y": "Value",
            "label": ["Model", "Value"],
            "itemName": "Data",
            "tooltip": ["Value"],
        },
    }
    for model in models
]

option = {
    "animationDuration": 10000,
    "dataset": [{"id": "dataset_raw", "source": raw_data}] + datasetWithFilters,
    "title": {"text": "Predictions of models"},
    "tooltip": {"order": "valueDesc", "trigger": "axis"},
    "xAxis": {"type": "category", "nameLocation": "middle"},
    "yAxis": {"name": "Value"},
    "grid": {"right": 140},
    "series": seriesList,
}
st_echarts(options=option, height="600px")

def plot_predictictions(all_models_forecasts, period):
    test = all_models_forecasts[-2*period:]['y']
    y_hat_gxb = all_models_forecasts[-2*period:]['y_prediction']
    y_hat_RNN = all_models_forecasts[-2*period:]['RNN_prediction']
    y_hat_Prophet = all_models_forecasts[-2*period:]['yhat']
    MAE_xgb = round(mean_absolute_error(test[-2*period:-period], y_hat_gxb[-2*period:-period]),3)
    r2_score_xgb = round(r2_score(test[-2*period:-period], y_hat_gxb[-2*period:-period]),3)
    r2_score_RNN = round(r2_score(test[-2*period:-period], y_hat_RNN[-2*period:-period]),3)
    MAE_RNN = round(mean_absolute_error(test[-2*period:-period], y_hat_RNN[-2*period:-period]),3)
    r2_score_Prophet = round(r2_score(test[-2*period:-period], y_hat_Prophet[-2*period:-period]),3)
    MAE_Prophet = round(mean_absolute_error(test[-2*period:-period], y_hat_Prophet[-2*period:-period]),3)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=all_models_forecasts[-2*period:].index, y=all_models_forecasts['y'], name="True value",line_color='#e74c3c'))
    fig.add_trace(go.Scatter(x=all_models_forecasts[-2*period:].index, y=all_models_forecasts['y_prediction'], name="XGB model prediction",line_color='deepskyblue'))
    fig.add_trace(go.Scatter(x=all_models_forecasts[-2*period:].index, y=all_models_forecasts['RNN_prediction'], name="GUN model prediction'",line_color='royalblue'))
    fig.add_trace(go.Scatter(x=all_models_forecasts[-2*period:].index, y=all_models_forecasts['yhat'], name="Prophet model prediction'",line_color='#ff9c24'))
    fig.layout.update(xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)
    # st.markdown('**R2 score of XGBoost model is** ' +str(r2_score_xgb)+ ' **daily observations**')
    st.markdown('**R2 score of XGBoost model is** ' +str(r2_score_xgb))
    st.markdown('**R2 score of GUN model is** ' +str(r2_score_RNN))
    st.markdown('**R2 score of Prophet model is** ' +str(r2_score_Prophet))
    st.markdown('**Mean absolute error of XGBoost model is** ' +str(MAE_xgb))
    st.markdown('**Mean absolute error of GUN model is** ' +str(MAE_RNN))
    st.markdown('**Mean absolute error of Prophet model is** ' +str(MAE_Prophet))
    # return fig
