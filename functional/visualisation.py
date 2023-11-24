import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import json
from streamlit_echarts import st_echarts
from streamlit_echarts import JsCode

def plot_mean_prediction(all_models_forecasts, period):
    all_predictions = all_models_forecasts.copy()
    all_predictions['Mean forecast'] = all_predictions[['y_prediction','RNN_prediction','yhat']].mean(axis= 1)
    all_predictions = all_predictions.dropna(subset=['y_prediction','RNN_prediction','yhat'])
    # st.write(all_predictions)
    all_week_predictions = all_predictions.resample('w').mean()
    all_month_predictions = all_predictions.resample('m').mean()
    st.header('Average prediction of all models')
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=all_predictions.index, y=all_predictions['y'], name="'True value'",line_color='deepskyblue'))
    fig.add_trace(go.Scatter(x=all_predictions.index, y=all_predictions['Mean forecast'], name="Average prediction",line_color='#e74c3c'))
    fig.layout.update(xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)
    @st.cache_data
    def convert_df(df):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
        return df.to_csv().encode('utf-8')
    daily_csv = convert_df(all_predictions)
    st.download_button(
        label="Download daily predictions as CSV",
        data=daily_csv,
        file_name='Daily predictions.csv',
        mime='text/csv',
    )

    st.header('Average prediction of all models resempled by weeks')
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=all_week_predictions.index, y=all_week_predictions['y'], name="'True value'",line_color='deepskyblue'))
    fig.add_trace(go.Scatter(x=all_week_predictions.index, y=all_week_predictions['Mean forecast'], name="Average week prediction",line_color='#e74c3c'))
    fig.layout.update(xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)
    weekly_csv = convert_df(all_week_predictions)
    st.download_button(
        label="Download weekly predictions as CSV",
        data=weekly_csv,
        file_name='Weekly predictions.csv',
        mime='text/csv',
    )
    st.header('Average prediction of all models resempled by months')
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=all_month_predictions.index, y=all_month_predictions['y'], name="'True value'",line_color='deepskyblue'))
    fig.add_trace(go.Scatter(x=all_month_predictions.index, y=all_month_predictions['Mean forecast'], name="Average month prediction",line_color='#e74c3c'))
    fig.layout.update(xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)
    monthly_csv = convert_df(all_month_predictions)
    st.download_button(
        label="Download monthly predictions as CSV",
        data=monthly_csv,
        file_name='Monthly predictions.csv',
        mime='text/csv',
    )



def plot_predictictions(all_models_forecasts, period):
    # Блок формирования файла Json для визуализации
    all_models_forecasts = all_models_forecasts.dropna(subset=['y_prediction','RNN_prediction','yhat'])
    df = all_models_forecasts[-3*period:].astype(float).round(2).copy()
    df = df.reset_index()
    df['ds'] = df['ds'].dt.strftime('%Y-%m-%d')
    df.rename(columns={'ds': 'Data','y_prediction': 'XGBoost Prediction', 'RNN_prediction': 'GUN prediction','yhat': 'Prophet prediction','y': 'Historical value'}, inplace=True)
    df = df.melt(id_vars="Data")
    df = df.dropna()
    df.rename(columns={'variable': 'Model', 'value': 'Value'}, inplace=True)
    df = pd.concat([pd.DataFrame([df.columns.values], columns=df.columns), df], ignore_index=True)
    with open('df.json', 'w') as f:
        f.write(df.to_json(orient ='values'))
    with open('df.json') as f:
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
        "animationDuration": 20000,
        "dataset": [{"id": "dataset_raw", "source": raw_data}] + datasetWithFilters,
        "title": {"text": "Predictions of models"},
        "tooltip": {"order": "valueDesc", "trigger": "axis"},
        "xAxis": {"type": "category", "nameLocation": "middle"},
        "yAxis": {"name": "Value"},
        "grid": {"right": 140},
        "series": seriesList,
    }
    st_echarts(options=option, height="500px")
