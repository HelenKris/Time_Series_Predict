import streamlit as st
import pandas as pd
import statsmodels.api as sm
import plotly.graph_objects as go
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics import tsaplots
import numpy as np
import scipy
from scipy.stats import boxcox
import matplotlib.pyplot as plt
from scipy.stats import shapiro

def decompose(data_pred):
    st.subheader("Decomposition plots of last 2 years")
    decomposition = seasonal_decompose(data_pred.iloc[-730:], model = 'additive')
    seasonal = decomposition.seasonal
    fig = decomposition.plot()
    st.write(fig)

    st.subheader("Seasonal decomposition of resampled data by months ")
    df = data_pred['y'].resample('m').mean().dropna()
    decomposition2 = seasonal_decompose(df, model = 'additive')
    seasonal2 = decomposition2.seasonal
    fig2 = decomposition2.plot()
    st.write(fig2)
    st.write('In this graph you can observe the presence of an annual seasonal component in the time series')
    # return seasonality


def ts_test(data_pred):
    fig1 = tsaplots.plot_acf(data_pred, lags = 24*4)
    fig2 = tsaplots.plot_pacf(data_pred, lags = 40)
    #addfuller
    p = round(sm.tsa.stattools.adfuller(data_pred)[1], 3)
    if p < 0.05:
        st.write('p_value: '+str(p)+'. The row is stationary.')
    else:
        st.write('p_value: '+str(p)+'. The row is not stationary.')
    #KPSS
    p = sm.tsa.stattools.kpss(data_pred)[1]
    if p < 0.05:
        st.write('p_value: '+str(p)+'. The row is not stationary.')
    else:
        st.write('p_value: '+str(p)+'. The row is stationary.')
    st.write(fig1)
    st.write(fig2)

def first_dif(data):
    data['shift_y'] = data['y'] - data['y'].shift(1)
    data =data.dropna()
    ts_test(data['shift_y'])


def preprocessing(df):
    """_summary_
Первичная загрузка данных должна подразумевать запрос на название столбца с датами и название столбца
с целевым признаком и установкой в качестве индекса столбца с датой (переименовать в 'ds' и 'y' как того требует  Prophet).
Значения дат обработаем функцией to_datetime. Сразу будем ресемплить данные по дням (программа не подразумевает анализ суточных
перепадов и минимальная гранула - это дни).
Пропущенные данные по дням (выходные, например), будем заполнять предыдущими значениями.
    Args:
        initial df

    Returns:
        data_pred after preprocessing
    """
    target_column = st.selectbox("Select target column", df.columns, index=1)
    data_column = st.selectbox("Select data column", df.columns, index=0)
    # Переименовываем колонки
    # data_pred = df.reset_index()
    data_pred = df.copy()
    data_pred=data_pred.rename(columns={data_column: "ds", target_column: "y"})
    data_pred['ds'] = pd.to_datetime(data_pred['ds'], errors='coerce')
    # Для ресемплирования устанавливаем колонку даты в качестве индекса
    data_pred = data_pred.set_index('ds')
    data_pred.sort_index(inplace=True)
    # Ресемплируем по дням и заполняем пропущенные предыдущими значением
    data_pred = data_pred.resample('d').mean()
    data_pred = data_pred.asfreq('d')
    data_pred.ffill(inplace=True)
    data_pred.bfill(inplace=True)
    data_pred = pd.DataFrame(data_pred["y"])
    return data_pred

def add_features_preprocessing(additional_data):
    """_summary_
Загрузка дополнительных данных должна подразумевать запрос на название столбца с датами и название столбцов
с дополнительными предикторами с установкой в качестве индекса столбца с датой (переименовать в 'ds' и 'y' как того требует  Prophet).
Значения дат обработаем функцией to_datetime. Сразу будем ресемплить данные по дням (программа не подразумевает анализ суточных
перепадов и минимальная гранула - это дни).
Пропущенные данные по дням (выходные, например), будем заполнять предыдущими значениями (под вопросом).
    Args:
        initial df

    Returns:
        data_pred after preprocessing
    """
    data_column = st.selectbox("Select data column", additional_data.columns, index=0)
    select = st.multiselect('Add column names with additional features from raw dataset', options =additional_data.columns)
    # Переименовываем колонки
    additional_data=additional_data.rename(columns={data_column: "ds"})
    additional_data['ds'] = pd.to_datetime(additional_data['ds'], errors='coerce')
    # Для ресемплирования устанавливаем колонку даты в качестве индекса
    additional_data = additional_data.set_index('ds')
    additional_features = additional_data[select]
    additional_features.sort_index(inplace=True)
    # Ресемплируем по дням и заполняем пропущенные предыдущими значением
    additional_features = additional_features.resample('d').mean()
    additional_features = additional_features.asfreq('d')
    # data_pred.fillna(method='ffill', inplace=True)
    # data_pred = pd.DataFrame(data_pred["y"])
    return additional_features


def plot_fig(data_pred, v_period):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data_pred.index[-v_period:], y=data_pred['y'][-v_period:], name="Value",line_color='deepskyblue'))
    fig.layout.update(title_text='Initial Time Series data',xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)
    return fig

def hist_plot(series):
    fig, ax = plt.subplots()
    ax.hist(series, bins = 30)
    st.pyplot(fig)
    st.markdown('**Result of Shapiro-Wilk test for normality:**  ' +str(is_normal(shapiro(series))))
    return fig


def plot_Ohlc_fig(data_pred):
    df_ohlc = data_pred['y'].resample('10D').ohlc()
    fig = go.Figure(
        data=go.Ohlc(
            x=df_ohlc.index,
            open=df_ohlc["open"],
            high=df_ohlc["high"],
            low=df_ohlc["low"],
            close=df_ohlc["close"]
        )
    )
    fig.layout.update(title_text='OHLC chart (type of bar chart that shows open, high, low, and closing prices).',xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)
    st.write('On the graph you can see the period of greatest decline or rise in the values ​​of the target')
    return fig

def is_normal(test, p_level=0.05):
    _, pval = test
    return 'Normal' if pval > 0.05 else 'Not Normal'

def to_normal_dist(data):
    data['y_sqrt'] = np.sqrt(data['y'])
    data['y_log'] = np.log(data['y'])
    data['y_boxcox'] = boxcox(data['y'], 0)
    return data
