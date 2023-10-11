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
from scipy.signal import find_peaks

def decompose(data_pred):
    st.subheader("Decomposition plots of last 2 years")
    decomposition = seasonal_decompose(data_pred.iloc[-730:], model = 'additive')
    seasonal = decomposition.seasonal
    fig = decomposition.plot()
    st.write(fig)
    st.subheader("Seasonal decomposition of resampled data by months ")
    df = data_pred['y'].resample('m').mean().dropna()
    decomposition2 = seasonal_decompose(df, model = 'additive')
    fig2 = decomposition2.plot()
    st.write(fig2)
    st.write('In this graph you can observe the presence of an annual seasonal component in the time series')


def ts_test(data_pred):
    fig1 = tsaplots.plot_acf(data_pred, lags = 24*4)
    fig2 = tsaplots.plot_pacf(data_pred, lags = 40)
    #addfuller
    p = round(sm.tsa.stattools.adfuller(data_pred)[1], 3)
    if p < 0.05:
        st.write('p_value: '+str(p)+'. The row is stationary.')
        addfuller_test = 'Augmented Dickey-Fuller test: The row is stationary'
    else:
        st.write('p_value: '+str(p)+'. The row is not stationary.')
        addfuller_test = 'Augmented Dickey-Fuller test: The row is not stationary'
    #KPSS
    p = sm.tsa.stattools.kpss(data_pred)[1]
    if p < 0.05:
        st.write('p_value: '+str(p)+'. The row is not stationary.')
        KPSS_test = 'Kwiatkowski-Phillips-Schmidt-Shin (KPSS) test: The row is not stationary'
    else:
        st.write('p_value: '+str(p)+'. The row is stationary.')
        KPSS_test = 'Kwiatkowski-Phillips-Schmidt-Shin (KPSS) test: The row is stationary'
    st.write(fig1)
    st.write(fig2)
    return addfuller_test, KPSS_test

def first_dif(data):
    data['shift_y'] = data['y'] - data['y'].shift(1)
    data =data.dropna()
    addfuller_test, KPSS_test = ts_test(data['shift_y'])
    return addfuller_test, KPSS_test

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


def plot_fig(data_pred):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data_pred.index, y=data_pred.values, name="Value",line_color='deepskyblue'))
    fig.layout.update(xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)
    return fig

def seson_conclusion(days_peek, week_peek, months_peek, year_peek):
    if days_peek is not None:
        st.markdown('**The time series contains a cycle, which is repeated every** ' +str(days_peek)+ ' days.')
    else:
        st.markdown('**The time series does not have a cyclicity of several days.**')
    if week_peek is not None:
        st.markdown('**The time series contains a cycle, which is repeated every** ' +str(week_peek)+ ' weeks.')
        if week_peek == 52:
            st.markdown('**This seasonality is called annual')
    else:
        st.markdown('**The time series does not have a cyclicity of several weeks.**')
    if months_peek is not None:
        st.markdown('**The time series contains a cycle, which is repeated every** ' +str(months_peek)+ ' months.')
        if months_peek == 12:
            st.markdown('**This seasonality is called annual**')
    else:
        st.markdown('**The time series does not have a cyclicity of several months.**')
    if year_peek is not None:
        st.markdown('**The time series contains a cycle, which is repeated every** ' +str(year_peek)+ ' years.')
    else:
        st.markdown('**The time series does not imply a cyclicity of several years**')


def find_sesonality(data_pred):
    day_series = data_pred['y']
    week_series = data_pred['y'].resample('w').mean().dropna()
    months_series = data_pred['y'].resample('m').mean().dropna()
    year_series = data_pred['y'].resample('y').mean().dropna()
    peak_idx_days, _ = find_peaks(day_series)
    peak_idx_weeks, _ = find_peaks(week_series, prominence = week_series.mean())
    peak_idx_months, _ = find_peaks(months_series, prominence = months_series.mean())
    peak_idx_year, _ = find_peaks(year_series, prominence = year_series.mean())
    peak_data_days = day_series.iloc[peak_idx_days]
    peak_data_weeks = week_series.iloc[peak_idx_weeks]
    peak_data_months = months_series.iloc[peak_idx_months]
    peak_data_year = year_series.iloc[peak_idx_year]
    days_peek = round(len(day_series) / len(peak_data_days)) if len(peak_data_days) else None
    week_peek = round(len(week_series) / len(peak_data_weeks)) if len(peak_data_weeks) else None
    months_peek = round(len(months_series) / len(peak_data_months)) if len(peak_data_months) else None
    year_peek = round(len(year_series) / len(peak_data_year)) if len(peak_data_year) else None
    st.subheader("Visualization of daytime observations of raw data")
    plot_fig(day_series)
    st.subheader("Visualization of weekly observations of data")
    plot_fig(week_series)
    st.subheader("Visualization of monthly observations of raw data")
    plot_fig(months_series)
    st.subheader("Visualization of annual observations of raw data")
    plot_fig(year_series)
    seson_conclusion(days_peek, week_peek, months_peek, year_peek)
    return (days_peek, week_peek, months_peek, year_peek)

def hist_plot(series):
    fig, ax = plt.subplots()
    ax.hist(series, bins = 30)
    st.pyplot(fig)
    st.markdown('**Result of Shapiro-Wilk test for normality:**  ' +str(is_normal(shapiro(series))))
    return is_normal(shapiro(series))


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
