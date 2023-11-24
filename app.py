import streamlit as st
import pandas as pd
from functional.preproc_functional import seson_conclusion,find_sesonality, first_dif, hist_plot, decompose, ts_test, preprocessing, plot_fig,plot_Ohlc_fig,add_features_preprocessing,to_normal_dist, is_normal
from functional.predict_functional import GRU_select, GRU_predict, prophet, create_features, XGB_predict, prophet_select,XGB_select
import yfinance as yf
import matplotlib.pyplot as plt
from functional.visualisation import plot_predictictions,plot_mean_prediction

plt.style.use('https://github.com/dhaitz/matplotlib-stylesheets/raw/master/pitayasmoothie-light.mplstyle')

style = "<style>h2 {text-align: center;}</style>"
st.markdown(style, unsafe_allow_html=True)

title_alignment= """
<style>
#the-title {
  text-align: center
}
</style>
"""
st.markdown(title_alignment, unsafe_allow_html=True)

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

st.markdown("""## This is an application for those who want to explore time series data and make predictions based on it""")
st.image('wolf.jpg')

page = st.sidebar.selectbox("Choose a page", ["Data loading and preprocessing", "Exploration", 'Prediction', 'Conclusions'])
if page == "Data loading and preprocessing":
    st.markdown('---')
    st.header("Data loading and data preprocessing")
    radio_btn = st.radio('Select data source: ', options = ('Select data from local database','Download data from YAHOO', 'Upload my own data'), index = 0)

    if radio_btn == 'Select data from local database':
        dataset = ('NKE','Electric data','BA_data','RTX_data','NOC_data', 'data_MW', 'data_weather')
        option = st.selectbox('Select dataset for prediction from database',dataset)
        DATA_URL =('./HISTORICAL_DATA/'+option+'.csv')
        df = pd.read_csv(DATA_URL)
        st.session_state.df = df

    elif radio_btn == 'Download data from YAHOO':
        st.markdown('Download your data directly from [YAHOO](https://www.finance.yahoo.com). Just enter the company ticker, the start and the end of time period')
        ticker = st.text_input("Enter a ticker symbol", 'GOOGL')
        start_time = st.date_input("Start of time period")
        end_time = st.date_input("End of time period")
        df = yf.download(ticker, start_time, end_time)
        df = df.reset_index()
        resuld = st.button('Apply')
        st.write('Data has been downloaded.')
        st.session_state.df = df

    else:
        uploaded_data = st.file_uploader('Please upload data in format .csv', type = 'csv')
        if uploaded_data is not None:
            df = pd.read_csv(uploaded_data)
            st.session_state.df = df

    df = st.session_state.df
    if st.checkbox('Show raw data'):
        st.subheader("This is your initial data.")
        st.write("Please select a page on the left.")
        st.write(df)

    st.markdown('---')
    target_column = st.selectbox("Select target column", df.columns, index=1)
    data_column = st.selectbox("Select data column", df.columns, index=0)
    st.session_state.target_column = target_column
    st.session_state.data_column = data_column

    agree = st.checkbox('Show data after preprocessing', value=True)
    if agree:
        try:
            data_pred = preprocessing(df,target_column,data_column)
            st.session_state.data_pred = data_pred
            st.subheader('Your data after preprocessing')
            st.write("Please select a page on the left.")
            st.write(data_pred)
        except Exception as e:
            st.write("Select the correct date and target column!")
    st.session_state.agree = agree
    st.markdown('---')


elif page == "Exploration":
    st.header("Data exploration")
    st.markdown('---')
    if st.session_state.agree:
        data_pred = st.session_state.data_pred
    else:
        data_pred = preprocessing(st.session_state.df, st.session_state.target_column,st.session_state.data_column)

    max_date = data_pred.index.max()
    min_date = data_pred.index.min()
    st.markdown('**Data are presented for the period from** '+str(min_date)+' to '+str(max_date))
    st.markdown('**Data includes** ' +str(len(data_pred))+ ' **daily observations**')
    # Формируем выжимки для выводов:
    st.session_state.min_date = min_date
    st.session_state.max_date = max_date
    st.session_state.len_data = len(data_pred)
    st.session_state.sesonality = find_sesonality(data_pred)
    # Свечной график для визуализации изменений
    plot_Ohlc_fig(data_pred)

    st.header("Tests for normality of target distribution")
    normal_raw_data = hist_plot(data_pred)
    st.session_state.normal_raw_data = normal_raw_data
    st.header("Methods for brining  distribution to normal distribution")
    normal_data = data_pred.copy()
    normal_data = to_normal_dist(normal_data)
    st.subheader('Power-law transformation of data (taking from the square root)')
    normal_sqrt_data = hist_plot(normal_data['y_sqrt'])
    st.session_state.normal_sqrt_data = normal_sqrt_data
    st.subheader('Transformation by Logarithm')
    normal_log_data = hist_plot(normal_data['y_log'])
    st.session_state.normal_log_data = normal_log_data
    st.subheader('Boxcox transformation')
    normal_boxcox_data = hist_plot(normal_data['y_boxcox'])
    st.session_state.normal_boxcox_data = normal_boxcox_data


    st.header("Time Series Decomposition")
    decompose(data_pred)
    addfuller_test, KPSS_test = ts_test(data_pred)
    st.session_state.addfuller_test = addfuller_test
    st.session_state.KPSS_test = KPSS_test

    st.header("Methods for bringing a series to stationarity")
    st.subheader("To stationarity by the method of first differences")
    dif_data = data_pred.copy()
    addfuller_test_dif, KPSS_test_dif = first_dif(dif_data)
    st.session_state.addfuller_test_dif = addfuller_test_dif
    st.session_state.KPSS_test_dif = KPSS_test_dif
    st.markdown('---')

elif page == "Prediction":
    st.markdown('---')
    st.header("Model selection")
    data_pred = st.session_state.data_pred
    df = st.session_state.df
    n_months = st.slider('Months of prediction:', 1,12, key = "n_months")
    period = n_months * 30
    st.session_state.period = period

    agree1 = st.checkbox('Select dataframe with additional features from local database')
    if agree1:
        dataset = ('NKE','Electric data','BA_data','NOC_data','RTX_data','RTX_data', 'data_MW', 'data_weather')
        option = st.selectbox('Select dataset for prediction from database',dataset)
        DATA_URL =('./HISTORICAL_DATA/'+option+'.csv')
        additional_data = pd.read_csv(DATA_URL)
        additional_features = add_features_preprocessing(additional_data)

    agree2 = st.checkbox('Download dataframe with additional features')
    if agree2:
        uploaded_data = st.file_uploader('Please upload data with additional features.csv', type = 'csv')
        if uploaded_data is not None:
            additional_data = pd.read_csv(uploaded_data)
            additional_features = add_features_preprocessing(additional_data)
    data_predict = data_pred.reset_index()
    st.session_state.data_predict = data_predict
    prophet_dict = prophet_select(data_predict,period)
    st.header("Forecasting with Prophet")
    all_forecasts = prophet(data_predict,period)
    Prophet_forecast = pd.DataFrame(all_forecasts['yhat'], index=all_forecasts.index)
    st.session_state.all_forecasts = all_forecasts

    st.markdown('---')
    feature_df = create_features(all_forecasts,period)
    # Объединяем загруженные признаки с дополнительным созданным признаковом пространством\
    if agree1 or agree2:
        feature_df = pd.merge(feature_df,additional_features, on='ds', how='left')
    st.header('Model: XGBoost. Test period: '+str(period)+' days')

    xgb_dict = XGB_select(feature_df,period)
    st.header("Forecasting with XGBoost")
    y_hat_gxb = XGB_predict(feature_df,period)

    st.header('Model: recurrent neural networks with GRU. Test period: '+str(period)+' days')
    load_GRU_state = st.text('Please,wait. GRU model is working...')
    gru_dict = GRU_select(feature_df,period=period, num_epochs = 300)
    load_GRU_state.text('GRU model is done!')

    st.header("Forecasting with GRU model")
    load_GRU_state = st.text('Please,wait. GRU model is working...')
    GRU_forecast = GRU_predict(feature_df,period=period,num_epochs=300)
    load_GRU_state.text('GRU model is done!')

    all_errors = pd.DataFrame({'Prophet model':pd.Series(prophet_dict),'XGBoost model':pd.Series(xgb_dict),'GRU model':pd.Series(gru_dict)})
    st.session_state.all_errors = all_errors
    all_models_forecasts = pd.merge(y_hat_gxb,GRU_forecast, on='ds', how='right')
    all_models_forecasts = pd.merge(all_models_forecasts,Prophet_forecast, on='ds', how='right')
    all_models_forecasts = pd.merge(all_models_forecasts,data_pred, on='ds', how='outer')
    st.session_state.all_models_forecasts = all_models_forecasts

elif page == 'Conclusions':
    st.header("Сonclusions from time series analysis and prediction results")
    st.markdown('---')
    st.markdown('**Data are presented for the period from** '+str(st.session_state.min_date)+' to '+str(st.session_state.max_date))
    st.markdown('**Data includes** ' +str(st.session_state.len_data)+ ' **daily observations**')
    # Формируем выжимки для выводов:
    st.markdown('---')
    st.header("Results of determining cyclicity in a time series")
    days_peek, week_peek, months_peek, year_peek  = st.session_state.sesonality
    st.markdown('---')
    seson_conclusion(days_peek, week_peek, months_peek, year_peek)
    st.markdown('---')
    st.header("Results of selecting models for predictions")
    st.markdown('---')
    st.write(st.session_state.all_errors)
    plot_predictictions(st.session_state.all_models_forecasts, st.session_state.period)
    plot_mean_prediction(st.session_state.all_models_forecasts, st.session_state.period)
