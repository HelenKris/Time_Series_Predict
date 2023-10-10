import streamlit as st
import pandas as pd
from functional.preproc_functional import first_dif, hist_plot, decompose, ts_test, preprocessing, plot_fig,plot_Ohlc_fig,add_features_preprocessing,to_normal_dist, is_normal
from functional.predict_functional import prophet, create_features, XGB_predict, prophet_select, XGB_select
import yfinance as yf
import matplotlib.pyplot as plt
import plotly.graph_objects as go

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

page = st.sidebar.selectbox("Choose a page", ["Data loading and preprocessing", "Exploration", 'Prediction'])
if page == "Data loading and preprocessing":
    st.markdown('---')
    st.header("Data loading and data preprocessing")
    radio_btn = st.radio('Select data source: ', options = ('Select data from local database','Download data from YAHOO', 'Upload my own data'), index = 0)

    if radio_btn == 'Select data from local database':
        dataset = ('BA_data','NOC_data','RTX_data', 'data_MW', 'data_weather')
        option = st.selectbox('Select dataset for prediction from database',dataset)
        DATA_URL =('./HISTORICAL_DATA/'+option+'.csv')
        df = pd.read_csv(DATA_URL)
        if 'df' not in st.session_state:
            st.session_state.df = df

    elif radio_btn == 'Download data from YAHOO':
        st.markdown('Download your data directly from [YAHOO](https://www.finance.yahoo.com). Just enter the company ticker, the start and the end of time period')
        ticker = st.text_input("Enter a ticker symbol")
        start_time = st.date_input("Start of time period")
        end_time = st.date_input("End of time period")
        df = yf.download(ticker, start_time, end_time)
        df = df.reset_index()
        resuld = st.button('Apply')
        st.write('Data has been downloaded.')
        if 'df' not in st.session_state:
            st.session_state.df = df


    else:
        uploaded_data = st.file_uploader('Please upload data in format .csv', type = 'csv')
        if uploaded_data is not None:
            df = pd.read_csv(uploaded_data)
            if 'df' not in st.session_state:
                st.session_state.df = df

    if st.checkbox('Show raw data'):
        st.subheader("This is your initial data.")
        st.write("Please select a page on the left.")
        st.write(df)

    st.markdown('---')
    st.subheader("Data for preprocessing")
    data_pred = preprocessing(df)

    st.markdown('---')
    st.subheader("Visualization of raw data")
    n_months = st.slider('Period of visualization in months:', 1,round(len(data_pred)/30),0)
    v_period = n_months * 30

    plot_fig(data_pred,v_period)

    if st.checkbox('Show data after preprocessing'):
        st.subheader('Your data after preprocessing')
        st.write("Please select a page on the left.")
        st.write(data_pred)

    st.markdown('---')

    st.session_state.data_pred = data_pred


elif page == "Exploration":
    st.header("Data exploration")
    st.markdown('---')
    data_pred = st.session_state.data_pred
    max_date = data_pred.index.max()
    min_date = data_pred.index.min()
    st.markdown('**Data are presented for the period from** '+str(min_date)+' to '+str(max_date))
    plot_Ohlc_fig(data_pred)

    st.header("Tests for normality of target distribution")

    hist_plot(data_pred['y'])

    st.header("Methods for brining  distribution to normal distribution")
    normal_data = data_pred.copy()
    normal_data = to_normal_dist(normal_data)
    st.subheader('Power-law transformation of data (taking from the square root)')
    hist_plot(normal_data['y_sqrt'])
    st.subheader('Transformation by Logarithm')
    hist_plot(normal_data['y_log'])
    st.subheader('Doxcox transformation')
    hist_plot(normal_data['y_boxcox'])


    st.header("Time Series Decomposition")
    decompose(data_pred)
    ts_test(data_pred)

    st.header("Methods for bringing a series to stationarity")
    st.subheader("To stationarity by the method of first differences")
    dif_data = data_pred.copy()
    first_dif(dif_data )
    st.markdown('---')

elif page == "Prediction":
    st.markdown('---')
    st.header("Model selection")
    data_pred = st.session_state.data_pred
    df = st.session_state.df
    n_months = st.slider('Months of prediction:', 1,12, key = "n_months")
    period = n_months * 30

    agree1 = st.checkbox('Select dataframe with additional features from local database')
    if agree1:
        dataset = ('BA_data','NOC_data','RTX_data', 'data_MW', 'data_weather')
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
    prophet_select(data_predict,period)

    st.header("Data Prediction")
    st.header("Forecasting with Prophet")
    all_forecasts = prophet(data_predict,period)
    st.markdown('---')
    feature_df = create_features(all_forecasts)
    # Объединяем загруженные признаки с дополнительным созданным признаковом пространством\
    if agree1 or agree2:
        feature_df = pd.merge(feature_df,additional_features, on='ds', how='left')

    st.header('Model: XGBoost. Test period: '+str(period)+' days')
    XGB_select(feature_df,period)

    st.header("Forecasting with XGBoost")
    XGB_predict(feature_df,period)
