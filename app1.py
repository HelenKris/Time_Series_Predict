import streamlit as st
import pandas as pd
from functional.preproc_functional import seson_conclusion,find_sesonality, first_dif, hist_plot, decompose, ts_test, preprocessing, plot_fig,plot_Ohlc_fig,add_features_preprocessing,to_normal_dist, is_normal
from functional.predict_functional import GRU_select, GRU_predict, prophet, create_features, XGB_predict, prophet_select,XGB_select
import yfinance as yf
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from functional.visualisation import plot_predictictions

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
        dataset = ('Electric data','BA_data','RTX_data','NOC_data', 'data_MW', 'data_weather')
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

    st.subheader("Data for preprocessing")
    data_pred = preprocessing(df)

    st.markdown('---')

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
    # period = 365
    st.session_state.period = period

    agree1 = st.checkbox('Select dataframe with additional features from local database')
    if agree1:
        dataset = ('Electric data','BA_data','NOC_data','RTX_data','RTX_data', 'data_MW', 'data_weather')
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

    prophet_dict = prophet_select(data_predict,period)
    # st.session_state.predict_period = period
    # st.session_state.prophet_MAPE = MAPE
    # st.session_state.prophet_MAE = MAE
    # st.session_state.prophet_MSE = MSE
    # st.write(len(data_pred))
    # st.header("Data Prediction")
    st.header("Forecasting with Prophet")
    all_forecasts = prophet(data_predict,period)
    Prophet_forecast = pd.DataFrame(all_forecasts['yhat'], index=all_forecasts.index)
    # st.write(Prophet_forecast)

    st.markdown('---')
    feature_df = create_features(all_forecasts,period)
    # Объединяем загруженные признаки с дополнительным созданным признаковом пространством\
    if agree1 or agree2:
        feature_df = pd.merge(feature_df,additional_features, on='ds', how='left')
    st.header('Model: XGBoost. Test period: '+str(period)+' days')
    xgb_dict = XGB_select(feature_df,period)
    st.header("Forecasting with XGBoost")
    # st.session_state.feature_df = feature_df
    y_hat_gxb = XGB_predict(feature_df,period)
    st.header('Model: recurrent neural networks with GRU. Test period: '+str(period)+' days')
    # n_epochs = st.number_input('Number of epochs for GRU model training',value=5, key = "num_select_epochs")
    # load_GRU_state = st.text('Please,wait. GRU model is training...')
    gru_dict = GRU_select(feature_df,period=period, num_epochs = 300)
    # load_GRU_state.text('GRU model is done!')

    st.header("Forecasting with GRU model")
    # load_GRU_state = st.text('Please,wait. GRU model is working...')
    # n_epochs = st.number_input('Number of epochs for GRU model training',value=5, key = "num_epochs")
    GRU_forecast = GRU_predict(feature_df,period=period,num_epochs=300)
    # load_GRU_state.text('GRU model is done!')

    all_errors = pd.DataFrame({'Prophet model':pd.Series(prophet_dict),'XGBoost model':pd.Series(xgb_dict),'GRU model':pd.Series(gru_dict)})
    # st.write(all_errors)
    st.session_state.all_errors = all_errors
    all_models_forecasts = pd.merge(y_hat_gxb,GRU_forecast, on='ds', how='right')
    # st.write(all_models_forecasts)
    all_models_forecasts = pd.merge(all_models_forecasts,Prophet_forecast, on='ds', how='right')
    # st.write(all_models_forecasts1)
    # st.write(data_pred)
    all_models_forecasts = pd.merge(all_models_forecasts,data_pred, on='ds', how='outer')
    # st.write(all_models_forecasts)
    st.session_state.all_models_forecasts = all_models_forecasts
    # plot_predictictions(all_models_forecasts, period)
    # plot_predictictions(all_models_forecasts, period)
    # st.write(json_models_forecasts)
    # all_models_forecasts3 = all_models_forecasts2.dropna(subset=['y_prediction','RNN_prediction','yhat'])
    # df = all_models_forecasts3.astype(float).round(2).copy()
    # df = pd.concat([pd.DataFrame([df.columns.values], columns=df.columns), df], ignore_index=True)
    # json_models_forecasts = df.to_json(orient ='values')


    # test = all_models_forecasts2[-2*period:]['y']
    # y_hat_gxb = all_models_forecasts2[-2*period:]['y_prediction']
    # y_hat_RNN = all_models_forecasts2[-2*period:]['RNN_prediction']
    # y_hat_Prophet = all_models_forecasts2[-2*period:]['yhat']
    # fig = go.Figure()
    # fig.add_trace(go.Scatter(x=all_models_forecasts2[-2*period:].index, y=all_models_forecasts2['y'], name="'True value'",line_color='#e74c3c'))
    # fig.add_trace(go.Scatter(x=all_models_forecasts2[-2*period:].index, y=all_models_forecasts2['y_prediction'], name="XGB_prediction",line_color='deepskyblue'))
    # fig.add_trace(go.Scatter(x=all_models_forecasts[-2*period:].index, y=all_models_forecasts['RNN_prediction'], name="'RNN_prediction'",line_color='royalblue'))
    # # fig.add_trace(go.Scatter(x=y_hat_gxb.index, y=y_hat_gxb['y_prediction'], name="XGB_prediction",line_color='#e74c3c'))
    # fig.layout.update(xaxis_rangeslider_visible=True)
    # st.plotly_chart(fig)


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
    # st.markdown('The conclusions about the automatically calculated cyclicity provided in the report are for informational purposes only.')
    # st.markdown('---')
    # st.header("Results of tests for normality of target distribution")
    # st.markdown('**Result of Shapiro-Wilk test for normality of raw target:**  ' +str(st.session_state.normal_raw_data))
    # st.markdown('**Result of Shapiro-Wilk test for normality of target after power-law transformation:**  ' +str(st.session_state.normal_sqrt_data))
    # st.markdown('**Result of Shapiro-Wilk test for normality of target after Logarithm transformation:**  ' +str(st.session_state.normal_log_data))
    # st.markdown('**Result of Shapiro-Wilk test for normality of target after Boxcox transformation:**  ' +str(st.session_state.normal_boxcox_data))
    # st.markdown('---')
    # st.header("Results of test on stationarity")
    # st.markdown('---')
    # st.markdown('**Result of raw target:**  ' +str(st.session_state.addfuller_test))
    # st.markdown('**Result of raw target:**  ' +str(st.session_state.KPSS_test))
    # st.markdown('**Result of target after first differences:**  ' +str(st.session_state.addfuller_test_dif))
    # st.markdown('**Result of target after first differences:**  ' +str(st.session_state.KPSS_test_dif))
    # st.markdown('---')
    st.header("Results of selecting models for predictions")
    st.markdown('---')
    st.write(st.session_state.all_errors)
    plot_predictictions(st.session_state.all_models_forecasts, st.session_state.period)
    # st.markdown('**Mean absolute error of XGBoost model for a period of '+str(st.session_state.predict_period)+' days is** '+str(st.session_state.XGB_MAE))
    # st.markdown('**Mean squared error of XGBoost model for a period of '+str(st.session_state.predict_period)+' days is** '+str(st.session_state.XGB_MSE))
    # st.markdown('**Mean absolute percentage error of XGBoost model for a period of '+str(st.session_state.predict_period)+' days is** '+str(st.session_state.XGB_MAPE*100)+ ' %')
    # st.markdown('---')
    # st.markdown('**Mean absolute error of Prophet model for a period of '+str(st.session_state.predict_period)+' days is** '+str(st.session_state.prophet_MAE))
    # st.markdown('**Mean squared error of Prophet model for a period of '+str(st.session_state.predict_period)+' days is** '+str(st.session_state.prophet_MSE))
    # st.markdown('**Mean absolute percentage error of Prophet model for a period of '+str(st.session_state.predict_period)+' days is** '+str(st.session_state.prophet_MAPE*100)+ ' %')
