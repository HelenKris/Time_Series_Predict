import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error,mean_absolute_percentage_error
import xgboost as xgb
import matplotlib.pyplot as plt

def plot_predict_data(data, v_period, col_name_target, col_name_predict):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index[-v_period:], y=data[col_name_target][-v_period:], name="True value",line_color='royalblue'))
    fig.add_trace(go.Scatter(x=data.index[-v_period:], y=data[col_name_predict][-v_period:], name="Predictions",line_color='#e74c3c'))
    st.plotly_chart(fig)
    return fig

def prophet(data_pred,period):
    """""_summary_""

    Args:
        data_pred (_type_): _description_
        period (_type_): _description_

    Returns:
    Возвращает датафрейм с индексом ds для конкатинации и столбцом у (содержат NaN в прогнозах)
    и столбцом 'yhat' - предсказаниями модели.
        _type_: _description_
    """
    m = Prophet(yearly_seasonality=True)
    m.fit(data_pred)
    future = m.make_future_dataframe(periods=period)
    forecast = m.predict(future)

    #plot forecast
    # fig1 = plot_plotly(m, forecast)

    st.header('Model: Prophet. Forecasting period: '+str(period)+' days')
    stock_price_forecast = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    all_forecasts = pd.merge(data_pred,stock_price_forecast, on='ds', how='right')
    all_forecasts = all_forecasts.set_index('ds')
    n_months = st.slider('Period of visualization in months:', 1,round(len(data_pred)/30),0)
    v_period = n_months * 30

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=stock_price_forecast[-v_period:].ds, y=all_forecasts['y'][-v_period:], name="Adj Close True",line_color='royalblue'))
    fig.add_trace(go.Scatter(x=stock_price_forecast[-v_period:].ds, y=all_forecasts['yhat_lower'][-v_period:], name="yhat_lower",line_color='#e74c3c'))
    fig.add_trace(go.Scatter(x=stock_price_forecast[-v_period:].ds, y=all_forecasts['yhat'][-v_period:], name="yhat",line_color='deepskyblue'))
    fig.add_trace(go.Scatter(x=stock_price_forecast[-v_period:].ds, y=all_forecasts['yhat_upper'][-v_period:], name="yhat_upper",line_color='#e74c3c'))
    st.plotly_chart(fig)
    if st.checkbox('Show forecast data'):
        st.subheader('forecast data')
        st.write(forecast)
    all_forecasts = pd.DataFrame(all_forecasts[['y','yhat']])
    return all_forecasts

def prophet_select(data_pred,period):
    """""_summary_""

    Args:
        data_pred (_type_): _description_
        period (_type_): _description_

    Returns:
    Возвращает датафрейм с индексом ds для конкатинации и столбцом у (содержат NaN в прогнозах)
    и столбцом 'yhat' - предсказаниями модели.
        _type_: _description_
    """
    m = Prophet(yearly_seasonality=True)
    train_df = data_pred[:-period]
    m.fit(train_df)
    future = m.make_future_dataframe(periods=period)
    forecast = m.predict(future)

    st.header('Model: Prophet. Test period: '+str(period)+' days')
    stock_price_forecast = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    all_forecasts = pd.merge(train_df,stock_price_forecast, on='ds', how='right')
    all_forecasts = all_forecasts.set_index('ds')
    n_months_visual = st.slider('Period of visualization in months:', 1,round(len(data_pred)/30),0, key = 'v_period')
    v_period = n_months_visual * 30
    test = data_pred['y'][len(data_pred)- period:]
    y_pred = stock_price_forecast['yhat'][len(data_pred)- period:]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data_pred[-v_period:].ds, y=data_pred['y'][-v_period:], name="True value",line_color='deepskyblue'))
    fig.add_trace(go.Scatter(x=data_pred['ds'][len(data_pred)- period:], y=data_pred['y'][len(data_pred)- period:], name="True value of test period",line_color='royalblue'))
    fig.add_trace(go.Scatter(x=stock_price_forecast[-v_period:].ds, y=stock_price_forecast['yhat'][-v_period:], name="Predicted values",line_color='#e74c3c'))
    st.plotly_chart(fig)

    # Логика оценки: мы оцениваем модель по такому же интервалу, по которому прогназируем. Т.е. если хотим прогнозировать на год, то
    # оцениваем модель по средней ошибке за предыдущий год, если прогнозируем на неделю, то оцениваем тоже по последней неделе.
    MAE = round(mean_absolute_error(test, y_pred),3)
    MAPE = round(mean_absolute_percentage_error(test, y_pred),3)
    st.markdown('**Mean absolute error of Prophet model for a period of '+str(period)+' days is** '+str(MAE))
    st.markdown('**Mean absolute percentage error of Prophet model for a period of '+str(period)+' days is** '+str(MAPE*100)+ ' %')
    return all_forecasts


def create_features(df):
    """В функцию передается all_forecasts = prophet(data_pred,period)
    Функция возвращает датафрейм с колонками доп.признаков и индексом в datetime для последующей конкатенации
    с целевым признаком и предсказаниями prophet тоже
    """
    # Extract time-based features
    df['week'] = df.index.weekday
    df['day'] = df.index.day
    df['month'] = df.index.month

    # Lagged features
    lag_list = [365]
    # lag_list = [2, 7, 14, 30,60,90,365]
    for i in lag_list:
        df["lag_{}".format(i)] = df.y.shift(i)

    # Presidential term cycle feature
    # df['presidential_term_cycle'] = df.index.year % 4  # Remainder of current year divided by 4
    return df

def XGB_select(feature_df,period):
    """""_summary_""

Args:
    feature_df (_type_):обогащенный дополнительными признаками датасет c целевым признаком и предсказаниями Prohet
    period (_type_): _description_

Returns:
Возвращает датафрейм с индексом ds для конкатинации и столбцом у (содержат NaN в прогнозах)
и столбцом 'yhat_xgb' - предсказаниями модели.
    _type_: _description_
"""
    train = feature_df[:-2*period]
    test = feature_df[len(feature_df)-2*period:]
    FEATUREAS_df = feature_df.drop(['y', 'yhat'], axis = 1)
    FEATUREAS = FEATUREAS_df.columns
    TARGET = 'y'
    X_train = train[FEATUREAS]
    y_train = train[TARGET]
    y_test = test[TARGET]
    reg = xgb.XGBRegressor(n_estimators = 1000, early_stopping_rounds = 50, learning_rate = 0.01)
    reg.fit(X_train, y_train, eval_set=[(X_train,y_train)], verbose = 0)
    y_hat_gxb = pd.DataFrame(reg.predict(feature_df[FEATUREAS][:-period]), index=feature_df[:-period].index, columns=['y_prediction'])
    all_forecasts = feature_df.drop(FEATUREAS, axis = 1)
    all_forecasts = pd.merge(all_forecasts,y_hat_gxb, on='ds', how='right')
    # Построим графики и посчитаем ошибку
    st.subheader('Forecasting of target for a period of '+str(period)+' days by XGBRegressor')
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=train.index, y=train['y'], name="True value",line_color='deepskyblue'))
    fig.add_trace(go.Scatter(x=test.index, y=test['y'], name="True value of test period",line_color='royalblue'))
    fig.add_trace(go.Scatter(x=y_hat_gxb.index, y=y_hat_gxb['y_prediction'], name="Predicted values",line_color='#e74c3c'))
    st.plotly_chart(fig)
    fig = plt.figure()

    # Считаем и выводим ошибку по последнему году
    y_test = test['y'][:-period]
    y_pred  = all_forecasts['y_prediction'][len(all_forecasts)-period:]
    MAE = round(mean_absolute_error(y_test, y_pred),3)
    MAPE = round(mean_absolute_percentage_error(y_test, y_pred),3)
    st.markdown('**Mean absolute error of XGBoost model for a period of one last year is** '+str(MAE))
    st.markdown('**Mean absolute percentage error of XGBoost model for a period of '+str(period)+' days is** '+str(MAPE*100)+ ' %')

    return all_forecasts




def XGB_predict(feature_df,period):
    """""_summary_""

Args:
    feature_df (_type_):обогащенный дополнительными признаками датасет c целевым признаком и предсказаниями Prohet
    period (_type_): _description_

Returns:
Возвращает датафрейм с индексом ds для конкатинации и столбцом у (содержат NaN в прогнозах)
и столбцом 'yhat_xgb' - предсказаниями модели.
    _type_: _description_
"""
    train = feature_df[:-period]
    test = feature_df[-period:]
    FEATUREAS_df = feature_df.drop(['y', 'yhat'], axis = 1)
    FEATUREAS = FEATUREAS_df.columns
    TARGET = 'y'
    X_train = train[FEATUREAS]
    y_train = train[TARGET]

    reg = xgb.XGBRegressor(n_estimators = 1000, early_stopping_rounds = 50, learning_rate = 0.01)
    reg.fit(X_train, y_train, eval_set=[(X_train,y_train)], verbose = 0)
    y_hat_gxb = pd.DataFrame(reg.predict(feature_df[FEATUREAS]), index=feature_df.index, columns=['y_prediction'])
    all_forecasts = feature_df.drop(FEATUREAS, axis = 1)
    all_forecasts = pd.merge(all_forecasts,y_hat_gxb, on='ds', how='right')
    # Построим графики и посчитаем ошибку
    st.subheader('Forecasting of target for a period of '+str(period)+' days by XGBRegressor')
    n_months = st.slider('Period of visualization in months:', 1,round(len(all_forecasts)/30),0)
    v_period = n_months * 30
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=all_forecasts[-v_period:].index, y=all_forecasts['y'][-v_period:], name="True value",line_color='deepskyblue'))
    fig.add_trace(go.Scatter(x=all_forecasts[-v_period:].index, y=all_forecasts['y_prediction'][-v_period:], name="XGB_prediction",line_color='#e74c3c'))
    st.plotly_chart(fig)
    fig = plt.figure()

    # Считаем и выводим ошибку по последнему году
    # test = all_forecasts['y'][len(all_forecasts)-365-period:len(all_forecasts)-period]
    # y_pred = all_forecasts['y_prediction'][len(all_forecasts)-365-period:len(all_forecasts)-period]
    # MAE = round(mean_absolute_error(test, y_pred),2)
    # MAPE = round(mean_absolute_percentage_error(test, y_pred),2)
    # st.markdown('**Mean absolute error of Prophet model for a period of one last year is** '+str(MAE))
    # st.markdown('**Mean absolute percentage error of Prophet model for a period of one last year is** '+str(MAPE*100)+ ' %')

    FI = pd.DataFrame(data = reg.feature_importances_, index = reg.feature_names_in_, columns = ['Importances'])
    st.subheader('Feature importances by XGBoost Model')
    st.bar_chart(FI.sort_values('Importances',ascending = False)[:10])
    return all_forecasts
