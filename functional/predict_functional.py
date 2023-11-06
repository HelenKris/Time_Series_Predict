import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error,mean_absolute_percentage_error
import xgboost as xgb
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, GRU


# os.environ['CMDSTAN'] = "C:/Users/38/anaconda3/Library/bin/cmdstan"

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

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=stock_price_forecast.ds, y=all_forecasts['y'], name="Adj Close True",line_color='royalblue'))
    fig.add_trace(go.Scatter(x=stock_price_forecast.ds, y=all_forecasts['yhat_lower'], name="yhat_lower",line_color='#e74c3c'))
    fig.add_trace(go.Scatter(x=stock_price_forecast.ds, y=all_forecasts['yhat'], name="yhat",line_color='deepskyblue'))
    fig.add_trace(go.Scatter(x=stock_price_forecast.ds, y=all_forecasts['yhat_upper'], name="yhat_upper",line_color='#e74c3c'))
    fig.layout.update(xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)
    if st.checkbox('Show forecast data'):
        st.subheader('forecast data')
        st.write(forecast)
    all_forecasts = pd.DataFrame(all_forecasts[['y','yhat']])
    # Prophet_forecast = pd.DataFrame(all_forecasts['yhat'], index=all_forecasts.index)
    return all_forecasts

# def prophet_select(data_pred,period):
#     """""_summary_""

#     Args:
#         data_pred (_type_): _description_
#         period (_type_): _description_

#     Returns:
#     Возвращает датафрейм с индексом ds для конкатинации и столбцом у (содержат NaN в прогнозах)
#     и столбцом 'yhat' - предсказаниями модели.
#         _type_: _description_
#     """
#     m = Prophet()
#     train_df = data_pred[:-period]
#     m.fit(train_df)
#     future = m.make_future_dataframe(periods=period)
#     forecast = m.predict(future)

#     st.header('Model: Prophet. Test period: '+str(period)+' days')
#     stock_price_forecast = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
#     all_forecasts = pd.merge(train_df,stock_price_forecast, on='ds', how='right')
#     all_forecasts = all_forecasts.set_index('ds')
#     test = data_pred['y'][len(data_pred)- period:]
#     y_pred = stock_price_forecast['yhat'][len(data_pred)- period:]

#     fig = go.Figure()
#     fig.add_trace(go.Scatter(x=data_pred.ds, y=data_pred['y'], name="True value",line_color='deepskyblue'))
#     fig.add_trace(go.Scatter(x=data_pred['ds'][len(data_pred)- period:], y=data_pred['y'][len(data_pred)- period:], name="True value of test period",line_color='royalblue'))
#     fig.add_trace(go.Scatter(x=stock_price_forecast.ds, y=stock_price_forecast['yhat'], name="Predicted values",line_color='#e74c3c'))
#     fig.layout.update(xaxis_rangeslider_visible=True)
#     st.plotly_chart(fig)

#     # Логика оценки: мы оцениваем модель по такому же интервалу, по которому прогназируем. Т.е. если хотим прогнозировать на год, то
#     # оцениваем модель по средней ошибке за предыдущий год, если прогнозируем на неделю, то оцениваем тоже по последней неделе.
#     MAE = round(mean_absolute_error(test, y_pred),3)
#     MAPE = round(mean_absolute_percentage_error(test, y_pred),3)
#     MSE = round(mean_squared_error(test, y_pred),3)
#     st.markdown('**Mean absolute error of Prophet model for a period of '+str(period)+' days is** '+str(MAE))
#     st.markdown('**Mean absolute percentage error of Prophet model for a period of '+str(period)+' days is** '+str(MAPE*100)+ ' %')
#     st.markdown('**Mean squared error of Prophet model for a period of '+str(period)+' days is** '+str(MSE))
#     return MAE, MAPE, MSE


def create_features(df):
    """В функцию передается all_forecasts = prophet(data_pred,period)
    Функция возвращает датафрейм с колонками доп.признаков и индексом в datetime для последующей конкатенации
    с целевым признаком и предсказаниями prophet тоже
    """
    # Extract time-based features
    df['week'] = df.index.weekday
    df['day'] = df.index.day
    df['month'] = df.index.month
    df['is_weekend'] = df.index.weekday.isin([5,6])*1
    # Lagged features
    lag_list = [365]
    # lag_list = [2,5,14,30,60,90,365]
    for i in lag_list:
        df["lag_{}".format(i)] = df.y.shift(i)

    # Presidential term cycle feature
    df['presidential_term_cycle'] = df.index.year % 4  # Remainder of current year divided by 4
    return df


# def XGB_select(feature_df,period):
#     """""_summary_""

# Args:
#     feature_df (_type_):обогащенный дополнительными признаками датасет c целевым признаком и предсказаниями Prohet
#     period (_type_): n_months, указанный пользователем в st.slider('Months of prediction:'), умноженный на 30

# Returns:
#     MAE, MAPE, MSE
#     _type_: _description_
# """

#     train = feature_df[:-2*period]
#     test = feature_df[len(feature_df)-2*period:]
#     FEATUREAS = list(feature_df.columns)
#     columns_to_delete = ['y', 'yhat']
#     FEATUREAS = [x for x in FEATUREAS if x not in columns_to_delete]
#     X_train = train[FEATUREAS]
#     y_train = train['y']
#     y_test = test['y']
#     reg = xgb.XGBRegressor(n_estimators = 1000, early_stopping_rounds = 50, learning_rate = 0.01)
#     reg.fit(X_train, y_train, eval_set=[(X_train,y_train)], verbose = 0)
#     y_hat_gxb = pd.DataFrame(reg.predict(feature_df[FEATUREAS][:-period]), index=feature_df[:-period].index, columns=['y_prediction'])
#     # all_forecasts = feature_df.drop(FEATUREAS, axis = 1)
#     # all_forecasts = pd.merge(all_forecasts,y_hat_gxb, on='ds', how='right')
#     # Построим графики и посчитаем ошибку
#     st.subheader('Forecasting of target for a period of '+str(period)+' days by XGBRegressor')
#     fig = go.Figure()
#     fig.add_trace(go.Scatter(x=train.index, y=train['y'], name="True value",line_color='deepskyblue'))
#     fig.add_trace(go.Scatter(x=test.index, y=test['y'], name="True value of test period",line_color='royalblue'))
#     fig.add_trace(go.Scatter(x=y_hat_gxb.index, y=y_hat_gxb['y_prediction'], name="Predicted values",line_color='#e74c3c'))
#     fig.layout.update(xaxis_rangeslider_visible=True)
#     st.plotly_chart(fig)
#     # Считаем и выводим ошибку
#     y_test = test['y'][:-period]
#     # y_pred  = all_forecasts['y_prediction'][len(all_forecasts)-period:]
#     y_pred  = y_hat_gxb['y_prediction'][len(y_hat_gxb)-period:]
#     MAE = round(mean_absolute_error(y_test, y_pred),3)
#     MAPE = round(mean_absolute_percentage_error(y_test, y_pred),3)
#     MSE = round(mean_squared_error(y_test, y_pred),3)
#     st.markdown('**Mean absolute error of XGBoost model for a period of '+str(period)+' days is** '+str(MAE))
#     st.markdown('**Mean absolute percentage error of XGBoost model for a period of '+str(period)+' days is** '+str(MAPE*100)+ ' %')
#     st.markdown('**Mean squared error of XGBoost model for a period of '+str(period)+' days is** '+str(MSE))
#     return MAE, MAPE, MSE

def XGB_predict(feature_df,period):
    """""_summary_""

Args:
    feature_df (_type_):обогащенный дополнительными признаками датасет c целевым признаком и предсказаниями Prohet
    period (_type_): _description_

Returns:

    _type_: _description_
"""
    train = feature_df[:-period]
    FEATUREAS = list(feature_df.columns)
    columns_to_delete = ['y', 'yhat']
    FEATUREAS = [x for x in FEATUREAS if x not in columns_to_delete]
    X_train = train[FEATUREAS]
    y_train = train['y']
    reg = xgb.XGBRegressor(n_estimators = 1000, early_stopping_rounds = 50, learning_rate = 0.01)
    reg.fit(X_train, y_train, eval_set=[(X_train,y_train)], verbose = 0)
    y_hat_gxb = pd.DataFrame(reg.predict(feature_df[FEATUREAS]), index=feature_df.index, columns=['y_prediction'])
    st.subheader('Forecasting of target for a period of '+str(period)+' days by XGBRegressor')
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=feature_df.index, y=feature_df['y'], name="True value",line_color='deepskyblue'))
    fig.add_trace(go.Scatter(x=y_hat_gxb.index, y=y_hat_gxb['y_prediction'], name="XGB_prediction",line_color='#e74c3c'))
    fig.layout.update(xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)
    FI = pd.DataFrame(data = reg.feature_importances_, index = reg.feature_names_in_, columns = ['Importances'])
    st.subheader('Feature importances by XGBoost Model')
    st.bar_chart(FI.sort_values('Importances',ascending = False)[:10])
    return y_hat_gxb

def getdata(data, data_x, data_y, lookback):
    start_shift = 365
    X, Y = [], []
    for i in range(len(data) - lookback + 1 - start_shift):
        X.append(data_x[i:i+lookback])
        Y.append(data_y[i+lookback-1])
    return np.array(X), np.array(Y).reshape(-1, 1)

def GRU_predict(feature_df,period, num_epochs):
    st.write(feature_df)
    # train = feature_df[:-period]
    FEATUREAS = list(feature_df.columns)
    columns_to_delete = ['y', 'yhat']
    FEATUREAS = [x for x in FEATUREAS if x not in columns_to_delete]
    y_columns = ["y"]
    # X_train = train[FEATUREAS]
    # y_train = train['y']
    # reg = xgb.XGBRegressor(n_estimators = 1000, early_stopping_rounds = 50, learning_rate = 0.01)
    # reg.fit(X_train, y_train, eval_set=[(X_train,y_train)], verbose = 0)
    # y_hat_gxb = pd.DataFrame(reg.predict(feature_df[FEATUREAS]), index=feature_df.index, columns=['y_prediction'])
    start_shift = 365
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    x_scaled = scaler_x.fit_transform(feature_df[FEATUREAS][start_shift:])
    y_scaled = scaler_y.fit_transform(feature_df[y_columns][start_shift:])
    st.write(len(x_scaled))
    lookback = 10
    x, y = getdata(feature_df, x_scaled, y_scaled, lookback)
    x = x.reshape(x.shape[0], x.shape[1], x.shape[2])
    st.write(x.shape)
    st.write(y.shape)
    batch_size=1
    model = Sequential()
    model.add(GRU(10, stateful=True, batch_input_shape=(batch_size, x.shape[1], x.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error',optimizer='adam')
    model.fit(x[:-365], y[:-365], epochs=num_epochs, batch_size=batch_size)
    y_test = pd.DataFrame(scaler_y.inverse_transform(y[:]))
    forecast = pd.DataFrame(scaler_y.inverse_transform(model.predict(x[:], batch_size=batch_size)))
    y_hat = pd.Series(forecast.squeeze())
    index=pd.Series(feature_df[period+lookback-1:].index)
    st.write(len(y_hat))
    st.write(y_hat)
    # ValueError: Shape of passed values is (2087, 1), indices imply (2452, 1)
    GRU_forecast = pd.DataFrame(pd.concat([index, y_hat], axis=1)).set_index('ds')
    GRU_forecast.rename(columns={0: 'RNN_prediction'}, inplace=True)
    # RNN_forecast = pd.DataFrame(y_hat,index=feature_df[period+lookback-1:].index,columns=['RNN_prediction'])
    # RNN_forecast = pd.DataFrame(y_hat,index=feature_df[period+lookback-1:].index,columns=['RNN_prediction'])
    # index=feature_df[period+lookback-1:].index
    # st.write(index)
    st.write(GRU_forecast)
    # GRU_forecast = pd.DataFrame(y_hat,index=feature_df[period+lookback-1:].index,columns=['RNN_prediction'])
    plot_GRU_forecast = pd.merge(feature_df,GRU_forecast, on='ds', how='right')
    st.write(plot_GRU_forecast)
    st.subheader('Forecasting of target for a period of '+str(period)+' days by recurrent neural networks with GRU')
    fig = go.Figure()
    # fig.add_trace(go.Scatter(x=y_test.index, y=y_test, name="['GRU prediction']",line_color='deepskyblue'))
    # fig.add_trace(go.Scatter(x=forecast.index, y=forecast, name="True value",line_color='#e74c3c'))
    fig.add_trace(go.Scatter(x=plot_GRU_forecast.index, y=plot_GRU_forecast['RNN_prediction'], name="['GRU prediction']",line_color='deepskyblue'))
    fig.add_trace(go.Scatter(x=plot_GRU_forecast.index, y=plot_GRU_forecast['y'], name="True value",line_color='#e74c3c'))
    fig.layout.update(xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)
    # st.write(GRU_forecast)
    st.write(plot_GRU_forecast)
    return GRU_forecast
