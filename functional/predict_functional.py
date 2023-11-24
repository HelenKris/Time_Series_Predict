import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error,mean_absolute_percentage_error,r2_score
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# os.environ['CMDSTAN'] = "C:/Users/38/anaconda3/Library/bin/cmdstan"

def plot_predict_data(data, v_period, col_name_target, col_name_predict):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index[-v_period:], y=data[col_name_target][-v_period:], name="True value",line_color='royalblue'))
    fig.add_trace(go.Scatter(x=data.index[-v_period:], y=data[col_name_predict][-v_period:], name="Predictions",line_color='#e74c3c'))
    st.plotly_chart(fig)
    return fig

def prophet(data_pred,period):
    """""_summary_""
    Returns:
    Возвращает датафрейм с индексом ds для конкатинации и столбцом у (содержат NaN в прогнозах)
    и столбцом 'yhat' - предсказаниями модели.
        _type_: _description_
    """
    m = Prophet(yearly_seasonality=True)
    m.fit(data_pred)
    future = m.make_future_dataframe(periods=period)
    forecast = m.predict(future)
    st.header('Model: Prophet. Forecasting period: '+str(period)+' days')
    stock_price_forecast = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    all_forecasts = pd.merge(data_pred,stock_price_forecast, on='ds', how='right')
    all_forecasts = all_forecasts.set_index('ds')
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=stock_price_forecast.ds, y=all_forecasts['y'], name="True value",line_color='deepskyblue'))
    fig.add_trace(go.Scatter(x=stock_price_forecast.ds, y=all_forecasts['yhat_lower'], name="Predicted lower bend",line_color='#ffb000'))
    fig.add_trace(go.Scatter(x=stock_price_forecast.ds, y=all_forecasts['yhat'], name="Predicted value",line_color='#e74c3c'))
    fig.add_trace(go.Scatter(x=stock_price_forecast.ds, y=all_forecasts['yhat_upper'], name="Predicted upper bend",line_color='#ffb000'))
    fig.layout.update(xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)
    all_forecasts = pd.DataFrame(all_forecasts[['y','yhat']])
    return all_forecasts

def prophet_select(data_pred,period):
    """""Функция для расчета ошибки модели Prophet на указанный прогнозный период period""
    Args:
    Принимает датафрейм после предобработки функцией preprocessing()
    Returns:
    Возвращает словарь с ошибками модели MAE, MAPE, MSE
    """
    m = Prophet()
    train_df = data_pred[:-period]
    m.fit(train_df)
    future = m.make_future_dataframe(periods=period)
    forecast = m.predict(future)

    st.header('Model: Prophet. Test period: '+str(period)+' days')
    stock_price_forecast = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    all_forecasts = pd.merge(train_df,stock_price_forecast, on='ds', how='right')
    all_forecasts = all_forecasts.set_index('ds')
    test = data_pred['y'][len(data_pred)- period:]
    y_pred = stock_price_forecast['yhat'][len(data_pred)- period:]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data_pred.ds, y=data_pred['y'], name="True value",line_color='deepskyblue'))
    fig.add_trace(go.Scatter(x=data_pred['ds'][len(data_pred)- period:], y=data_pred['y'][len(data_pred)- period:], name="True value of test period",line_color='royalblue'))
    fig.add_trace(go.Scatter(x=stock_price_forecast.ds, y=stock_price_forecast['yhat'], name="Predicted values",line_color='#e74c3c'))
    fig.layout.update(xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

    # Логика оценки: мы оцениваем модель по такому же интервалу, по которому прогназируем. Т.е. если хотим прогнозировать на год, то
    # оцениваем модель по средней ошибке за предыдущий год, если прогнозируем на неделю, то оцениваем тоже по последней неделе.
    MAE = round(mean_absolute_error(test, y_pred),3)
    MAPE = round(mean_absolute_percentage_error(test, y_pred),3)
    MSE = round(mean_squared_error(test, y_pred),3)
    prophet_dict = {'MAE': MAE, 'MAPE': MAPE, 'MSE': MSE}
    st.markdown('**Mean absolute error of Prophet model for a period of '+str(period)+' days is** '+str(MAE))
    st.markdown('**Mean absolute percentage error of Prophet model for a period of '+str(period)+' days is** '+str(MAPE*100)+ '%')
    st.markdown('**Mean squared error of Prophet model for a period of '+str(period)+' days is** '+str(MSE))
    return prophet_dict

def create_features(df,period):
    """В функцию передается all_forecasts = prophet(data_pred,period)
    Функция возвращает датафрейм с колонками доп.признаков и индексом в datetime для последующей конкатенации
    с целевым признаком и предсказаниями prophet тоже
    """
    # Extract time-based features
    df['week_of_year'] = df.index.week
    df['dayofweek'] = df.index.dayofweek
    df['day'] = df.index.day
    df['month'] = df.index.month
    df['is_weekend'] = df.index.weekday.isin([5,6])*1
    df['sin_dayofweek'] = np.sin(2*np.pi*(df['dayofweek']-0)/7)
    df['cos_dayofweek'] = np.cos(2*np.pi*(df['dayofweek']-0)/7)
    df['sin_month'] = np.sin(2*np.pi*(df['month']-0)/7)
    df['cos_month'] = np.cos(2*np.pi*(df['month']-0)/7)
    df['sin_week_of_year'] = np.sin(2*np.pi*(df['week_of_year']-0)/7)
    df['cos_week_of_year'] = np.cos(2*np.pi*(df['week_of_year']-0)/7)
    df["time_idx"] = df.index.year * 12 + df.index.month
    df["time_idx"] -= df["time_idx"].min()
    # Lagged features
    lag_list = [period]
    for i in lag_list:
        df["lag_{}".format(i)] = df.y.shift(i)

    # Presidential term cycle feature
    df['year_index'] = df.index.year % 4  # Remainder of current year divided by 4
    return df


def XGB_select(feature_df,period):
    """""Функция для расчета ошибки модели XGBoost на указанный прогнозный период period""
    Args:
    Принимает датафрейм после предобработки функцией preprocessing()
    Returns:
    Возвращает словарь с ошибками модели MAE, MAPE, MSE
"""
    train = feature_df[:-2*period]
    test = feature_df[len(feature_df)-2*period:]
    FEATUREAS = list(feature_df.columns)
    columns_to_delete = ['y', 'yhat']
    FEATUREAS = [x for x in FEATUREAS if x not in columns_to_delete]
    X_train = train[FEATUREAS]
    y_train = train['y']
    y_test = test['y']
    reg = xgb.XGBRegressor(n_estimators = 1000, early_stopping_rounds = 50, learning_rate = 0.01)
    reg.fit(X_train, y_train, eval_set=[(X_train,y_train)], verbose = 0)
    y_hat_gxb = pd.DataFrame(reg.predict(feature_df[FEATUREAS][:-period]), index=feature_df[:-period].index, columns=['y_prediction'])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=train.index, y=train['y'], name="True value",line_color='deepskyblue'))
    fig.add_trace(go.Scatter(x=test.index, y=test['y'], name="True value of test period",line_color='royalblue'))
    fig.add_trace(go.Scatter(x=y_hat_gxb.index, y=y_hat_gxb['y_prediction'], name="Predicted values",line_color='#e74c3c'))
    fig.layout.update(xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)
    # Считаем и выводим ошибку
    y_test = test['y'][:-period]
    y_pred  = y_hat_gxb['y_prediction'][len(y_hat_gxb)-period:]
    MAE = round(mean_absolute_error(y_test, y_pred),3)
    MAPE = round(mean_absolute_percentage_error(y_test, y_pred),3)
    MSE = round(mean_squared_error(y_test, y_pred),3)
    xgb_dict = {'MAE': MAE, 'MAPE': MAPE, 'MSE': MSE}
    st.markdown('**Mean absolute error of XGBoost model for a period of '+str(period)+' days is** '+str(MAE))
    st.markdown('**Mean absolute percentage error of XGBoost model for a period of '+str(period)+' days is** '+str(MAPE*100)+ '%')
    st.markdown('**Mean squared error of XGBoost model for a period of '+str(period)+' days is** '+str(MSE))
    return xgb_dict

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

def getdata(data, data_x, data_y, lookback,start_shift):
    X, Y = [], []
    for i in range(len(data) - lookback + 1 - start_shift):
        X.append(data_x[i:i+lookback])
        Y.append(data_y[i+lookback-1])
    return np.array(X), np.array(Y).reshape(-1, 1)


def GRU_predict(feature_df, period, num_epochs):
    # Rest of the import statements and initial data processing
    FEATUREAS = list(feature_df.columns)
    columns_to_delete = ['y', 'yhat']
    FEATUREAS = [x for x in FEATUREAS if x not in columns_to_delete]
    y_columns = ["y"]
    start_shift = period
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    x_scaled = scaler_x.fit_transform(feature_df[FEATUREAS][start_shift:])
    y_scaled = scaler_y.fit_transform(feature_df[y_columns][start_shift:])
    hidden_size = 10
    x, y = getdata(feature_df, x_scaled, y_scaled, hidden_size,start_shift)
    x = x.reshape(x.shape[0], x.shape[1], x.shape[2])
    # Convert the data to PyTorch tensors and move them to the designated device
    x = torch.from_numpy(x.astype(np.float32)).to(device)
    y = torch.from_numpy(y.astype(np.float32)).to(device)

    # Define the GRU model
    class GRUModel(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super(GRUModel, self).__init__()
            self.hidden_size = hidden_size
            self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
            self.fc = nn.Linear(hidden_size, output_size)

        def forward(self, x):
            out, _ = self.gru(x)
            out = self.fc(out[:, -1, :])  # Use the output of the last time step
            return out

    input_size = x.shape[2]
    hidden_size = 10  # Number of hidden units
    output_size = 1  # Single output
    model = GRUModel(input_size, hidden_size, output_size).to(device)

    # Define the loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Training loop
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        output = model(x[:-period])
        loss = criterion(output, y[:-period])
        loss.backward()
        optimizer.step()

    # Inference and post-processing
    with torch.no_grad():
        y_hat = pd.Series(scaler_y.inverse_transform(model(x[:])).squeeze())

    index=pd.Series(feature_df[period+hidden_size-1:].index)
    GRU_forecast = pd.DataFrame(pd.concat([index, y_hat], axis=1)).set_index('ds')
    GRU_forecast.rename(columns={0: 'RNN_prediction'}, inplace=True)
    plot_GRU_forecast = pd.merge(feature_df,GRU_forecast, on='ds', how='right')

    st.subheader('Forecasting of target for a period of '+str(period)+' days by recurrent neural networks with GRU')
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=plot_GRU_forecast.index, y=plot_GRU_forecast['y'], name="True value",line_color='deepskyblue'))
    fig.add_trace(go.Scatter(x=plot_GRU_forecast.index, y=plot_GRU_forecast['RNN_prediction'], name="['GRU prediction']",line_color='#e74c3c'))
    fig.layout.update(xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)
    return GRU_forecast

def GRU_select(feature_df,period, num_epochs):
    feature_df = feature_df[:-period]
    FEATUREAS = list(feature_df.columns)
    columns_to_delete = ['y', 'yhat']
    FEATUREAS = [x for x in FEATUREAS if x not in columns_to_delete]
    y_columns = ["y"]
    start_shift = period
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    x_scaled = scaler_x.fit_transform(feature_df[FEATUREAS][start_shift:])
    y_scaled = scaler_y.fit_transform(feature_df[y_columns][start_shift:])
    hidden_size = 10
    x, y = getdata(feature_df, x_scaled, y_scaled, hidden_size,start_shift)
    x = x.reshape(x.shape[0], x.shape[1], x.shape[2])
    # Convert the data to PyTorch tensors and move them to the designated device
    x = torch.from_numpy(x.astype(np.float32)).to(device)
    y = torch.from_numpy(y.astype(np.float32)).to(device)

    # Define the GRU model
    class GRUModel(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super(GRUModel, self).__init__()
            self.hidden_size = hidden_size
            self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
            self.fc = nn.Linear(hidden_size, output_size)

        def forward(self, x):
            out, _ = self.gru(x)
            out = self.fc(out[:, -1, :])  # Use the output of the last time step
            return out

    input_size = x.shape[2]
    hidden_size = 10  # Number of hidden units
    output_size = 1  # Single output
    model = GRUModel(input_size, hidden_size, output_size).to(device)

    # Define the loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Training loop
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        output = model(x[:-period])
        loss = criterion(output, y[:-period])
        loss.backward()
        optimizer.step()

    # Inference and post-processing
    with torch.no_grad():
        y_hat = pd.Series(scaler_y.inverse_transform(model(x[:])).squeeze())

    index=pd.Series(feature_df[period+hidden_size-1:].index)
    GRU_forecast = pd.DataFrame(pd.concat([index, y_hat], axis=1)).set_index('ds')
    GRU_forecast.rename(columns={0: 'RNN_prediction'}, inplace=True)
    plot_GRU_forecast = pd.merge(feature_df,GRU_forecast, on='ds', how='right')
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=feature_df[:-2*period].index, y=feature_df[:-2*period]['y'], name="True value",line_color='deepskyblue'))
    fig.add_trace(go.Scatter(x=feature_df[len(feature_df)-2*period:].index, y=feature_df[len(feature_df)-2*period:]['y'], name="True value of test period",line_color='royalblue'))
    fig.add_trace(go.Scatter(x=plot_GRU_forecast.index, y=plot_GRU_forecast['RNN_prediction'], name="['GRU prediction']",line_color='#e74c3c'))
    fig.layout.update(xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)
    test = plot_GRU_forecast[-period:]['y']
    y_hat_RNN = plot_GRU_forecast[-period:]['RNN_prediction']
    MAPE= round(mean_absolute_percentage_error(test[-period:], y_hat_RNN[-period:]),3)
    MAE = round(mean_absolute_error(test[-period:], y_hat_RNN[-period:]),3)
    MSE = round(mean_squared_error(test[-period:], y_hat_RNN[-period:]),3)
    st.markdown('**Mean absolute error of GRU model for a period of '+str(period)+' days is** '+str(MAE))
    st.markdown('**Mean absolute percentage error of GRU model for a period of '+str(period)+' days is** '+str(MAPE*100)+ '%')
    st.markdown('**Mean squared error of GRU model for a period of '+str(period)+' days is** '+str(MSE))
    gru_dict = {'MAE': MAE, 'MAPE': MAPE, 'MSE': MSE}
    return gru_dict
