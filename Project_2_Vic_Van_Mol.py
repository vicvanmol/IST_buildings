# -*- coding: utf-8 -*-
"""
@author: Vic Van Mol
"""

import dash
from dash import html
from dash import dcc
from dash.dependencies import Input, Output
import pandas as pd
import plotly.express as px
import pickle
from sklearn import  metrics
import numpy as np
from xgboost import XGBRegressor

# Define CSS style
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

# Load data
df = pd.read_csv('data_2019.csv')
df['Date'] = pd.to_datetime(df['Date'])  # create a new column 'data time' of datetime type
# df = df.set_index('Date') # make 'datetime' into index
# df.rename(columns = {'Power-1':'power', 'Day week':'day'}, inplace = True)
df2 = df.iloc[:, 2:8]
X2 = df2.values
fig = px.line(df, x="Date", y=df.columns[1:7])


df_forecast_real = df.iloc[:, 1]
y2 = df_forecast_real.values

# Load and run models

with open('NN_model1.pkl', 'rb') as file:
    LR_model2 = pickle.load(file)

y2_pred_NN = LR_model2.predict(X2)



# Load RF model
with open('XGB_model1.pkl', 'rb') as file:
    RF_model2 = pickle.load(file)

y2_pred_XGB = RF_model2.predict(X2)

results = [y2, y2_pred_NN, y2_pred_XGB]
fig2 = px.line(df, x= "Date",y=results)
fig2.data[0].name = "Real data"
fig2.data[1].name = "Neural Networks"
fig2.data[2].name = "Extreme gradient boosting"



MAE_NN=metrics.mean_absolute_error(y2, y2_pred_NN)
MSE_NN=metrics.mean_squared_error(y2, y2_pred_NN)
RMSE_NN= np.sqrt(metrics.mean_squared_error(y2, y2_pred_NN))
cvRMSE_NN=RMSE_NN/np.mean(y2)

MAE_XGB=metrics.mean_absolute_error(y2, y2_pred_XGB)
MSE_XGB=metrics.mean_squared_error(y2, y2_pred_XGB)
RMSE_XGB= np.sqrt(metrics.mean_squared_error(y2, y2_pred_XGB))
cvRMSE_XGB=RMSE_XGB/np.mean(y2)

metrics_list = [['Neural networks', MAE_NN,MSE_NN,RMSE_NN, cvRMSE_NN],['Extreme Gradient Boosting',MAE_XGB,MSE_XGB,RMSE_XGB, cvRMSE_XGB] ]

# Calling DataFrame constructor on list
df_metrics = pd.DataFrame(metrics_list)

df_metrics.columns = ["Forecasting methods", "MAE", "MSE", "RMSE", "cvRMSE"]
#df_metrics = df_metrics.set_index('Forecasting methods')


# Define auxiliary functions
def generate_table(dataframe, max_rows=10):
    return html.Table([
        html.Thead(
            html.Tr([html.Th(col) for col in dataframe.columns])
        ),
        html.Tbody([
            html.Tr([
                html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
            ]) for i in range(min(len(dataframe), max_rows))
        ])
    ])


app = dash.Dash(__name__, external_stylesheets=external_stylesheets, suppress_callback_exceptions=True)

app.layout = html.Div([
    html.H2('IST Energy Forecast tool South Tower (kWh)'),
    html.H5('Vic Van Mol'),
    dcc.Tabs(id='tabs', value='tab-1', children=[
        dcc.Tab(label='Raw Data', value='tab-1'),
        dcc.Tab(label='Forecast', value='tab-2'),
    ]),
    html.Div(id='tabs-content')
])
available_variables = ['Power_kW', 'temp_C', 'solarRad_W/m2', 'Day_of_week', 'Hour_of_day', 'Power-1']


@app.callback(Output('tabs-content', 'children'),
              Input('tabs', 'value'))
def render_content(tab):
    # if tab == 'tab-1':
    #     return html.Div([
    #         html.H3('IST Raw Data'),
    #         dcc.Graph(
    #             id='yearly-data',
    #             figure=fig,
    #         ),
    #
    #     ])
    if tab == 'tab-1':
        return html.Div([
            dcc.Dropdown(
                id='menu',
                options=[{'label': i, 'value': i} for i in available_variables],
                value='Power_kW'
            ),
            html.Div([
                dcc.Graph(id='yearly-data')
            ])
        ])
    elif tab == 'tab-2':
        return html.Div([
            #html.H3('IST Electricity Forecast South Tower (kWh)'),
            dcc.Graph(
                id='yearly-dat',
                figure=fig2,
            ),
                      generate_table(df_metrics)
        ])

@app.callback(
    dash.dependencies.Output('yearly-data', 'figure'),
    [dash.dependencies.Input('menu', 'value')])

def update_graph(value):
    df_2 = df.loc[:, value]
    return create_graph(df_2)


def create_graph(df_2):
    return {
        'data': [
            {'x': df['Date'], 'y': df_2, 'type': 'line', 'name': 'Power (kW)'},
        ],
        # 'layout': {
        #     'title': ''
        # }
    }



if __name__ == '__main__':
    app.run_server(debug=True)