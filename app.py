import json
import os
from random import random
import secrets
from flask import Flask, Response, request, jsonify
import bcrypt
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pypfopt import expected_returns, risk_models, BlackLittermanModel
import yfinance as yf
import pypfopt
from requests import session
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import read_csv
import math
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.layers.core import Dense, Activation, Dropout
from sklearn.model_selection import train_test_split
import tensorflow as tf
import keras
import time  # helper libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pypfopt
import yfinance as yf
import pandas_datareader as pdr
import seaborn as sns
import random

app = Flask(__name__)
con = sqlite3.connect('auth.db', check_same_thread=False)
cur = con.cursor()
etfs = pd.read_csv('data/etfs.csv')

cur.execute("CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY, username TEXT, password TEXT, sessionId TEXT, risk_parameter REAL, portfolio_value REAL)")

dataLocation = 'data/'

very_low = ['GOLDBEES.NS', 'BLV', 'VUSTX', 'VNQ', 'VPU']
low = ['GOLDBEES.NS', 'BLV', 'VNQ', 'VTI','JUST']
medium = ['GOLDBEES.NS', 'BSV', 'VGSH', 'VTI','JUST']
high = ['VO', 'IVOG', 'BSV','VTI','JUST']
very_high = ['VBK', 'VIOG', 'VO', 'IVOG','VTI']

data = {}
for i in range(len(etfs)):
    data[etfs.iloc[i, 0]] = pd.read_csv(dataLocation + etfs.iloc[i, 0] + '.csv', index_col=0)

confidence = {}
for ticker in data:
    if ticker == 'GOLDBEES.NS':
        confidence[ticker] = np.log(data['GOLDBEES.NS']['Close']/data['GOLDBEES.NS']['Close'].shift(-1).dropna()).std()*np.sqrt(252)
        continue
    confidence[ticker] = abs(np.log(data[ticker]['Close'] /
                                   data[ticker]['Close'].shift(-1)).std()*np.sqrt(252))
#    volatility_etfs[ticker] = 1- np.log(data[ticker]['Close'].pct_change()).dropna()

onlyClosePrices = pd.DataFrame()
for ticker in data:
    onlyClosePrices[ticker] = data[ticker]['Close']

S = pypfopt.risk_models.CovarianceShrinkage(onlyClosePrices).ledoit_wolf()

scalar = MinMaxScaler(feature_range=(-1, 1))

for ticker in data:
    data[ticker] = scalar.fit_transform(data[ticker])

trainingData = {}
testData = {}

for ticker in data:
    trainingData[ticker], testData[ticker] = train_test_split(
        data[ticker], test_size=0.8, shuffle=False)


def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), -2]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)


trainX, trainY = {}, {}
testX, testY = {}, {}
timeStep = 20
for ticker in data:
    trainX[ticker], trainY[ticker] = create_dataset(
        trainingData[ticker], timeStep)
    testX[ticker], testY[ticker] = create_dataset(testData[ticker], timeStep)
print("Dataset created")

models = {}
for ticker in data:
    models[ticker] = load_model("models/" + ticker+"_4" + '.h5')
    models[ticker].load_weights("weights/" + ticker + '_4_weights.h5')
print("Models loaded")


netAssets = {}
assets = pd.read_csv('data/assets.csv', index_col=0)
for ticker in data:
    netAssets[ticker] = assets['0'][ticker]
print("Net assets loaded")

viewDict = {}
for ticker in data:
    pred = models[ticker].predict(testX[ticker])
    viewDict[ticker] = (testX[ticker]
                        [-1, 0] -pred[-1, 0])/10/testX[ticker][-1, 0]
print("View dict loaded")
    

print('server started')


@app.route('/signup', methods=['POST'])
def signup():
    res = Response()
    try:
        request
        data = request.form
        username = data['username']
        password = data['password']
        hashed_password = bcrypt.hashpw(
            password.encode('utf-8'), bcrypt.gensalt())
        sessionId = secrets.token_hex(16)
        risk_parameter = data['risk_parameter']
        user = cur.execute(
            "SELECT * FROM users WHERE username = ?", (username,)).fetchone()
        print(user)
        if user is not None:
            res.status_code = 400
            return jsonify({'error': 'User already exists'})
        cur.execute("INSERT INTO users (username, password, sessionId, risk_parameter) VALUES (?, ?, ?, ?)",
                    (username, hashed_password, sessionId, risk_parameter))
        con.commit()
        res.status_code = 200
        return jsonify({'sessionId': sessionId})
    except Exception as e:
        print(e)
        res.status_code = 400
        res.data = 'Error'
    return res


@app.route('/login', methods=['GET'])
def login():
    username = request.args.get('username')
    password = request.args.get('password')
    try:
        cur.execute("SELECT * FROM users WHERE username = ?", (username,))
        user = cur.fetchone()
        if user is None:
            return jsonify({'success': False, 'error': 'User not found'})
        if not bcrypt.checkpw(password.encode('utf-8'), user[2]):
            return jsonify({'success': True, 'sessionId': user[3]})
        sessionId = secrets.token_hex(16)
        cur.execute(
            'UPDATE users SET sessionId = ? WHERE username = ?', (sessionId, username))
        return jsonify({'success': True, 'sessionId': sessionId, 'user': {'username': user[1], 'risk_parameter': user[4], 'portfolio': user[5]}})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/logout')
def logout():
    sessionId = request.args.get('sessionId')
    try:
        cur.execute(
            'UPDATE users SET sessionId = NULL WHERE sessionId = ?', (sessionId,))
        con.commit()
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/get_user')
def get_user():
    sessionId = request.args.get('sessionId')
    try:
        cur.execute('SELECT * FROM users WHERE sessionId = ?', (sessionId,))
        user = cur.fetchone()
        if user is None:
            return jsonify({'success': False, 'error': 'User not found'})
        return jsonify({'success': True, 'username': user[0]})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/get_users')
def get_users():
    try:
        cur.execute('SELECT * FROM users')
        users = cur.fetchall()
        return jsonify({'success': True, 'users': users})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/get_user_by_username')
def get_user_by_username():
    username = request.args.get('username')
    try:
        cur.execute('SELECT * FROM users WHERE username = ?', (username,))
        user = cur.fetchone()
        if user is None:
            return jsonify({'success': False, 'error': 'User not found'})
        return jsonify({'success': True, 'user': user})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/get_prediction')
def get_prediction():
    res = Response()
    try:
        user = request.args.get('user')
        sessionId = request.args.get('sessionId')
        sessionIdDb = cur.execute(
            'SELECT sessionId FROM users WHERE username = ?', (user,)).fetchone()[0]
        risk_parameter = cur.execute(
            'SELECT risk_parameter FROM users WHERE username = ?', (user,)).fetchone()[0]
        if sessionIdDb != sessionId:
            return jsonify({'success': False, 'error': 'Session ID mismatch'})
        temp_netAssets = {}
        temp_viewdict = {}
        temp_confidence = []
        temp_onlyClosePrices = pd.DataFrame()
        if risk_parameter <0.2:
            for ticker in very_low:
                temp_netAssets[ticker] = netAssets[ticker]
                temp_viewdict[ticker] = viewDict[ticker]
                temp_confidence.append(confidence[ticker])
                temp_onlyClosePrices[ticker] = onlyClosePrices[ticker]
        elif risk_parameter <0.4:
            for ticker in low:
                temp_netAssets[ticker] = netAssets[ticker]
                temp_viewdict[ticker] = viewDict[ticker]
                temp_confidence.append(confidence[ticker])
                temp_onlyClosePrices[ticker] = onlyClosePrices[ticker]
        elif risk_parameter <0.6:
            for ticker in medium:
                temp_netAssets[ticker] = netAssets[ticker]
                temp_viewdict[ticker] = viewDict[ticker]
                temp_confidence.append(confidence[ticker])
                temp_onlyClosePrices[ticker] = onlyClosePrices[ticker]
        elif risk_parameter <0.8:
            for ticker in high:
                temp_netAssets[ticker] = netAssets[ticker]
                temp_viewdict[ticker] = viewDict[ticker]
                temp_confidence.append(confidence[ticker])
                temp_onlyClosePrices[ticker] = onlyClosePrices[ticker]
        else :
            for ticker in very_high:
                temp_netAssets[ticker] = netAssets[ticker]
                temp_viewdict[ticker] = viewDict[ticker]
                temp_confidence.append(confidence[ticker])
                temp_onlyClosePrices[ticker] = onlyClosePrices[ticker]
        temp_S = pypfopt.risk_models.CovarianceShrinkage(
            temp_onlyClosePrices).ledoit_wolf()
        delta = pypfopt.black_litterman.market_implied_risk_aversion(
            pd.Series(temp_netAssets))
        bl_confi = pypfopt.BlackLittermanModel(
            temp_S, absolute_views=temp_viewdict, rho=delta, view_confidences=temp_confidence, risk_aversion=risk_parameter)
        bl_return_confi = bl_confi.bl_returns()
        bl_return_confi.name = 'BL Returns with Confidence'
        S_bl_confi = bl_confi.bl_cov()
        ef = pypfopt.EfficientFrontier(
            bl_return_confi, S_bl_confi, weight_bounds=(0.01, 0.3))
        weights = ef.min_volatility()
        ef.portfolio_performance(verbose=True)
        res.status_code = 200
        return jsonify({'success': True, 'weights': json.dumps(weights)})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/update_risk_parameter')
def update_risk_parameter():
    newRiskParameter = request.args.get('risk_parameter')
    sessionId = request.args.get('sessionId')
    user = request.args.get('username')
    try:
        cur.execute(
            "UPDATE users SET risk_parameter = ? WHERE sessionId = ? OR username = ?", (newRiskParameter, sessionId, user))
        con.commit()
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/get_risk_parameter')
def getRiskParameter():
    try:
        riskParameter = cur.execute(
            "SELECT risk_parameter FROM users WHERE sessionId = ?", (sessionId,)).fetchone()[0]
        return jsonify({'success': True, 'risk_parameter': riskParameter})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/update_portfolio_value')
def updatePortfolioValue():
    newPortfolioValue = request.args.get('portfolio_value')
    sessionId = request.args.get('sessionId')
    try:
        temp = cur.execute(
            "UPDATE users SET portfolio_value = ? WHERE sessionId = ?", (newPortfolioValue, sessionId))
        return jsonify({'success': True, 'portfolio_value': newPortfolioValue})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/restart_server')
def restart_server():
    app.stop()
    return jsonify({'success': True})
