import pandas as pd
from warnings import filterwarnings
filterwarnings("ignore")

import plotly.graph_objects as go

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pmdarima import auto_arima
import joblib
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


def load_data(data):
    raw_data = pd.read_excel(data)  # keep the raw data as original
    return raw_data

def new_label(df):
    df["UrClimate Hız_Sıcaklık"] = df["UrClimate Rüzgar Hızı"] * df["UrClimate Sıcaklık"]
    df["UrClimate Hız_Yön"] = df["UrClimate Rüzgar Hızı"] * df["UrClimate Rüzgar Yönü"]
    df["UrClimate Hız_Nem"] = df["UrClimate Rüzgar Hızı"] * df["UrClimate Nem"]

    df["UrClimate Sıcaklık_Yön"] = df["UrClimate Sıcaklık"] * df["UrClimate Rüzgar Yönü"]
    df["UrClimate Sıcaklık_Nem"] = df["UrClimate Sıcaklık"] * df["UrClimate Nem"]

    df["UrClimate Yön_Nem"] = df["UrClimate Rüzgar Yönü"] * df["UrClimate Nem"]
    return df

def lag_features(df):
    df = df.set_index("Zaman")
    
    lag_features= df.columns
    window1=3
    window2=7
    
    for feature in lag_features:
        df[feature+'rolling_mean_3'] = df[feature].rolling(window=window1).mean()
        df[feature+'rolling_mean_7'] = df[feature].rolling(window=window2).mean()
    
    for feature in lag_features:
        df[feature+'rolling_std_3'] = df[feature].rolling(window=window1).std()
        df[feature+'rolling_std_7'] = df[feature].rolling(window=window2).std()
        
    df.dropna(inplace=True)    
    return df

def split(raw_data, df):
    training_data=df[:int(raw_data.shape[0]*.70)] # data x 0.70 --> training
    test_data=df[int(raw_data.shape[0]*.70):]      # data x 0.30 --> test
    return training_data, test_data

def feature():
    features=['Ölçüm Rüzgar Hızırolling_mean_3', 'Ölçüm Rüzgar Hızırolling_mean_7',
              'UrClimate Rüzgar Hızırolling_mean_3', 'UrClimate Rüzgar Hızırolling_mean_7',
              'UrClimate Hız_Nemrolling_mean_3', 'UrClimate Hız_Nemrolling_mean_7',
              'UrClimate Hız_Sıcaklıkrolling_mean_3', 'UrClimate Hız_Sıcaklıkrolling_mean_7',
              'UrClimate Hız_Yönrolling_mean_3', 'UrClimate Hız_Yönrolling_mean_7',
              'UrClimate Nemrolling_mean_3', 'UrClimate Hız_Nemrolling_std_7',
              'UrClimate Hız_Sıcaklıkrolling_std_7', 'UrClimate Nemrolling_mean_7',
              'UrClimate Sıcaklıkrolling_std_7','UrClimate Sıcaklık_Yönrolling_mean_3',
              'UrClimate Sıcaklık_Yönrolling_mean_7','UrClimate Sıcaklıkrolling_mean_7', 'UrClimate Sıcaklıkrolling_mean_3']
    return features


def arima_model(df, training_data, test_data):
    features = feature()
    model = auto_arima(y=training_data['Ölçüm Rüzgar Hızı'],exogenous=training_data[features],
                   d=0,
                   start_p=1,
                   start_q=1,
                   max_p=5,
                   max_q=5,
                   seasonal=True,
                   m=4,
                   D=1,
                   stepwise=True,
                   trace=True)
    
    model.fit(training_data['Ölçüm Rüzgar Hızı'], training_data[features])
    
    return model

def main(data, output):
    features = feature()
    raw_data = load_data(data)   
    df = raw_data.copy()
    df=new_label(df)
    df= lag_features(df)
    training_data, test_data = split(raw_data, df)

    model = arima_model(df, training_data, test_data)
    joblib.dump(model, 'alkazar.pkl')

    forecast = model.predict(len(test_data), test_data[features])
    test_data['Forecast_ARIMA']=forecast

    # save as .csv to prediction
    d = {'Ölçüm Rüzgar Hızı':test_data['Ölçüm Rüzgar Hızı'], 'UrClimate Rüzgar Hızı':test_data['UrClimate Rüzgar Hızı'] , 'Predicted Data':test_data['Forecast_ARIMA']}
    result = pd.DataFrame(data=d)
    result.to_csv(output, index=True)
    
    print('Başarı Puanı: ' + str(r2_score(test_data['Ölçüm Rüzgar Hızı'], test_data['Forecast_ARIMA'])))
    print("Root Mean Squarred Error:", np.sqrt(mean_squared_error(test_data['Ölçüm Rüzgar Hızı'],test_data['Forecast_ARIMA'])))
    print("Mean Absolute Error:", mean_absolute_error(test_data['Ölçüm Rüzgar Hızı'],test_data['Forecast_ARIMA']))

    return model, training_data, test_data

def visul_old(output):
    datax = pd.read_csv(output)
    
    #datax[['Ölçüm Rüzgar Hızı', 'UrClimate Rüzgar Hızı', 'Predicted Data']].plot(figsize=(24,10))
    
    # Visualization Data
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=datax['Zaman'][int(datax.shape[0]*.70):], y=datax['Ölçüm Rüzgar Hızı'], name='Gerçek Veriler', mode='lines'))
    fig1.add_trace(go.Scatter(x=datax['Zaman'][int(datax.shape[0]*.70):], y=datax['UrClimate Rüzgar Hızı'], name='UrClimate', mode='lines'))
    fig1.add_trace(go.Scatter(x=datax['Zaman'][int(datax.shape[0]*.70):], y=datax['Predicted Data'], name='Öngörü', mode='lines'))

def visul(output):
    datax = pd.read_csv(output)
    
    plt.figure(figsize=(10,4))
    
    plt.plot(datax['Ölçüm Rüzgar Hızı'], marker='o', label = 'Ölçüm Rüzgar Hızı')
    plt.plot(datax['Predicted Data'],  'v--r', label = 'Predicted Data')
    plt.plot(datax['UrClimate Rüzgar Hızı'], label = 'UrClimate Rüzgar Hızı')
    
    plt.xlabel('Zaman')
    plt.xticks(rotation = 45)
    plt.ylabel('Values')
    plt.title('Rüzgar Hızı Grafik', fontsize=14)
    plt.grid()
    plt.legend()
    
    # Visualization Data
    #fig1 = go.Figure()
    #fig1.add_trace(go.Scatter(x=datax['Zaman'][int(datax.shape[0]*.70):], y=datax['Ölçüm Rüzgar Hızı'], name='Gerçek Veriler', mode='lines'))
    #fig1.add_trace(go.Scatter(x=datax['Zaman'][int(datax.shape[0]*.70):], y=datax['UrClimate Rüzgar Hızı'], name='UrClimate', mode='lines'))
    #fig1.add_trace(go.Scatter(x=datax['Zaman'][int(datax.shape[0]*.70):], y=datax['Predicted Data'], name='Öngörü', mode='lines'))


if __name__ == "__main__":
    print("İşlem başladı")
    main('datasets/soke_res_temmuz2022.xlsx', 'predictionx.csv')
