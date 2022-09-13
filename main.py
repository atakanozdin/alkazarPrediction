from turtle import title
import streamlit as st 
from plotly import graph_objs as go
from function import *
from io import StringIO
import pandas as pd

st.title('Alkazar Rüzgar Hızı Yapay Zeka')

uploaded_file = st.file_uploader("Excel formatında bir dosya seçiniz")
if uploaded_file is not None:
#read excel
    uploaded_file_load_state = st.write('Tahminleme yapılıyor. Lütfen bekleyiniz.')
    model, training_data, test_data =main(uploaded_file, 'prediction.csv')
    
    df_pred = pd.read_csv('prediction.csv')
    df_real = pd.read_excel(uploaded_file)

    #-----------------------------------------------------------------------------------------------------------
    st.subheader('Gerçek Veri')
    st.write(df_real.sample(10))

    def plot_realData():
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_real['Zaman'], y=df_real['Ölçüm Rüzgar Hızı']))
        #fig.layout.update(title_text='Gerçek Rüzgar Hızları', x_axis_rangeslider_visible=True)
        st.plotly_chart(fig)
    plot_realData()
    
    #-----------------------------------------------------------------------------------------------------------
    st.subheader('Yapay Zeka Tahmini Başarı Puanları')
    st.write('R2 Score: ' + str(r2_score(test_data['Ölçüm Rüzgar Hızı'], test_data['Forecast_ARIMA'])))
    st.write("Root Mean Squarred Error: " + str(np.sqrt(mean_squared_error(test_data['Ölçüm Rüzgar Hızı'],test_data['Forecast_ARIMA']))))
    st.write("Mean Absolute Error: " + str(mean_absolute_error(test_data['Ölçüm Rüzgar Hızı'],test_data['Forecast_ARIMA'])))

    #-----------------------------------------------------------------------------------------------------------
    st.subheader('Tahmin Verisi')
    st.write(df_pred.sample(10))

    def plot_predictData():
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=df_pred['Zaman'], y=df_pred['Ölçüm Rüzgar Hızı']))
        fig2.add_trace(go.Scatter(x=df_pred['Zaman'], y=df_pred['UrClimate Rüzgar Hızı']))
        fig2.add_trace(go.Scatter(x=df_pred['Zaman'], y=df_pred['Predicted Data']))
        #fig.layout.update(title_text='Gerçek Rüzgar Hızları', x_axis_rangeslider_visible=True)
        st.plotly_chart(fig2)
    plot_predictData()

else:
    st.write('Excel formatında bir dosya seçmelisiniz. Lütfen sayfayı yenileyip yeniden seçim yapınız.')