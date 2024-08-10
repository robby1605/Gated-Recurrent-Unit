import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from config.conn import *
from proses.augmentasi import *
from proses.gru import *
from proses.proses_file import *

# Fungsi untuk memproses file yang sesuai dengan nama dari database
def prediksi():
    st.header("PREDIKSI TIME SERIES")
    data, df_tm, chart_data, nm_file, df, df_normalisasi , y_col, latih, x_train, y_train, x_val, y_val, x_test, totaldatatrain, totaldataval, lag, scaler = split_data_db()
    # Tampilkan dalam Chart
    st.write(f"### Prediksi Data {y_col}")
    st.line_chart(chart_data)

    # Muat model yang telah disimpan
    model_folder = "models"
    model_filename = f"trained_model_{y_col}.pkl"
    model_path = os.path.join(model_folder, model_filename)

    if os.path.exists(model_path):
        model = joblib.load(model_path)
        st.success(f"Model berhasil dimuat dari {model_path}")
    else:
        st.warning(f"Model dengan nama {model_path} tidak ditemukan. Silakan lakukan pelatihan terlebih dahulu.")
        return
    
    # # Architecture Gated Recurrent Unit
    def run_model(n_future):
        def prepare_future_data(data, n_future):
            last_data = data[-lag:]
            future_data = []

            for _ in range(n_future):
                # Reshape data to match input shape
                input_data = last_data.reshape((1, lag, 1))
                future_pred = model.predict(input_data)
                future_data.append(future_pred[0][0])
                # Append the predicted value to last_data and slide the window
                last_data = np.append(last_data[1:], future_pred[0][0])

            return np.array(future_data)

        # Predict future 12 months
        norm1, scaler1 = normalisasi(df_tm, y_col)

        n_future = n_future
        future_predictions = prepare_future_data(np.array(norm1), n_future)
        future_predictions_invert_norm = scaler1.inverse_transform(future_predictions.reshape(-1, 1))

        # Membuat indeks baru untuk prediksi
        last_date = chart_data.index[-1]
        future_dates = pd.date_range(start=last_date, periods=n_future + 1, freq='M')[1:]

        # Membuat DataFrame baru untuk prediksi
        future_df = pd.DataFrame(future_predictions_invert_norm, index=future_dates, columns=[y_col])

        # Menggabungkan data asli dengan prediksi
        combined_df = pd.concat([chart_data.tail(12), future_df])

        actual_data = combined_df.iloc[:-n_future]
        predicted_data = combined_df.iloc[-n_future:]

        # Plotting dengan plotly
        fig = go.Figure()

        # Garis data aktual
        fig.add_trace(go.Scatter(x=actual_data.index, y=actual_data[y_col], mode='lines', name='Actual Data', line=dict(color='blue')))

        # Garis data prediksi
        fig.add_trace(go.Scatter(x=predicted_data.index, y=predicted_data[y_col], mode='lines', name='Predicted Data', line=dict(color='red', dash='dash')))

        # Menampilkan data dalam chart
        st.write(f"### Prediksi Data {y_col}")
        st.plotly_chart(fig)

        # Menyimpan hasil prediksi ke database
        simpan_hasil_prediksi(future_dates, y_col, future_predictions_invert_norm.flatten())

    bulan = st.number_input("Berapa bulan yang akan diprediksi ???", min_value=1, max_value=1000)
    if st.button("Prediksi"):
        run_model(bulan)
