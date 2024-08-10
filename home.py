import streamlit as st
from config.conn import *
import pandas as pd

def home():
    st.header("SELAMAT DATANG")
    st.write("**Aplikasi ini khusus memprediksi data time series menggunakan Gated Recurrent Unit**")

    data = tampil_parameter()
    if not data:
        st.warning("Belum Riwayat Peramalan")
        st.stop()

    # Misalnya kita ingin mencari nilai terkecil dari kolom 'mae'
    kolom_mae = 12  
    nilai_terkecil = None

    for row in data:
        mae = row[kolom_mae]
        if nilai_terkecil is None or mae < nilai_terkecil:
            nilai_terkecil = mae

    data = {col: list({row[col] for row in data if row[12] == nilai_terkecil }) 
        for col in [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]}

    from streamlit_extras.metric_cards import style_metric_cards
    col1, col2= st.columns(2) 
    bagi_dt = f"{data[4][0]*100:,.0f}% : {data[5][0]*100:,.0f}%"
    n_error = f"{data[12][0]:,.0f} , {data[13][0]} , {data[14][0]}s"
    parameter1 = f"{data[7][0]} , {data[6][0]} , {data[8][0]}"
    parameter2 = f"{data[11][0]} , {data[9][0]} , {data[10][0]}"
    
    #Total customer
    col1.metric("Pembagian Data", value=str(bagi_dt), delta="Data Latih dan Uji")
    col1.metric("MAE , MAPE , Waktu ", value=n_error, delta="Nilai Error")
    col2.metric("Epoch , Batch Size , Dropout", value=str(parameter1), delta="Parameter")
    col2.metric("Optimalisasi , Unit, Dropout", value=str(parameter2), delta="Parameter")

    style_metric_cards(background_color="#071021", border_left_color="#1f66bd")

    data_prediksi = tampil_hasil_predikisi()
    # Buat DataFrame dengan kolom yang sesuai
    pred = pd.DataFrame(data_prediksi, columns=["Bulan", "Data", "Prediksi"])
    selected_nama = st.selectbox('Prediksi Dari Data', sorted(pred['Data'].unique()))
            # Filter DataFrame berdasarkan epoch yang dipilih
    his_filtered = pred[
            (pred['Data'] == selected_nama)
            ] 
    t1, t2 = st.columns([1, 1.5])
    with t1:
        st.write(his_filtered)
    # # Konversi kolom 'Bulan' ke tipe datetime
    his_filtered['Bulan'] = pd.to_datetime(his_filtered['Bulan'])

    # # Ambil hanya bulan dan atur sebagai index
    his_filtered['Bulan'] = his_filtered['Bulan'].dt.strftime('%Y-%m')
    his_filtered.set_index("Bulan", inplace=True)
    with t2:
        st.line_chart(his_filtered["Prediksi"], height= 400)


