import streamlit as st
from config.conn import *
from proses.augmentasi import *
from proses.gru import *
from proses.proses_file import *

# Fungsi untuk memproses file yang sesuai dengan nama dari database
def latih_terbaik():
    st.header("LATIH DATA MENGGUNAKAN PARAMETER TERBAIK")
    data, df_tm, chart_data, nm_file, df, df_normalisasi , y_col, latih, x_train, y_train, x_val, y_val, x_test, totaldatatrain, totaldataval, lag, scaler = split_data_db()
    learning_rate, hidden_unit, batch_size, optimizer, dropout_rate, epoch = dt_param(data, nm_file, y_col, latih)
    # Tampilkan dalam Chart
    st.write(f"### Latih Data {y_col}")
    st.line_chart(chart_data)

    st.sidebar.write("Learning Rate:", learning_rate)
    st.sidebar.write("Hidden Unit:", hidden_unit)
    st.sidebar.write("Batch Size:", batch_size)
    st.sidebar.write("Dropout Rate:", dropout_rate)
    st.sidebar.write("optimizer:", optimizer)
        # Custom CSS tombol 
    st.markdown("""
        <style>
        .stButton>button {
            background-color: #0000FF;
            color: white;
            width: 100%;
        }
        </style>
        """, unsafe_allow_html=True)
    if st.sidebar.button("Latih Model"):
        mae, mape, lama_jalan = run_model(df, y_col, scaler, totaldatatrain, totaldataval, lag,x_train, y_train,x_val,y_val,x_test, batch_size, optimizer, dropout_rate, learning_rate, epoch, hidden_unit)
