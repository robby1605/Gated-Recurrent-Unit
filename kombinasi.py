# Import Library Python

import streamlit as st

# Import dari file yang ada di directory
from config.conn import *
from proses.gru import *
from proses.proses_file import *
from proses.augmentasi import *

def proses_kombinasi():
    st.header("KOMBINASI HYPERPARAMETER")
    st.write("Kombinasi Hyperparameter terdiri dari 2 pembagian data, 3 epoch, 3 learning rate, 2 units, 2 dropout, 3 batch size, dan 2")
    # Sidebar upload folder
    upload_folder, up_files = up_file()
    # Sidebar Membaca File
    y_col, df, chart_data = baca_file(upload_folder, up_files)

    # Normalisasi Data
    df_normalisasi, scaler  = normalisasi(df, y_col)

    # Spliting Data
    totaldatatrain, totaldataval, jml_train, jml_tes ,lag, x_train, x_val, y_train, y_val, x_test, y_test, training_set, val_set, test_set  = split_data(df_normalisasi)

    # tampil 
    tampil(df, y_col, chart_data, df_normalisasi,training_set, val_set, test_set)

    # Input parameters berdasarkan inputan user
    st.sidebar.header("Parameter")
    p_param = st.sidebar.selectbox("Pilih Input Parameter", options=["Berdasarkan Id", "Manual"], index=1)
    if p_param == "Berdasarkan Id":
        data = ambil_id()
        id = st.sidebar.number_input("Masukkan Id", min_value=1, max_value=1000)

        # # Hyperparameters
        params = {col: list({row[col] for row in data if row[0] == id }) 
            for col in [1, 2, 3, 4, 5, 6]}
        batch_size = params[2][0]
        epoch = params[1][0]
        dropout_rate = params[3][0]
        hidden_unit = params[4][0]
        learning_rate = params[5][0]
        optimizer = params[6][0]
        epoch = int(epoch)
        st.sidebar.write("Batcha Size : ", batch_size)
        st.sidebar.write("Epoch : ", epoch)
        st.sidebar.write("Dropout : ", dropout_rate)
        st.sidebar.write("Unit : ", hidden_unit)
        st.sidebar.write("Learning Rate : ", learning_rate)
        st.sidebar.write("Optimizer : ", optimizer)
    elif p_param == "Manual":
        param1, param2, param3 = st.sidebar.columns(3)
        with param1:
            batch_size = st.selectbox('Batch Size', options=[16, 32, 64])
            epoch = st.selectbox('Epoch', options=[100, 500, 1000])
        with param2:
            dropout_rate = st.selectbox('Dropout', options=[0.2, 0.25])
            hidden_unit = st.selectbox('Unit', options=[32, 64])
        with param3:
            learning_rate = st.selectbox('Learning Rate', options=[0.01, 0.001, 0.0001])
            optimizer = st.selectbox('Optimizer', options=['Adam', 'RMSprop'])

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

    if st.sidebar.button("Jalankan Model"):
        mae, mape, lama_jalan = run_model(df, y_col, scaler, totaldatatrain, totaldataval, lag,x_train, y_train,x_val,y_val,x_test, batch_size, optimizer, dropout_rate, learning_rate, epoch, hidden_unit)
        
        # simpan hasil ke dalam database
        simpan_parameter(up_files[0].name, y_col, len(chart_data), jml_train, jml_tes, batch_size, epoch, dropout_rate, hidden_unit, learning_rate, optimizer, mae, mape, lama_jalan)
