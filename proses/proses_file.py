import streamlit as st
import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from proses.augmentasi import *
from config.conn import *

# FUngsi untuk upload file pada fitur gridsearch(grid.py) dan Kombinasi(kombinasi.py)
def up_file():
    # Sidebar upload folder
    st.sidebar.header("Upload File")
    upload_folder = "uploaded_files"
    if not os.path.exists(upload_folder):
        os.makedirs(upload_folder)

    up_files = st.sidebar.file_uploader("Pilih file CSV", accept_multiple_files=True, type=['csv'])
    if up_files:
        for up_file in up_files:
            bytes_data = up_file.read()

            # simpan file ke dalam folder
            file_path = os.path.join(upload_folder, up_file.name)
            with open(file_path, 'wb') as f:
                f.write(bytes_data)

            st.success(f"File {up_file.name} berhasil di-upload dan disimpan di {file_path}.")
        # cek file yang di upload
    if not up_files:
        st.warning("Silakan unggah file CSV terlebih dahulu.")
        st.stop()
    return upload_folder, up_files

# FUngsi untuk membaca file pada fitur gridsearch(grid.py) dan Kombinasi(kombinasi.py)
def baca_file(upload_folder, up_files):
    # Membaca data dalam file csv
    file_path = os.path.join(upload_folder, up_files[0].name)
    df_tm = pd.read_csv(file_path)
    daftar_col = df_tm.columns.tolist()
    df = pd.read_csv(file_path, index_col=daftar_col[0], parse_dates=[daftar_col[0]])

    # list kolom dan tampilkan data berdasarkan kolom yang dipilih
    daftar_kolom = df.columns.tolist()
    y_col = st.sidebar.selectbox("value", options=daftar_kolom)
    chart_data = df[y_col]

    if len(df) < 1000:
        # Augmentasi data dengan penggeseran
        shift_range = 2  # Anda bisa menyesuaikan nilai ini
        df = augment_shift(df, y_col, shift_range)
    else:
        df

    return y_col, df, chart_data

# Fungsi Normalisasi
def normalisasi(df, y_col):
    # Min-Max Normalization
    scaler = MinMaxScaler()
    dfnorm = scaler.fit_transform(df[[y_col]])
    df_normalisasi = dfnorm

    return df_normalisasi, scaler

# FUngsi untuk split pada fitur gridsearch(grid.py) dan Kombinasi(kombinasi.py)
def split_data(df_normalisasi):
    # Split data into train, validation, and test sets
    totaldata = df_normalisasi
    st.sidebar.header("Split Data")

    # Kolom untuk memilih persentase data
    sel1, sel2 = st.sidebar.columns(2)
    jml_val = 0.2
    totaldataval = int(len(totaldata) * jml_val)
    with sel1:
        jml_train = st.selectbox("Train Set", [0.7, 0.8], index=1)
        train_jml = jml_train - jml_val
        totaldatatrain = int(len(totaldata) * train_jml)
    with sel2:
        i = 0
        if jml_train == 0.7 :
            i = 1
        jml_tes = st.selectbox("Test Set", [0.2, 0.3], index=i)
        totaldatatest = int(len(totaldata) * jml_tes)
    if train_jml + jml_val + jml_tes > 1.0:
        st.error("Total dari train, val, dan test set melebihi jumlah data keseluruhan.")
        st.stop()

    # Store data into each partition
    training_set = df_normalisasi[0:totaldatatrain]
    val_set = df_normalisasi[totaldatatrain:totaldatatrain+totaldataval]
    test_set = df_normalisasi[totaldatatrain+totaldataval:]
        
    # Sliding windows function
    lag = 12
    def create_sliding_windows(data, len_data, lag):
        x, y = [], []
        for i in range(lag, len_data):
            x.append(data[i-lag:i, 0])
            y.append(data[i, 0])
        return np.array(x), np.array(y)

    # Formatting data into array for creating sliding windows
    array_training_set = np.array(training_set).reshape(-1, 1)
    array_val_set = np.array(val_set).reshape(-1, 1)
    array_test_set = np.array(test_set).reshape(-1, 1)

    # Buat sliding windows for training, validation, and test sets
    x_train, y_train = create_sliding_windows(array_training_set, len(array_training_set), lag)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    x_val, y_val = create_sliding_windows(array_val_set, len(array_val_set), lag)
    x_val = np.reshape(x_val, (x_val.shape[0], x_val.shape[1], 1))

    x_test, y_test = create_sliding_windows(array_test_set, len(array_test_set), lag)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    return  totaldatatrain, totaldataval, jml_train, jml_tes ,lag, x_train, x_val, y_train, y_val, x_test, y_test, training_set, val_set, test_set

# Fungsi Untuk mengambil data pada fitur prediksi hasil.py
def split_data_db():
    pilih = st.sidebar.selectbox("Data Parameter", ["GridShearch", "Kombinasi Manual"], index=1)
    if pilih == "GridShearch":
        data = tampil_grid_best()
    else:
        data = tampil_parameter()

    if not data :
        st.warning("Silahkan Lakukan GridSearch Terlebih Dahulu")
        st.stop()
    
    st.sidebar.subheader("Kombinasi Parameter")
    kom1, kom2 = st.sidebar.columns(2)
    with kom1:
        # Mengambil Nama File Sesuai Nama di Database
        unique_nm_files = list(set(row[1] for row in data))
        nm_file = st.selectbox("file", unique_nm_files  )
        file_path = os.path.join("uploaded_files", nm_file)
    with kom2:
        # Tampilkan Data dan pilih yang data yang ditampilkan berdasarkan kolom
        df_tm = pd.read_csv(file_path)
        daf_col = df_tm.columns.tolist()
        df = pd.read_csv(file_path, index_col=daf_col[0], parse_dates=[daf_col[0]])
        daftar_col = list(set(row[2] for row in data if row[1] == nm_file))
        y_col = st.selectbox("value", options=daftar_col)
    
    chart_data = df[y_col]
    # st.line_chart(chart_data.tail(12))
    
    if len(df) < 1000:
        # Augmentasi data dengan penggeseran
        shift_range = 2  # Anda bisa menyesuaikan nilai ini
        df = augment_shift(df, y_col, shift_range)
        
    df_normalisasi, scaler  = normalisasi(df, y_col)

    # Split Data / Membagi Data
    latih, uji = st.sidebar.columns(2)
    with latih:
        daf_latih = list(set(row[4] for row in data if row[1] == nm_file and row[2] == y_col))
        latih = st.selectbox("Latih", options=daf_latih)
    with uji:
        daf_uji = list(set(row[5] for row in data if row[1] == nm_file and row[2] == y_col and row[4] == latih))
        uji = st.selectbox("Uji", options=daf_uji)
    val_jml = 0.2
    train_jml = latih - val_jml
    test_jml = uji

    totaldata = df_normalisasi

    totaldatatrain = int(len(totaldata)*train_jml)
    totaldataval = int(len(totaldata)*val_jml)
    totaldatatest = int(len(totaldata)*test_jml)

    # Store data into each partition
    training_set = df_normalisasi[0:totaldatatrain]
    val_set=df_normalisasi[totaldatatrain:totaldatatrain+totaldataval]
    test_set = df_normalisasi[totaldatatrain+totaldataval:]

    # Sliding windows function
    lag = 12

    def create_sliding_windows(data, len_data, lag):
        x, y = [], []
        for i in range(lag, len_data):
            x.append(data[i-lag:i, 0])
            y.append(data[i, 0])
        return np.array(x), np.array(y)

    # Formatting data into array for creating sliding windows
    array_training_set = np.array(training_set).reshape(-1, 1)
    array_val_set = np.array(val_set).reshape(-1, 1)
    array_test_set = np.array(test_set).reshape(-1, 1)

    # Create sliding windows for training, validation, and test sets
    x_train, y_train = create_sliding_windows(array_training_set, len(array_training_set), lag)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    x_val, y_val = create_sliding_windows(array_val_set, len(array_val_set), lag)
    x_val = np.reshape(x_val, (x_val.shape[0], x_val.shape[1], 1))

    x_test, y_test = create_sliding_windows(array_test_set, len(array_test_set), lag)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    
    return data, df_tm, chart_data,nm_file, df, df_normalisasi , y_col, latih, x_train, y_train, x_val, y_val, x_test, totaldatatrain, totaldataval, lag, scaler

# Fungsi Untuk mengambil parameter pada fitur prediksi hasil.py
def dt_param(data, nm_file, y_col, latih):
    # # Hyperparameters
    epoch = list(set(row[7] for row in data if row[1] == nm_file and row[2] == y_col and row[4] == latih))
    epoch = st.sidebar.selectbox("Epoch:", epoch)
    epoch = int(epoch)
    
    nilai_terkecil = None
    mae_list = list(set(row[12] for row in data if row[1] == nm_file and row[2] == y_col and row[4] == latih and row[7] == epoch))

    # Mencari nilai terkecil dalam list MAE
    for nilai in mae_list:
        if nilai_terkecil is None or nilai < nilai_terkecil:
            nilai_terkecil = nilai

    # Extract values from the data  
    learning_rate = list(row[10] for row in data if row[12] == nilai_terkecil)[0]
    hidden_unit = list(row[9] for row in data if row[12] == nilai_terkecil)[0]
    batch_size = list(row[6] for row in data if row[12] == nilai_terkecil)[0]
    dropout_rate = list(row[8] for row in data if row[12] == nilai_terkecil)[0]
    optimizer = list(row[11] for row in data if row[12] == nilai_terkecil)[0]

    return learning_rate, hidden_unit, batch_size, optimizer, dropout_rate, epoch

# FUngsi untuk tampilan pada fitur gridsearch(grid.py) dan Kombinasi(kombinasi.py)
def tampil(df, y_col, chart_data, df_normalisasi,training_set, val_set, test_set):
    st.subheader(f"Data Penumpang {y_col}")
    st.line_chart(chart_data)
    
    if (len(df)/5) < 1000:
        st.write("#### Data di Augmentasi dan Normalisasi")
    else:
        st.write("#### Data Normalisasi")
    # Menampilkan data latih, validasi, uji
    col_nor1,col_nor2,col_nor3,col_nor4 = st.columns(4)
    with col_nor1:
        st.write(f"Data Ternormalisasi")
        st.line_chart(df_normalisasi, height=200, color='#00008b')
    with col_nor2:
        st.write(f"Latih : {len(training_set)} data")
        st.line_chart(training_set, height=200, color='#0000ff')
    with col_nor3:
        st.write(f"Validasi: {len(val_set)} data")
        st.line_chart(val_set, height=200)
    with col_nor4:
        st.write(f"Uji: {len(test_set)} data")
        st.line_chart(test_set, height=200, color="#1e90ff")