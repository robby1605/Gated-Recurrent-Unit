import os
import time
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dropout, Dense
from tensorflow.keras.optimizers import Adam, RMSprop
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error

def gru(x_train, optimizer, dropout_rate, learning_rate, hidden_unit):
    model = Sequential()
    # # First GRU layer with dropout
    model.add(GRU(units=hidden_unit, return_sequences=True, input_shape=(x_train.shape[1],1), activation = 'tanh', recurrent_activation='sigmoid'))
    model.add(Dropout(dropout_rate))
    # Second GRU layer with dropout
    model.add(GRU(units=hidden_unit, return_sequences=True, activation = 'tanh', recurrent_activation='sigmoid'))
    model.add(Dropout(dropout_rate))
    # Third GRU layer with dropout
    model.add(GRU(units=hidden_unit, return_sequences=False, activation = 'tanh', recurrent_activation='sigmoid'))
    model.add(Dropout(dropout_rate))
    # Output layer
    model.add(Dense(units=1))
    # Assuming optimalisasi is a string like 'Adam' or 'RMSprop'
    if optimizer == 'Adam':
        optimizer = Adam(learning_rate=learning_rate)
    elif optimizer == 'RMSprop':
        optimizer = RMSprop(learning_rate=learning_rate)
    else:
        raise ValueError(f"Optimizer '{optimizer}' is not supported.")
    # # Compiling the Gated Recurrent Unit
    model.compile(optimizer=optimizer, loss='mean_squared_error')

    return model

   # # Architecture Gated Recurrent Unit
def run_model(df, y_col, scaler, totaldatatrain, totaldataval, lag,x_train, y_train,x_val,y_val,x_test, batch_size, optimizer, dropout_rate, learning_rate, epoch, hidden_unit):
    model = gru(x_train, optimizer, dropout_rate, learning_rate, hidden_unit)
    
    time_placeholder = st.empty()
    start_time = time.time()  # Record start time

    with st.spinner('Melakukan Prediksi...'):
        pred = model.fit(x_train, y_train, validation_data=(x_val,y_val), batch_size=batch_size, epochs=epoch)

    end_time = time.time()  # Record end time
    duration = end_time - start_time  # Calculate duration
    lama_jalan = f"{duration:.0f}"

    st.success('Prediksi completed!')
    time_placeholder.text(f"Total Waktu yang Diperlukan: {lama_jalan} detik") 
    # Graph model loss (train loss & val loss)
    plt.figure(figsize=(10, 4))
    plt.plot(pred.history['loss'], label='train loss')
    plt.plot(pred.history['val_loss'], label='val loss')
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(loc='upper right')
    # plt.show()
    st.pyplot(plt)

    # Buat folder models jika belum ada
    model_folder = "../models"
    os.makedirs(model_folder, exist_ok=True)
    # Tentukan path file model
    model_filename = f"trained_model_{y_col}.pkl"
    model_path = os.path.join(model_folder, model_filename)
    # Simpan model ke file dengan nama berdasarkan y_col
    joblib.dump(model, model_path)
    st.success(f"Model berhasil disimpan ke {model_path}")

    y_pred = model.predict(x_test)
    # Denormalisasi min-max
    y_pred_invert_norm = scaler.inverse_transform(y_pred)

    # Uji dengan data setelah data di denormalisasi
    datacompare = pd.DataFrame()
    datatest = df[y_col][totaldatatrain + totaldataval + lag:]
    datapred = y_pred_invert_norm
    datacompare['Data Uji'] = datatest
    datacompare['Hasi Prediksi'] = datapred
    st.write("### Prediksi Setelah Denormalisasi")
    st.line_chart(datacompare)

    # Menghitung MAE, MAPE
    mae = mean_absolute_error(datatest, datapred)
    mape = mean_absolute_percentage_error(datatest, datapred)
    mape = f'{mape*100:.2f}%'
    error1, error2= st.columns(2)
    with error1:
        st.success(f'Hasil MAE : {mae:.2f}')
    with error2:
        st.success(f'Hasil MAPE : {mape}')

    return mae, mape, lama_jalan