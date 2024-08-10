# Import Library Python

import streamlit as st
import numpy as np
import tensorflow as tf
from scikeras.wrappers import KerasRegressor
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dropout, Dense
from tensorflow.keras.optimizers import Adam, RMSprop
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error
import matplotlib.pyplot as plt
import time
import joblib


# Import dari file yang ada di directory
from config.conn import *
from proses.augmentasi import *
from proses.proses_file import *

def proses():
    st.header("GRID SEARCH")
    st.write("Grid Search metode untuk mencari hyperparameter terbaik dari model GRU")
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

    def gru(optimizer, dropout_rate, learning_rate, hidden_unit):
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


    # Input parameters berdasarkan inputan user
    st.sidebar.header("Grid Search")
    batch_sizes = st.sidebar.multiselect('Batch Sizes', options=[16, 32, 64], default=[32])
    epochs = st.sidebar.multiselect('Epochs', options=[100, 500, 1000], default=[100])
    dropout_rates = st.sidebar.multiselect('Dropout Rates', options=[0.2, 0.25], default=[0.2])
    hidden_units = st.sidebar.multiselect('Hidden Units', options=[32, 64], default=[64])
    learning_rates = st.sidebar.multiselect('Learning Rates', options=[0.01, 0.001, 0.0001], default=[0.01])
    optimizers = st.sidebar.multiselect('Optimizers', options=['Adam', 'RMSprop'], default=['Adam'])

    kombinasi = f"{batch_sizes} , {epochs}, {dropout_rates}, {hidden_units}, {learning_rates}, {optimizers}"

    # buat kombinasi parameter gridsearch
    param_grid = {
        'batch_size': batch_sizes,
        'epochs': epochs,
        'model__dropout_rate': dropout_rates,
        'model__hidden_unit': hidden_units,
        'model__learning_rate': learning_rates,
        'model__optimizer': optimizers
    }

    def run_grid_search():
        model = KerasRegressor(build_fn=gru, verbose=1)

        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='neg_mean_squared_error', cv=3, verbose=1)

        log_placeholder = st.empty()
        time_placeholder = st.empty()

        class StreamlitLogger(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                log_placeholder.text(f"Epoch {epoch+1}/{self.params['epochs']}\n"
                                    f"{'='*30}\n"
                                    f"loss: {logs['loss']:.4f} - val_loss: {logs['val_loss']:.4f}\n")

        start_time = time.time()

        with st.spinner('Melakukan Grid Search Model...'):
            grid_result = grid_search.fit(x_train, y_train, validation_data=(x_val, y_val), callbacks=[StreamlitLogger()])

        end_time = time.time()  # Record end time
        duration = end_time - start_time  # Calculate duration
        lama_jalan = f"{duration:.2f}"
        
        st.success('Grid Search completed!')
        time_placeholder.text(f"Total Waktu yang Diperlukan: {lama_jalan} detik")  #

        if hasattr(grid_result, 'best_params_'):
            best_score = grid_result.best_score_
            best_params = grid_result.best_params_
            best_model = grid_result.best_estimator_
            # val_loss = best_model.score(x_val, y_val)

            df_best_params = pd.DataFrame(list(best_params.items()), columns=['Parameter', 'Value'])
            y_pred = best_model.predict(x_test)

            # tampilkan best parameter
            hasil1, hasil2 = st.columns([1,2])
            with hasil1:
                st.write("### Parameter Terbaik")
                st.write(f"**Best scoren :** {best_score:.6f}")
                st.write(df_best_params.set_index('Parameter'))
            with hasil2:
                plt.figure(figsize=(10, 6))
                plt.plot(y_test, label='Actual')
                plt.plot(y_pred, label='Predicted')
                plt.title('Actual vs Predicted')
                plt.xlabel('Observation')
                plt.ylabel('Value')
                plt.legend()
                st.pyplot(plt)

            # Denormalisasi min-max
            y_pred_invert_norm = scaler.inverse_transform(y_pred.reshape(-1, 1))

            # Uji dengan data setelah data di denormalisasi
            datacompare = pd.DataFrame()
            datatest = np.array(df[y_col][totaldatatrain + totaldataval + lag:])
            datapred = y_pred_invert_norm
            datacompare['Data Test'] = datatest
            datacompare['Prediction Results'] = datapred
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

            # simpan hasil ke dalam database
            simpan_grid_best(up_files[0].name, y_col, len(chart_data), jml_train, jml_tes, kombinasi, best_params['batch_size'], best_params['epochs'], best_params['model__dropout_rate'], best_params['model__hidden_unit'], best_params['model__learning_rate'], best_params['model__optimizer'], best_score, mae, mape, lama_jalan)
                
        else:
            st.write("Grid Search belum selesai atau tidak menemukan parameter terbaik.")

    # Custom CSS tombol 
    st.markdown("""
        <style>
        .stButton>button {
            color: white;
            width: 100%;
        }
        </style>
        """, unsafe_allow_html=True)

    if st.sidebar.button("Jalankan Grid Search"):
        run_grid_search()
