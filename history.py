import streamlit as st
from config.conn import *
import pandas as pd

def tampil_history():
    st.header("History Best Parameter Model")
    def grid():
        st.subheader("Grid Search")
        grid=tampil_grid_best()
        if not grid:
            st.warning("Anda tidak memiliki riwayat GridSearch.")
            st.stop()

        # Buat DataFrame dengan kolom yang sesuai
        his = pd.DataFrame(grid, columns=["ID", "Nama", "Nama Kolom", "Total","Data latih","Data uji", "Batch size","Epochs","Dropout","Hidden unit","Learning rate","Optimizer", "MAE", "MAPE", "Waktu"])
        
        # Buat widget untuk memilih nilai
        filter1, filter2, filter3, filter4, filter5 = st.columns(5)
        with filter1:
            selected_nama = st.selectbox('File', sorted(his['Nama'].unique()))
        with filter2:
            selected_kolom = st.selectbox('Kolom', sorted(his['Nama Kolom'].unique()))
        with filter3:   
            selected_latih = st.selectbox('Latih', sorted(his['Data latih'].unique()))
        with filter4:   
            selected_epoch = st.selectbox('Epoch', sorted(his['Epochs'].unique()))
        with filter5:   
            jml_data = st.number_input('Tampilkan', min_value=1, max_value=100, value=10)

        # Filter DataFrame berdasarkan epoch yang dipilih
        his_filtered = his[
            (his['Nama'] == selected_nama) & 
            (his['Nama Kolom'] == selected_kolom) &
            (his['Data latih'] == selected_latih) &
            (his['Epochs'] == selected_epoch)
            ] 
        # Urutkan DataFrame berdasarkan MAE terkecil
        his_sorted = his_filtered.sort_values(by="MAE")

        # Ambil 10 baris data pertama setelah pengurutan
        top = his_sorted.head(int(jml_data))
        top.index = range(1, len(top) + 1)

        # Tampilkan DataFrame yang telah diurutkan
        st.dataframe(top)

        # Tambahkan widget untuk memilih data yang akan dihapus
        selected_ids = st.multiselect(
            'Pilih ID untuk menghapus data',
            top['ID'].tolist()
        )

        # Tombol untuk menghapus data berdasarkan ID
        if st.button('Hapus Data'):
            if selected_ids:
                delete_data_by_id_grid(selected_ids)
                st.success(f'Data dengan ID {selected_ids} telah dihapus.')
                # Refresh DataFrame setelah penghapusan
                grid=tampil_grid_best()
                his = pd.DataFrame(grid, columns=["ID", "Nama", "Nama Kolom", "Total","Data latih","Data uji", "Batch size","Epochs","Dropout","Hidden unit","Learning rate","Optimizer", "MAE", "MAPE", "Waktu"])
                his_filtered = his[
                    (his['Nama'] == selected_nama) &
                    (his['Nama Kolom'] == selected_kolom) &
                    (his['Data latih'] == selected_latih) &
                    (his['Epochs'] == selected_epoch)
                ]
                his_sorted = his_filtered.sort_values(by="MAE")
                top = his_sorted.head(int(jml_data))
                top.index = range(1, len(top) + 1)
                st.dataframe(top)
            else:
                st.error('Tidak ada ID yang dipilih untuk dihapus.')


    def kombinasi():
        st.subheader("Kombinasi Manual")
        grid = tampil_parameter()
        if not grid:
            st.warning("Anda tidak memiliki riwayat GridSearch.")
            st.stop()

        # Buat DataFrame dengan kolom yang sesuai
        his = pd.DataFrame(grid, columns=["ID", "Nama", "Nama Kolom", "Total", "Data latih", "Data uji", "Batch size", "Epochs", "Dropout", "Hidden unit", "Learning rate", "Optimizer", "MAE", "MAPE", "Waktu"])

        # Buat widget untuk memilih nilai epoch
        filter1, filter2, filter3, filter4, filter5 = st.columns(5)
        with filter1:
            selected_nama = st.selectbox('File', sorted(his['Nama'].unique()))
        with filter2:
            selected_kolom = st.selectbox('Kolom', sorted(his['Nama Kolom'].unique()))
        with filter3:
            selected_latih = st.selectbox('Latih', sorted(his['Data latih'].unique()))
        with filter4:
            selected_epoch = st.selectbox('Epoch', sorted(his['Epochs'].unique()))
        with filter5:
            jml_data = st.number_input('Tampilkan ', min_value=1, max_value=100, value=10)

        # Filter DataFrame berdasarkan epoch yang dipilih
        his_filtered = his[
            (his['Nama'] == selected_nama) &
            (his['Nama Kolom'] == selected_kolom) &
            (his['Data latih'] == selected_latih) &
            (his['Epochs'] == selected_epoch)
        ]
        # Urutkan DataFrame yang sudah difilter berdasarkan MAE terkecil
        his_sorted = his_filtered.sort_values(by="MAE")

        # Ambil beberapa baris data pertama setelah pengurutan
        top = his_sorted.head(int(jml_data))
        top.index = range(1, len(top) + 1)

        # Tampilkan DataFrame yang telah diurutkan
        st.dataframe(top)

        # Tambahkan widget untuk memilih data yang akan dihapus
        selected_ids = st.multiselect(
            'Pilih ID untuk menghapus data',
            top['ID'].tolist()
        )

        # Tombol untuk menghapus data berdasarkan ID
        if st.button('Hapus Data'):
            if selected_ids:
                delete_data_by_id(selected_ids)
                st.success(f'Data dengan ID {selected_ids} telah dihapus.')
                # Refresh DataFrame setelah penghapusan
                grid = tampil_parameter()
                his = pd.DataFrame(grid, columns=["ID", "Nama", "Nama Kolom", "Total", "Data latih", "Data uji", "Batch size", "Epochs", "Dropout", "Hidden unit", "Learning rate", "Optimizer", "MAE", "MAPE", "Waktu"])
                his_filtered = his[
                    (his['Nama'] == selected_nama) &
                    (his['Nama Kolom'] == selected_kolom) &
                    (his['Data latih'] == selected_latih) &
                    (his['Epochs'] == selected_epoch)
                ]
                his_sorted = his_filtered.sort_values(by="MAE")
                top = his_sorted.head(int(jml_data))
                top.index = range(1, len(top) + 1)
                st.dataframe(top)
            else:
                st.error('Tidak ada ID yang dipilih untuk dihapus.')

    pil = st.sidebar.selectbox('Pilih Data', ['GridSearch', 'Kombinasi'], index=1)
    if pil == 'GridSearch':
        grid()
    else :
        kombinasi()
