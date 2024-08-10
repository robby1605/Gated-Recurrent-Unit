import mysql.connector
import hashlib

def get_connection():
    """Establish a new connection to the database."""
    return mysql.connector.connect(
        host="localhost",
        port="3306",
        user="root",
        passwd="",
        database="gru"
    )

def tampil_grid_best():
    """Fetch all data from the grid_best table."""
    conn = get_connection()
    try:
        c = conn.cursor()
        c.execute('SELECT id, nm_file, nm_kolom, jml_data, latih, uji, batch_size, epochs, dropout_rate, hidden_unit, learning_rate, optimizer, mae, mape, waktu FROM grid_best')
        data = c.fetchall()
    finally:
        c.close()
        conn.close()
    return data

def delete_data_by_id_grid(delete_ids):
    """Delete data from the grid_best table based on the given IDs."""
    conn = get_connection()
    try:
        c = conn.cursor()
        # Pastikan delete_ids adalah daftar tuple
        c.executemany('DELETE FROM grid_best WHERE id = %s', [(id,) for id in delete_ids])
        conn.commit()
    finally:
        c.close()
        conn.close()


def tampil_parameter():
    """Fetch all data from the hasil_kombinasi table."""
    conn = get_connection()
    try:
        c = conn.cursor()
        c.execute('SELECT id, nm_file, nm_kolom, jml_data, latih, uji, batch_size, epoch, dropout_rate, hidden_unit, learning_rate, optimizer, mae, mape, waktu FROM hasil_kombinasi')
        data = c.fetchall()
    finally:
        c.close()
        conn.close()
    return data

def delete_data_by_id(delete_ids):
    """Delete data from the hasil_kombinasi table based on the given IDs."""
    conn = get_connection()
    try:
        c = conn.cursor()
        c.executemany('DELETE FROM hasil_kombinasi WHERE id = %s', [(id,) for id in delete_ids])
        conn.commit()
    finally:
        c.close()
        conn.close()


def ambil_id():
    """Fetch all data from the parameter table."""
    conn = get_connection()
    try:
        c = conn.cursor()
        c.execute('SELECT id, epoch, batch_size, dropout, unit, learning_rate, optimizer FROM parameter')
        data = c.fetchall()
    finally:
        c.close()
        conn.close()
    return data

def simpan_grid_best(nm_file, nm_kolom, jml_data, latih, uji, kom_param, batch_size, epochs, dropout_rate, hidden_unit, learning_rate, optimizer, best_score, mae, mape, waktu):
    """Save results to the grid_best table."""
    conn = get_connection()
    try:
        c = conn.cursor()

        c.execute("SELECT COUNT(*) FROM grid_best")
        result = c.fetchone()
        count = result[0]

        if count == 0:
            c.execute("ALTER TABLE grid_best AUTO_INCREMENT = 1")
            conn.commit()
        query = """INSERT INTO grid_best (nm_file, nm_kolom, jml_data, latih, uji, kom_param,batch_size, epochs, dropout_rate, hidden_unit, learning_rate, optimizer, best_score, mae, mape, waktu)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"""
        values = (nm_file, nm_kolom, jml_data, latih, uji, kom_param, batch_size, epochs, dropout_rate, hidden_unit, learning_rate, optimizer, best_score, mae, mape, waktu)
        c.execute(query, values)
        conn.commit()
    finally:
        c.close()
        conn.close()

def simpan_parameter(nm_file, nm_kolom, jml_data, latih, uji, batch_size, epoch, dropout_rate, hidden_unit, learning_rate, optimizer, mae, mape, waktu):
    """Save parameters to the hasil_kombinasi table."""
    conn = get_connection()
    try:
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM hasil_kombinasi")
        result = c.fetchone()
        count = result[0]

        if count == 0:
            c.execute("ALTER TABLE hasil_kombinasi AUTO_INCREMENT = 1")
            conn.commit()
       
        query = """INSERT INTO hasil_kombinasi (nm_file, nm_kolom, jml_data, latih, uji, batch_size, epoch, dropout_rate, hidden_unit, learning_rate, optimizer, mae, mape, waktu)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"""
        values = (nm_file, nm_kolom, jml_data, latih, uji, batch_size, epoch, dropout_rate, hidden_unit, learning_rate, optimizer, mae, mape, waktu)
        c.execute(query, values)
        conn.commit()
    finally:
        c.close()
        conn.close()


def simpan_hasil_prediksi(bulan, data, prediksi):
    conn = get_connection()
    try:
        # Membuat cursor
        c = conn.cursor()

        # Mengecek jumlah data di dalam tabel hasil_prediksi
        c.execute("SELECT COUNT(*) FROM hasil_prediksi WHERE data = %s", (data,))
        result = c.fetchone()
        count = result[0]

        # Jika ada data, hapus semua data dengan data yang sesuai
        if count > 0:
            c.execute("DELETE FROM hasil_prediksi WHERE data = %s", (data,))
            conn.commit()

        # Mengatur auto increment dari 1 jika tabel kosong
        c.execute("ALTER TABLE hasil_prediksi AUTO_INCREMENT = 1")

        # Menyimpan hasil prediksi ke dalam tabel
        query = 'INSERT INTO hasil_prediksi (bulan, data, prediksi) VALUES (%s, %s, %s)'
        values = [(str(b), data, float(prediksi[i])) for i, b in enumerate(bulan)]
        c.executemany(query, values)
        
        conn.commit()  
    finally:
        c.close()
        conn.close()
        print("Data berhasil disimpan.")

def tampil_hasil_predikisi():
    """Fetch all data from the hasil_prediksi table."""
    conn = get_connection()
    try:
        c = conn.cursor()
        c.execute('SELECT bulan,data, prediksi FROM hasil_prediksi')
        data = c.fetchall()
    finally:
        c.close()
        conn.close()
    return data

# LOGIN
def hash_password(password):
    """Hash a password using SHA-256."""
    return hashlib.sha256(password.encode()).hexdigest()

def add_user(nama, username, password):
    """Add a new user to the users table."""
    conn = get_connection()
    try:
        c = conn.cursor()
        c.execute('INSERT INTO users (nama, username, password) VALUES (%s, %s, %s)', (nama, username, hash_password(password)))
        conn.commit()
        return True
    except mysql.connector.IntegrityError:
        return False
    finally:
        c.close()
        conn.close()

def authenticate(username, password):
    """Authenticate a user by checking the username and password."""
    conn = get_connection()
    try:
        c = conn.cursor()
        c.execute('SELECT password FROM users WHERE username = %s', (username,))
        user = c.fetchone()
        if user and user[0] == hash_password(password):
            return True
        return False
    finally:
        c.close()
        conn.close()
