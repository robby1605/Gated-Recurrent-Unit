import streamlit as st
from streamlit_option_menu import option_menu
from proses.grid import *
from home import home
from kombinasi import proses_kombinasi
from config.conn import authenticate, add_user
from latih import *
from history import tampil_history
from prediksi import *

# Set page configuration
st.set_page_config(page_title="Gated Recurrent Unit", page_icon="😊", layout="wide")

def handle_authentication():
    """Handle user authentication and registration."""
    with st.sidebar:
        choice = option_menu(
            menu_title="Menu", 
            options=["Login", "Registrasi"],
            icons=["door-open", "person"],
            menu_icon="house",
            default_index=0,
            orientation="vertical",
        )
    # choice = st.sidebar.selectbox("Menu", sel)
    
    if choice == "Login":
        st.header("LOGIN")
        login()
    elif choice == "Registrasi":
        st.header("REGISTRASI")
        register()

def login():
    """Handle user login."""
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    
    if st.button("Login"):
        if authenticate(username, password):
            st.session_state.authenticated = True
            st.experimental_rerun()
        else:
            st.error("Username atau password salah")

def register():
    """Handle user registration."""
    nama = st.text_input("Nama")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    password_confirmation = st.text_input("Konfirmasi Password", type="password")
    
    if st.button("Daftar"):
        if password != password_confirmation:
            st.error("Password dan konfirmasi password tidak cocok")
        elif add_user(nama, username, password):
            st.success("Registrasi berhasil. Silakan login.")
        else:
            st.error("Username sudah ada. Silakan pilih username lain.")

def display_main_content(selected):
    """Display content based on the selected menu item."""
    if selected == "Home":
        home()
    elif selected == "GridSearch":
        proses()
    elif selected == "Kombinasi":
        proses_kombinasi()
    elif selected == "Latih":
        latih_terbaik()
    elif selected == "Prediksi":
        prediksi()
    elif selected == "History":
        tampil_history()
    elif selected == "Logout":
        st.session_state.authenticated = False
        st.experimental_rerun()

def main():
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False

    # Sidebar menu
    with st.sidebar:
        if st.session_state.authenticated:
            selected = option_menu(
                menu_title="G R U", 
                options=["Home", "GridSearch", "Kombinasi", "Latih", "Prediksi", "History", "Logout"],
                icons=["house", "search-heart", "cpu", "arrow-repeat", "gift", "hourglass-split", "box-arrow-left"],
                menu_icon="motherboard",
                default_index=0,
                orientation="vertical",
            )
        else:
            st.sidebar.subheader("APLIKASI FORECASTING")
    
    if not st.session_state.authenticated:
        handle_authentication()
    else:
        display_main_content(selected)

if __name__ == "__main__":
    main()

st.markdown("""
    <style>
        .css-1d391kg { /* Custom class for Streamlit components */
            background-color: #f0f2f6;
            padding: 20px;
            border-radius: 10px;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
        }
        .stButton>button:hover {
            background-color: #45a049;
        }
        .stTextInput>div>div>input {
            border-radius: 5px;
        }
    </style>
""", unsafe_allow_html=True)
