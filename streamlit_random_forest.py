import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Streamlit App
st.title("Klasifikasi Obesitas dengan Random Forest")

# File uploader
uploaded_file = st.file_uploader("Unggah file CSV", type="csv")

if uploaded_file is not None:
    # Load dataset
    df = pd.read_csv(uploaded_file)
    st.write("## Data Awal")
    st.dataframe(df.head())

    # Display dataset info
    st.write("### Info Dataset")
    buffer = pd.io.common.StringIO()
    df.info(buf=buffer)
    s = buffer.getvalue()
    st.text(s)

    # Correlation matrix
    st.write("### Matriks Korelasi")
    correlation_matrix = df.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', ax=ax)
    st.pyplot(fig)

    # Select target column
    st.write("### Pilih Kolom Target")
    target_column = st.selectbox("Pilih kolom target untuk klasifikasi:", df.columns)
    
    if target_column:
        try:
            # Preprocessing
            st.write("### Preprocessing Data")
            st.write("Data kategori akan dikonversi menjadi numerik jika diperlukan.")

            # Dummy variable creation
            df = pd.get_dummies(df, drop_first=True)

            # Splitting data
            X = df.drop(columns=[target_column], axis=1)
            y = df[target_column]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            st.write("Data berhasil dibagi menjadi data latih dan data uji.")

            # Model training
            st.write("### Pelatihan Model Random Forest")
            rf_model = RandomForestClassifier(random_state=42)
            rf_model.fit(X_train, y_train)

            # Prediction
            y_pred = rf_model.predict(X_test)

            # Evaluation
            accuracy = accuracy_score(y_test, y_pred)
            st.write("### Evaluasi Model")
            st.write(f"Akurasi: {accuracy:.2f}")

            st.write("#### Laporan Klasifikasi")
            st.text(classification_report(y_test, y_pred))

            st.write("#### Matriks Kebingungan")
            fig, ax = plt.subplots()
            sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', ax=ax)
            st.pyplot(fig)

        except KeyError as e:
            st.error(f"Error: Kolom '{e.args[0]}' tidak ditemukan dalam data.")
        except Exception as e:
            st.error(f"Terjadi kesalahan: {e}")

else:
    st.write("Silakan unggah file CSV untuk memulai.")
