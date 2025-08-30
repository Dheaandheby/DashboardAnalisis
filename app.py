import streamlit as st
import pandas as pd
import pickle
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# ========== Load Data & Model ==========
@st.cache_data
def load_data():
    df = pd.read_csv("D:\KP\dashboard sentimen\cleaned_reviews.csv")
    return df

df = load_data()

# Load model & vectorizer
try:
    model = pickle.load(open("models/naive_bayes.pkl", "rb"))
    vectorizer = pickle.load(open("models/tfidf.pkl", "rb"))
except:
    model, vectorizer = None, None

# ========== Sidebar Menu ==========
st.sidebar.title("Menu")
menu = st.sidebar.radio("Pilih Tampilan:", ["Komentar", "Grafik", "Analisis", "Insight & Rekomendasi"])

# ========== Judul ==========
st.title("📊 Dashboard Sentimen Analisis")

# ========== Menu 1: Komentar ==========
if menu == "Komentar":
    st.subheader("💬 Data Review Play Store")
    st.dataframe(df[['userName', 'content', 'score', 'sentiment']])

# ========== Menu 2: Grafik ==========
elif menu == "Grafik":
    st.subheader("📈 Distribusi Sentimen")
    st.bar_chart(df['sentiment'].value_counts())

    st.subheader("📊 Perbandingan Jumlah Sentimen")
    fig, ax = plt.subplots()
    df['sentiment'].value_counts().plot(kind='pie', autopct='%1.1f%%', ax=ax)
    ax.set_ylabel('')
    st.pyplot(fig)

# ========== Menu 3: Analisis ==========
elif menu == "Analisis":
    st.subheader("☁ WordCloud Sentimen")

    col1, col2 = st.columns(2)

    with col1:
        st.write("Wordcloud Positif")
        text_pos = " ".join(df[df['sentiment']=="positif"]['clean'])
        wc = WordCloud(width=400, height=300).generate(text_pos)
        plt.imshow(wc, interpolation='bilinear')
        plt.axis("off")
        st.pyplot(plt)

    with col2:
        st.write("Wordcloud Negatif")
        text_neg = " ".join(df[df['sentiment']=="negatif"]['clean'])
        wc = WordCloud(width=400, height=300).generate(text_neg)
        plt.imshow(wc, interpolation='bilinear')
        plt.axis("off")
        st.pyplot(plt)

    st.subheader("📉 Tren Sentimen Berdasarkan Rating")
    fig, ax = plt.subplots()
    df.groupby(['score'])['sentiment'].count().plot(kind='bar', ax=ax)
    st.pyplot(fig)

# ========== Menu 4: Insight & Rekomendasi ==========
elif menu == "Insight & Rekomendasi":
    st.subheader("🔎 Insight")
    total_review = len(df)
    positif = len(df[df['sentiment']=="positif"])
    negatif = len(df[df['sentiment']=="negatif"])
    netral = len(df[df['sentiment']=="netral"])

    st.write(f"📌 Total review yang dianalisis: **{total_review}**")
    st.write(f"✅ Review Positif: **{positif}**")
    st.write(f"❌ Review Negatif: **{negatif}**")
    st.write(f"➖ Review Netral: **{netral}**")

    st.subheader("💡 Rekomendasi untuk Samsat")
    st.markdown("""
    - Meningkatkan **stabilitas aplikasi SIGnal**, karena review negatif biasanya terkait bug dan error.
    - Menyediakan **panduan penggunaan lebih jelas** agar pengguna mudah melakukan pembayaran.
    - Memperkuat **fitur notifikasi & reminder** agar wajib pajak tidak terlambat.
    - Menjawab feedback negatif secara cepat agar kepuasan pengguna meningkat.
    """)

# ========== Prediksi Review Baru ==========
st.sidebar.subheader("🔎 Coba Prediksi Sentimen")
user_input = st.sidebar.text_area("Tulis review di sini...")
if st.sidebar.button("Prediksi"):
    if model and vectorizer:
        X_input = vectorizer.transform([user_input])
        prediction = model.predict(X_input)[0]
        st.sidebar.write(f"Hasil prediksi: **{prediction.upper()}**")
    else:
        st.sidebar.write("⚠ Model belum tersedia. Latih model dulu.")
