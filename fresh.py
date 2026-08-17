import streamlit as st
from PIL import Image
import numpy as np

st.set_page_config(page_title="Meyve Tazelik Tespiti", page_icon="🍎", layout="centered")

st.title("🍎 Meyve Tazelik Tespit Sistemi")
st.write("Yapay zeka modelimiz taze ve çürük meyveleri tespit eder.")

# Kaggle / Keras klasör sıralaması
class_names = ['freshapples', 'freshbanana', 'freshoranges', 'rottenapples', 'rottenbanana', 'rottenoranges']

turkce_etiketler = {
    'freshapples': '🍏 Taze Elma',
    'freshbanana': '🍌 Taze Muz',
    'freshoranges': '🍊 Taze Portakal',
    'rottenapples': '🍎 Çürük / Bayat Elma',
    'rottenbanana': '🍌 Çürük / Bayat Muz',
    'rottenoranges': '🍊 Çürük / Bayat Portakal'
}

uploaded_file = st.file_uploader("Lütfen bir meyve fotoğrafı yükleyin...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Yüklenen Görsel", use_container_width=True)

    with st.spinner("Model yükleniyor ve fotoğraf analiz ediliyor..."):
        import tensorflow as tf

        model = tf.keras.models.load_model("meyve_tazelik_modeli.keras")

        # Resmi hazırla
        img_resized = image.resize((128, 128))
        img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
        img_array = np.expand_dims(img_array, axis=0)
        
        # Piksel normalizasyonu (0-1 aralığı)
        img_array = img_array / 255.0

        # Tahmin yap
        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0]) if np.max(predictions[0]) > 1.0 or np.min(predictions[0]) < 0.0 else predictions[0]

        predicted_class = class_names[np.argmax(score)]
        confidence = float(np.max(score)) * 100

   # Sonuç kartı
    st.success(f"### Sonuç: {turkce_etiketler.get(predicted_class, predicted_class)} ✨")
    st.info(f"**Modelin Güven Oranı:** %{confidence:.2f}")

    # Detaylı tahmin dağılımı (Sıralamayı kontrol etmek için)
    with st.expander("📊 Tüm Sınıf Olasılıklarını Gör"):
        for name, prob in zip(class_names, score):
            st.write(f"- **{turkce_etiketler.get(name, name)} ({name}):** %{float(prob)*100:.2f}")