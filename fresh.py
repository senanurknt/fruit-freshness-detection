import streamlit as st
from PIL import Image
import numpy as np

st.set_page_config(page_title="Meyve Tazelik Tespiti", page_icon="🍎", layout="centered")

st.title("🍎 Meyve Tazelik Tespit Sistemi")
st.write("Yapay zeka modelimiz taze ve çürük meyveleri tespit eder.")

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
        
        # Modeli sadece resim geldiğinde yüklüyoruz
        model = tf.keras.models.load_model("meyve_tazelik_modeli.keras")
        
        # Resmi hazırla
        img_resized = image.resize((128, 128))
        img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
        img_array = tf.expand_dims(img_array, 0)
        
        # Tahmin yap
        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])
        predicted_class = class_names[np.argmax(score)]
        confidence = float(100 * np.max(score))
        
    st.divider()
    sonuc_metni = turkce_etiketler.get(predicted_class, predicted_class)
    
    if "fresh" in predicted_class:
        st.success(f"### Sonuç: {sonuc_metni} ✨")
    else:
        st.error(f"### Sonuç: {sonuc_metni} ⚠️")
        
    st.info(f"**Modelin Güven Oranı:** %{confidence:.2f}")