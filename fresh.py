import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

st.set_page_config(page_title="Meyve Tazelik Tespiti", page_icon="🍎", layout="centered")

st.title("🍎 Meyve Tazelik Tespit Sistemi")
st.write("Yapay zeka modeli ile meyvelerin tazelik ve çürüklük durumunu analiz edin.")

# Modeli önbelleğe alarak hızlı yükleme
@st.cache_resource
def load_fruit_model():
    return tf.keras.models.load_model("meyve_tazelik_modeli.keras")

model = load_fruit_model()


CLASS_NAMES = [
    'freshapples',
    'freshbanana',
    'freshoranges',
    'rottenapples',
    'rottenbanana',
    'rottenoranges'
]

TURKCE_ISIMLER = {
    'freshapples': '🍏 Taze Elma',
    'freshbanana': '🍌 Taze Muz',
    'freshoranges': '🍊 Taze Portakal',
    'rottenapples': '🍎 Çürük / Bayat Elma',
    'rottenbanana': '🍌 Çürük / Bayat Muz',
    'rottenoranges': '🍊 Çürük / Bayat Portakal'
}

uploaded_file = st.file_uploader("Bir meyve fotoğrafı yükleyin...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    raw_image = Image.open(uploaded_file)
    
    # PNG formatındaki şeffaf arka planları beyaza çevir
    if raw_image.mode in ("RGBA", "P"):
        raw_image = raw_image.convert("RGBA")
        background = Image.new("RGB", raw_image.size, (255, 255, 255))
        background.paste(raw_image, mask=raw_image.split()[3])
        image = background
    else:
        image = raw_image.convert("RGB")

    st.image(image, caption="Yüklenen Görsel", use_container_width=True)

    with st.spinner("Model fotoğrafı analiz ediyor..."):
        # Görseli 128x128 boyutuna getir ve modele ver
        img_resized = image.resize((128, 128))
        img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
        img_array = np.expand_dims(img_array, axis=0)

        # Tahmin üret
        predictions = model.predict(img_array)
        probabilities = tf.nn.softmax(predictions[0]).numpy()
        
        predicted_idx = int(np.argmax(probabilities))
        selected_class = CLASS_NAMES[predicted_idx]
        confidence = float(probabilities[predicted_idx]) * 100

    # Sonuç kartı
    st.success(f"### Sonuç: {TURKCE_ISIMLER.get(selected_class, selected_class)} ✨")
    st.info(f"**Modelin Güven Oranı:** %{confidence:.2f}")

    with st.expander("📊 Tüm Sınıf Dağılımını İncele"):
        for idx, name in enumerate(CLASS_NAMES):
            st.write(f"- **{TURKCE_ISIMLER[name]}:** %{float(probabilities[idx])*100:.2f}")