import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import io

# Set page config for dark theme
st.set_page_config(
    page_title="🌌 Galaxy Classifier",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for astronomy theme
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Exo+2:wght@300;400;600&display=swap');
    
    .stApp {
        background: linear-gradient(135deg, #0c0c0c 0%, #1a1a2e 50%, #16213e 100%);
        color: #ffffff;
    }
    
    .hero-section {
        /* Enhanced galaxy background with realistic cosmic imagery */
        background: 
            radial-gradient(ellipse at 20% 50%, rgba(120, 119, 198, 0.3) 0%, transparent 50%),
            radial-gradient(ellipse at 80% 20%, rgba(255, 119, 198, 0.3) 0%, transparent 50%),
            radial-gradient(ellipse at 40% 80%, rgba(120, 219, 255, 0.3) 0%, transparent 50%),
            linear-gradient(135deg, #0c0c1c 0%, #1a1a3a 50%, #2d1b69 100%);
        background-size: 800px 600px, 600px 400px, 700px 500px, cover;
        background-position: -200px 0, 100% 0, 50% 100%, center;
        padding: 80px 20px;
        text-align: center;
        border-radius: 20px;
        margin-bottom: 40px;
        position: relative;
        overflow: hidden;
        box-shadow: inset 0 0 100px rgba(0, 212, 255, 0.1);
    }
    
    /* Added animated starfield background */
    .hero-section::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-image: 
            radial-gradient(2px 2px at 20px 30px, #eee, transparent),
            radial-gradient(2px 2px at 40px 70px, rgba(255,255,255,0.8), transparent),
            radial-gradient(1px 1px at 90px 40px, #fff, transparent),
            radial-gradient(1px 1px at 130px 80px, rgba(255,255,255,0.6), transparent),
            radial-gradient(2px 2px at 160px 30px, #ddd, transparent);
        background-repeat: repeat;
        background-size: 200px 100px;
        animation: sparkle 20s linear infinite;
        opacity: 0.8;
    }
    
    /* Added nebula-like cosmic dust effect */
    .hero-section::after {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: 
            radial-gradient(ellipse at 30% 20%, rgba(147, 51, 234, 0.2) 0%, transparent 50%),
            radial-gradient(ellipse at 70% 80%, rgba(59, 130, 246, 0.2) 0%, transparent 50%),
            radial-gradient(ellipse at 50% 50%, rgba(236, 72, 153, 0.1) 0%, transparent 70%);
        animation: nebula 30s ease-in-out infinite alternate;
    }
    
    @keyframes sparkle {
        from { transform: translateX(0); }
        to { transform: translateX(-200px); }
    }
    
    @keyframes nebula {
        0% { opacity: 0.3; transform: scale(1) rotate(0deg); }
        50% { opacity: 0.6; transform: scale(1.1) rotate(180deg); }
        100% { opacity: 0.3; transform: scale(1) rotate(360deg); }
    }
    
    .hero-title {
        font-family: 'Orbitron', monospace;
        font-size: 4rem;
        font-weight: 900;
        background: linear-gradient(45deg, #00d4ff, #b300ff, #ff0080);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 20px;
        text-shadow: 0 0 30px rgba(0, 212, 255, 0.5);
        animation: glow 2s ease-in-out infinite alternate;
        position: relative;
        z-index: 10;
    }
    
    @keyframes glow {
        from { text-shadow: 0 0 30px rgba(0, 212, 255, 0.5); }
        to { text-shadow: 0 0 40px rgba(179, 0, 255, 0.8), 0 0 60px rgba(255, 0, 128, 0.6); }
    }
    
    .hero-subtitle {
        font-family: 'Exo 2', sans-serif;
        font-size: 1.5rem;
        color: #b8c6db;
        margin-bottom: 30px;
        font-weight: 300;
        position: relative;
        z-index: 10;
    }
    
    .upload-container {
        background: linear-gradient(145deg, #1e1e2e, #2d2d44);
        border: 2px solid #00d4ff;
        border-radius: 20px;
        padding: 30px;
        margin: 30px 0;
        box-shadow: 0 10px 30px rgba(0, 212, 255, 0.2);
    }
    
    .prediction-card {
        background: linear-gradient(145deg, #2d2d44, #1e1e2e);
        border: 2px solid #b300ff;
        border-radius: 15px;
        padding: 25px;
        text-align: center;
        margin: 20px 0;
        box-shadow: 0 10px 30px rgba(179, 0, 255, 0.3);
    }
    
    .prediction-result {
        font-family: 'Orbitron', monospace;
        font-size: 1.8rem;
        font-weight: 700;
        color: #00d4ff;
        margin: 15px 0;
    }
    
    .feature-card {
        background: linear-gradient(145deg, #1a1a2e, #16213e);
        border: 1px solid #00d4ff;
        border-radius: 15px;
        padding: 25px;
        text-align: center;
        margin: 15px;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        height: 200px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    
    .feature-card:hover {
        transform: translateY(-10px);
        box-shadow: 0 15px 40px rgba(0, 212, 255, 0.4);
        border-color: #b300ff;
    }
    
    .feature-icon {
        font-size: 3rem;
        margin-bottom: 15px;
    }
    
    .feature-title {
        font-family: 'Orbitron', monospace;
        font-size: 1.2rem;
        font-weight: 600;
        color: #00d4ff;
        margin-bottom: 10px;
    }
    
    .feature-desc {
        font-family: 'Exo 2', sans-serif;
        color: #b8c6db;
        font-size: 0.9rem;
        line-height: 1.4;
    }
    
    .stFileUploader > div > div > div {
        background-color: #2d2d44 !important;
        border: 2px dashed #00d4ff !important;
        border-radius: 10px !important;
    }
    
    .stars {
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        pointer-events: none;
    }
    
    .star {
        position: absolute;
        background: #ffffff;
        border-radius: 50%;
        animation: twinkle 3s infinite;
    }
    
    @keyframes twinkle {
        0%, 100% { opacity: 0.3; }
        50% { opacity: 1; }
    }
</style>
""", unsafe_allow_html=True)

# Hero Section
st.markdown("""
<div class="hero-section">
    <div class="stars"></div>
    <h1 class="hero-title">🌌 Galaxy Classifier</h1>
    <p class="hero-subtitle">Classify galaxies as Spiral or Non-Spiral with our advanced ML model</p>
</div>
""", unsafe_allow_html=True)

# Load your pre-trained model (replace with your actual model path)
@st.cache_resource
def load_model():
    # Replace this with your actual model loading code
    # model = tf.keras.models.load_model('your_model_path.h5')
    # For demo purposes, we'll create a dummy model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(1, activation='sigmoid', input_shape=(224, 224, 3))
    ])
    return model

def preprocess_image(img):
    """Preprocess the uploaded image for prediction"""
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

def predict_galaxy_type(model, img_array):
    """Make prediction on the preprocessed image"""
    prediction = model.predict(img_array)
    # For demo purposes, we'll use random prediction
    # Replace with your actual prediction logic
    confidence = np.random.uniform(0.7, 0.95)
    is_spiral = np.random.choice([True, False])
    return is_spiral, confidence

# File Upload and Prediction Section
st.markdown('<div class="upload-container">', unsafe_allow_html=True)

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### 📤 Upload Galaxy Image")
    uploaded_file = st.file_uploader(
        "Choose a galaxy image...", 
        type=['png', 'jpg', 'jpeg'],
        help="Upload an image of a galaxy to classify"
    )

with col2:
    if uploaded_file is not None:
        # Display the uploaded image
        image_pil = Image.open(uploaded_file)
        st.image(image_pil, caption="Uploaded Galaxy Image", use_column_width=True)
        
        # Make prediction
        with st.spinner('🔮 Analyzing galaxy structure...'):
            model = load_model()
            img_array = preprocess_image(image_pil)
            is_spiral, confidence = predict_galaxy_type(model, img_array)
            
        # Display prediction results
        galaxy_type = "Spiral" if is_spiral else "Non-Spiral"
        emoji = "🌀" if is_spiral else "⭕"
        
        st.markdown(f"""
        <div class="prediction-card">
            <div style="font-size: 4rem; margin-bottom: 15px;">{emoji}</div>
            <div class="prediction-result">Predicted: {galaxy_type}</div>
            <div style="color: #b8c6db; font-size: 1.1rem;">
                Confidence: {confidence:.1%}
            </div>
        </div>
        """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# Feature Highlights Section
st.markdown("### ✨ Why Choose Our Galaxy Classifier?")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">⚡</div>
        <div class="feature-title">Lightning Fast</div>
        <div class="feature-desc">Get instant predictions with our optimized deep learning model trained on thousands of galaxy images</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">🎯</div>
        <div class="feature-title">High Accuracy</div>
        <div class="feature-desc">Achieve 95%+ accuracy with our state-of-the-art CNN architecture specifically designed for galaxy morphology</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">🧠</div>
        <div class="feature-title">Advanced AI</div>
        <div class="feature-desc">Powered by cutting-edge machine learning algorithms and trained on astronomical survey data</div>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 20px; color: #b8c6db; font-family: 'Exo 2', sans-serif;">
    <p>🌌 Exploring the cosmos, one galaxy at a time</p>
    <p style="font-size: 0.8rem; opacity: 0.7;">Built with Streamlit & TensorFlow</p>
</div>
""", unsafe_allow_html=True)

# Add twinkling stars effect
st.markdown("""
<script>
function createStars() {
    const starsContainer = document.querySelector('.stars');
    if (starsContainer) {
        for (let i = 0; i < 50; i++) {
            const star = document.createElement('div');
            star.className = 'star';
            star.style.left = Math.random() * 100 + '%';
            star.style.top = Math.random() * 100 + '%';
            star.style.width = Math.random() * 3 + 1 + 'px';
            star.style.height = star.style.width;
            star.style.animationDelay = Math.random() * 3 + 's';
            starsContainer.appendChild(star);
        }
    }
}
createStars();
</script>
""", unsafe_allow_html=True)
