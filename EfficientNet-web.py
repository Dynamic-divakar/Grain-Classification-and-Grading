import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Set page config
st.set_page_config(page_title="Rice Classifier 🌾", layout="wide")

# Load classification model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('models/Resnet50.h5')

model = load_model()
class_labels = ['Arborio', 'Basmati', 'Ipsala', 'Jasmine', 'Karacadag']

# Load quality grading model
@st.cache_resource
def load_grading_model():
    return tf.keras.models.load_model('models/rice_grad_resnet50.h5')

grading_model = load_grading_model()
grading_labels = ['Good', 'Average', 'Poor']

# Preprocess image
def preprocess_image(image):
    image = image.resize((224, 224))
    img_array = np.array(image)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
    return img_array

# Convert background image to base64 and paste it here
base64_bg = "<PASTE_YOUR_BASE64_STRING_HERE>"

# Apply background and glassmorphism
st.markdown(f"""
    <style>
        body {{
            background-image: url("data:image/jpg;base64,{base64_bg}");
            background-size: cover;
            background-attachment: fixed;
            background-position: center;
            background-repeat: no-repeat;
        }}

        .glass {{
            background: rgba(255, 255, 255, 0.2);
            border-radius: 16px;
            padding: 2rem;
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
            color: black;
            margin: 2rem 0;
        }}

        .navbar {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            background-color: #f1f8e9;
            padding: 1rem 2rem;
            border-radius: 10px;
            margin-bottom: 2rem;
            flex-wrap: wrap;
            font-family: 'Segoe UI';
        }}
        .nav-links {{
            display: flex;
            gap: 1.5rem;
            flex-wrap: wrap;
        }}
        .nav-links button {{
            background: none;
            border: none;
            cursor: pointer;
            color: #33691e;
            font-weight: 600;
            font-size: 1rem;
        }}
        .brand {{
            font-size: 1.5rem;
            font-weight: bold;
            color: #33691e;
        }}
        .footer {{
            text-align: center;
            font-size: 0.9rem;
            margin-top: 3rem;
            color: #888;
        }}
    </style>
""", unsafe_allow_html=True)

# Navbar
st.markdown("""
    <div class="navbar">
        <div class="brand">🌾 Rice Grain Classifier</div>
        <div class="nav-links">
            <form action="" method="get">
                <button name="page" value="Home">Home</button>
                <button name="page" value="Services">Services</button>
                <button name="page" value="Prediction">Prediction</button>
                <button name="page" value="Grading">Grading</button>
                <button name="page" value="Contact">Contact</button>
            </form>
        </div>
    </div>
""", unsafe_allow_html=True)

# Navigation logic
query_params = st.query_params
page = query_params.get("page", "Home")

# Page functions
def show_home():
    st.markdown("""
        <div class="glass">
            <h1>Welcome to the Rice Grain Classifier</h1>
            <p>Our system uses <b>image processing</b> and <b>machine learning</b> to classify and grade rice grains.</p>
            <ul>
                <li>📷 Upload rice grain images</li>
                <li>🧠 Classify by type using EfficientNet</li>
                <li>📊 Evaluate quality</li>
            </ul>
            <p><b>Let's revolutionize rice quality analysis!</b></p>
        </div>
    """, unsafe_allow_html=True)

def show_services():
    st.header("🛠️ Our Services")
    st.markdown("""
    - **📷 Rice Grain Classification**
    - **📊 Quality Grading** 
    - **⚡ Fast & Accurate Predictions**
    """)

def show_predict():
    st.header("🔍 Predict Rice Grain Type")
    uploaded_file = st.file_uploader("Upload rice grain image", type=["jpg", "jpeg", "png"], key="prediction_uploader")

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", width=250)

        if st.button("Predict", key="predict_button"):
            with st.spinner("Analyzing image..."):
                processed = preprocess_image(image)

                class_pred = model.predict(processed)
                class_result = class_labels[np.argmax(class_pred)]
                class_conf = np.max(class_pred) * 100

            st.success(f"✅ Type: **{class_result}** ({class_conf:.2f}%)")
            st.progress(int(class_conf))

def show_grading():
    st.header("🏷️ Grade Rice Quality Only")
    grading_file = st.file_uploader("Upload rice grain image for grading", type=["jpg", "jpeg", "png"], key="grading_uploader")

    if grading_file is not None:
        image = Image.open(grading_file)
        st.image(image, caption="Uploaded Image", width=250)

        if st.button("Grade", key="grade_button"):
            with st.spinner("Grading image..."):
                processed = preprocess_image(image)
                grade_pred = grading_model.predict(processed)
                grade_result = grading_labels[np.argmax(grade_pred)]
                grade_conf = np.max(grade_pred) * 100

            st.success(f"🏷️ Quality: **{grade_result}** ({grade_conf:.2f}%)")
            st.progress(int(grade_conf))

def show_contact():
    st.header("📬 Contact Us")
    st.markdown("""
    - 📧 Email: 
      - [tejaswini@gmail.com](mailto:tejaswini@gmail.com)
      - [surekha@gmail.com](mailto:surekha@gmail.com)
      - [divakar@gmail.com](mailto:divakar@gmail.com)
      - [luckyjoy@gmail.com](mailto:luckyjoy@gmail.com)
      - [bobby@gmail.com](mailto:bobby@gmail.com)
    - 📍 Address: GMRIT, Rajam
    """)

# Router
if page == "Home":
    show_home()
elif page == "Services":
    show_services()
elif page == "Prediction":
    show_predict()
elif page == "Grading":
    show_grading()
elif page == "Contact":
    show_contact()
else:
    st.error("Page not found.")

# Footer
st.markdown("""
    <div class='footer'>
        &copy; 2025 Rice Classifier | Designed for agriculture
    </div>
""", unsafe_allow_html=True)
