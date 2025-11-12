import streamlit as st
from TTS.api import TTS
import os

# Model ka naam (XTTS v2)
# Streamlit Cloud par pehli baar run hone par yeh model download hoga (yeh bada ho sakta hai)
model_name = "tts_models/multilingual/multi-dataset/xtts_v2"

# Model ko load karna (GPU par run karne ke liye cuda=True)
# Streamlit Cloud ke free tier par GPU nahi hota, isliye cuda=False set karein
# @st.cache_resource ko istemal kar rahe hain taake model baar baar load na ho
@st.cache_resource
def load_model():
    try:
        # Check karein agar GPU available hai ya nahi
        # Yeh locally test karne ke liye faidemand hai
        import torch
        use_cuda = torch.cuda.is_available()
    except ImportError:
        use_cuda = False
        
    print(f"Loading model '{model_name}'. CUDA available: {use_cuda}")
    model = TTS(model_name, gpu=use_cuda)
    return model

try:
    tts_model = load_model()
    st.success(f"XTTS Model '{model_name}' successfully loaded!")
except Exception as e:
    st.error(f"Model load karne mein error aaya: {e}")
    st.stop()


# --- Streamlit UI ---

st.title("Coqui XTTS Voice Cloning App 🎙️")
st.write("Instant Voice Cloning ke liye Coqui XTTS v2 model ka istemal karein.")

st.markdown("""
**Kaise istemal karein:**
1.  **Reference Audio Upload:** Ek audio file (e.g., .wav, .mp3) upload karein jiski awaz aap clone karna chahte hain. (Behtareen results ke liye 10-30 seconds ki clear audio istemal karein).
2.  **Text Likhein:** Woh text likhein jo aap is awaz mein bulwana chahte hain.
3.  **Generate:** 'Generate Audio' button par click karein.
""")

# 1. Reference Audio Uploader
uploaded_file = st.file_uploader("Reference Audio File Upload Karein (WAV/MP3)", type=["wav", "mp3"])

# 2. Text Input
text_to_speak = st.text_area("Yahan text likhein:", "Hello, this is a test of instant voice cloning with XTTS model.")

# 3. Generate Button
if st.button("Generate Audio"):
    if uploaded_file is not None and text_to_speak:
        with st.spinner("Processing... model awaz generate kar raha hai..."):
            try:
                # Uploaded file ko temporarily save karein
                # XTTS ko file path ki zaroorat hoti hai
                with open("temp_reference.wav", "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                reference_audio_path = "temp_reference.wav"
                output_audio_path = "output_clone.wav"

                # Debugging info
                st.info(f"Reference audio saved to: {reference_audio_path}")
                st.info(f"Text to speak: {text_to_speak}")

                # --- Voice Cloning ---
                # XTTS model ko call karein
                # `speaker_wav` reference audio ka path hai
                tts_model.tts_to_file(
                    text=text_to_speak,
                    speaker_wav=reference_audio_path,
                    language="en", # Language code set karein (e.g., "en", "es", "hi")
                    file_path=output_audio_path
                )

                # --- Result ---
                st.success("Audio successfully generate ho gayi!")
                
                # Audio player dikhayein
                audio_file = open(output_audio_path, 'rb')
                audio_bytes = audio_file.read()
                st.audio(audio_bytes, format='audio/wav')
                
                # Temporary files ko delete karein
                os.remove(reference_audio_path)
                os.remove(output_audio_path)

            except Exception as e:
                st.error(f"Audio generation mein error aaya: {e}")
                # Agar error aaye toh bhi temp files delete karne ki koshish karein
                if os.path.exists("temp_reference.wav"):
                    os.remove("temp_reference.wav")
                if os.path.exists("output_clone.wav"):
                    os.remove("output_clone.wav")

    elif not uploaded_file:
        st.warning("Please pehle reference audio file upload karein.")
    else:
        st.warning("Please text area mein kuch likhein.")

st.sidebar.header("About")
st.sidebar.info("Yeh app [Coqui XTTS](https://github.com/coqui-ai/TTS) model ka istemal karta hai jo Streamlit par deploy kiya gaya hai.")
