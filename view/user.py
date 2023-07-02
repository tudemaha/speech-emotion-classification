import streamlit as st
from pandas import DataFrame

from controller.checking import checking
from view.config import get_config

get_config("Checking")

def start():
    st.write("<h1 style='text-align: center;'>Let's Check Your Emotion!</h1>", unsafe_allow_html = True)


    instructions = """
    Instructions:
    1. Prepare your files
    2. The files must contain human speech
    2. Make sure each file is between 3 to 5 seconds long
    3. Upload the files
    4. Listen to the audio before processing (if necessary)
    5. Process them!
    """
    st.markdown(instructions)

    global uploaded_files
    uploaded_files = st.file_uploader("Input Audio :sound:", ['wav', 'avi', 'aac', 'mp3'], accept_multiple_files = True)

    if uploaded_files != []:
        st.write("#### Uploaded Files")
        for uploaded_file in uploaded_files:
            col1, col2 = st.columns([0.2, 0.8])
            col1.write(uploaded_file.name)
            col2.audio(uploaded_file)

        if st.button("Process", type = "primary"):
            st.write("#### Predicted Emotions")
            pred = checking(uploaded_files)
            if 0 in pred: st.snow()
            if 1 in pred: st.balloons()

            emoji = {"0": "🥲", "1": "😄"}
            result = []
            for i, p in enumerate(pred):
                result.append((uploaded_files[i].name, emoji[str(p)]))
            
            result_df = DataFrame(result, columns = ["Filename", "Emotion"])
            st.dataframe(result_df, 400)

def show_warning():
    st.warning("Model not created yet! Go to \"Machine Modeling\" page, do preprocessing, training, and testing.", icon = "⚠️")

if st.session_state["test"]:
    start()
else:
    show_warning()