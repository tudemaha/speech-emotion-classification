# import necessary modules
import streamlit as st
from pandas import DataFrame

# import app module (controller and view)
from controller.checking import checking
from view.config import get_config

# get config function to start page config (for each page)
get_config("Checking")

# start the ckecking page
def start():
    # show the page title
    st.write("<h1 style='text-align: center;'>Let's Check Your Emotion!</h1>", unsafe_allow_html = True)

    # show instructions for user
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

    # show file uploader to upload speech audio
    uploaded_files = st.file_uploader("Input Audio :sound:", ['wav', 'avi', 'aac', 'mp3'], accept_multiple_files = True)

    # if user upload the files
    if uploaded_files != []:
        # show the uploaded files with their name and audio player
        st.write("#### Uploaded Files")
        for uploaded_file in uploaded_files:
            col1, col2 = st.columns([0.2, 0.8])
            col1.write(uploaded_file.name)
            col2.audio(uploaded_file)

        # if user click the process button
        if st.button("Process", type = "primary"):
            # show the predicton result
            st.write("#### Predicted Emotions")
            # start prediction process
            raw, pred = checking(uploaded_files)
            # show the result (if sad show snow, if happy show balloons)
            if 0 in pred: st.snow()
            if 1 in pred: st.balloons()

            # prepare the emoji dictionary
            emoji = {"0": "🥲", "1": "😄"}
            # prepare the result list
            result = []
            # append the result to the list with their filename and predicted emotion's emoji
            for i, p in enumerate(pred):
                result.append((uploaded_files[i].name, emoji[str(p)], "{:.2f}%".format(raw[i][p] * 100)))
            
            # show the result in dataframe
            result_df = DataFrame(result, columns = ["Filename", "Emotion", "Persentage"])
            st.dataframe(result_df, 400)

# if the traing process has not been done
def show_warning():
    # show warning message for user to go to machine modeling page
    st.warning("Model not created yet! Go to \"Machine Modeling\" page, do preprocessing, training, and testing.", icon = "⚠️")

# check if the model has been tested
if st.session_state["test"]:
    # start the checking page
    start()
else:
    # show warning message
    show_warning()

# code by @tudemaha