import streamlit as st

def get_config(page_title):
    title = "Speech Clasification Using Neural Network"
    st.set_page_config(layout='wide', page_title=f"{page_title} | {title}", menu_items={
            'About': f"""
            ### {title}
            Made with :heart: by Group 1 Class A  
            Introduction to Multimedia Data Processing Subject  
            GitHub: https://github.com/tudemaha/speech-emotion-classification
            """
        })