# import streamlit module
import streamlit as st

# get config function to start page config (for each page)
def get_config(page_title):
    # set app layout, set page title (on tab), set menu items in "about" section
    title = "Speech Clasification Using Neural Network"
    st.set_page_config(layout='wide', page_title=f"{page_title} | {title}", menu_items={
            'About': f"""
            ### {title}
            Made with :heart: by Group 1 Class A  
            Introduction to Multimedia Data Processing Subject  
            GitHub: https://github.com/tudemaha/speech-emotion-classification
            """
        })