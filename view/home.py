import streamlit as st
from st_pages import Page, show_pages

from view.config import get_config

def start_home():
    get_config("Home")

    show_pages(
        [
            Page("main.py", "Home", ":house:"),
            Page("view/process.py", "Machine Modeling", ":gear:"),
            Page("view/user.py", "Checking", ":sound:")
        ]
    )