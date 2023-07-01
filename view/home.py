from st_pages import Page, show_pages
import streamlit.components.v1 as components

from view.config import get_config

def start_home():
    get_config("Home")

    render_page = open("view/index.html")
    components.html(render_page.read(), height = 1700)

    show_pages(
        [
            Page("main.py", "Home", ":house:"),
            Page("view/process.py", "Machine Modeling", ":gear:"),
            Page("view/user.py", "Checking", ":sound:")
        ]
    )