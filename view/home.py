# import necssary modules
from st_pages import Page, show_pages
import streamlit.components.v1 as components

# import config view
from view.config import get_config

# start home function
def start_home():
    # prepare current page config, set the page's title
    get_config("Home")

    # open the html home page to be rendered
    render_page = open("view/index.html")
    # render the html home
    components.html(render_page.read(), height = 1940)

    # create pages (home, machine modeling and checking)
    show_pages(
        [
            Page("main.py", "Home", ":house:"),
            Page("view/process.py", "Machine Modeling", ":gear:"),
            Page("view/user.py", "Checking", ":sound:")
        ]
    )