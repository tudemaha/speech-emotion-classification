from st_pages import Page, show_pages, add_page_title

add_page_title()

show_pages(
    [
        Page("view/process.py", "Machine Modeling", ":gear:"),
        Page("view/user.py", "Checking", ":sound:")
    ]
)