button_style = """
<style>
div.stButton > button:first-child {
    background-color: rgb(255, 75, 75);
    color: rgb(255, 255, 255);
}
div.stButton > button:hover {
    background-color: rgb(255, 75, 75);
    color: rgb(255, 255, 255);
}
div.stButton > button:active {
    background-color: rgb(255, 75, 75);
    color: rgb(255, 255, 255);
}
div.stButton > button:focus {
    background-color: rgb(255, 75, 75);
    color: rgb(255, 255, 255);
}
.css-1cpxqw2:focus:not(:active) {
    background-color: rgb(255, 75, 75);
    border-color: rgb(255, 75, 75);
    color: rgb(255, 255, 255);
}
"""

style = """
<style>
#MainMenu {
    visibility: hidden;
}
footer {
    visibility: hidden;
}
header {
    visibility: hidden;
}
</style>
"""


def apply_prod_style(st):
    return st.markdown(style, unsafe_allow_html=True)