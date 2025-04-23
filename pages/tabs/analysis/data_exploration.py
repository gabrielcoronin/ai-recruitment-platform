import streamlit as st
from pages.tabs.tab import TabInterface
import nbformat
from nbconvert import HTMLExporter

class DataExplorationTab(TabInterface):
    def __init__(self, tab):
        self.tab = tab
        self.render()
    
    def notebook_to_html(self, notebook_path):
        try:
            with open(notebook_path, "r", encoding="utf-8") as f:
                notebook = nbformat.read(f, as_version=4)
            
            html_exporter = HTMLExporter()
            body, _ = html_exporter.from_notebook_node(notebook)
            return body
        except Exception as e:
            return f"Error: {e}"

    def render(self):
        with self.tab:
            st.components.v1.html(self.notebook_to_html("notebooks/02_data_exploration.ipynb"), height=800, scrolling=True)
