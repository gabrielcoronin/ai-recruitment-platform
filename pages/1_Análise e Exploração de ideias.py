import streamlit as st

from pages.tabs.analysis.data_exploration import DataExplorationTab
from pages.tabs.analysis.feature_engineering import FeatureEngineeringTab
from pages.tabs.analysis.model_training import ModelTrainingTab
from pages.tabs.analysis.pre_processing import PreProcessingTab

st.set_page_config(page_title="Exploração e Ideias", layout='wide')

# Estilo global
with open('assets/css/style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Título e introdução
with st.container():
    st.markdown("""
        <h1 class="main-title" style="text-align: center;">
            Exploração e Ideias
        </h1>
    """, unsafe_allow_html=True)

    st.markdown("""
        <p class="section-text" style="text-align: center; font-size: 18px; color: #555; max-width: 850px; margin: 0 auto; padding-top: 10px;">
            Aqui você acompanha nossa jornada de desenvolvimento da solução de IA para ranqueamento de candidatos.
            Explore cada etapa do pipeline — da análise exploratória até o treinamento dos modelos.
        </p>
    """, unsafe_allow_html=True)

    st.markdown("<hr class='separator'>", unsafe_allow_html=True)

# Abas com o pipeline
tab0, tab1, tab2, tab3 = st.tabs([
    "Pré-processamento",
    "Análise Exploratória",
    "Feature Engineering",
    "Treinamento dos Modelos"
])

# Chamadas dos módulos
PreProcessingTab(tab0)
DataExplorationTab(tab1)
FeatureEngineeringTab(tab2)
ModelTrainingTab(tab3)
