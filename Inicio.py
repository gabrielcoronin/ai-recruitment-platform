import streamlit as st
import warnings

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Plataforma de Análise com IA", layout='wide')

st.markdown("""
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
""", unsafe_allow_html=True)

with open('assets/css/style.css') as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

with st.container():
    st.markdown("""
        <h1 style="text-align: center; color: #E1E1E1; font-size: 42px; margin-bottom: 10px;">
            Plataforma de Análise com IA
        </h1> 
    """, unsafe_allow_html=True)
    st.markdown("---")


with st.container():
    st.subheader("Contexto")
    st.markdown("""
    A **Decision IT** é uma consultoria especializada na alocação de talentos em tecnologia da informação. Seu foco é conectar rapidamente os melhores profissionais do mercado às necessidades específicas de seus clientes.

    Atualmente, esse processo ainda depende fortemente de análise humana de currículos — o que pode causar gargalos operacionais, vieses subjetivos e retrabalho.
    """)

with st.container():
    st.subheader("Desafio do Projeto")
    st.markdown("""
    Participamos do **Datathon Decision** com o desafio de construir uma solução baseada em inteligência artificial capaz de:

    > Automatizar e otimizar o ranqueamento de candidatos para diferentes vagas da empresa.

    Utilizando dados históricos de vagas e aprovações, nossa missão foi responder à pergunta:
    """)
    st.info("**Como prever, com base em dados históricos, quais candidatos têm maior probabilidade de serem aprovados para uma vaga?**", icon="💡")

with st.container():
    st.subheader("Abordagem")
    st.markdown("""
    A solução foi estruturada em três grandes etapas:

    1. **Exploração e limpeza dos dados:** tratamento de textos e padronização de campos para criar uma base confiável.
    2. **Engenharia de atributos:** extração de variáveis como tecnologias citadas, tempo de experiência e aderência ao perfil da vaga.
    3. **Modelagem preditiva e ranqueamento:** experimentamos modelos como **XGBoost**, **Random Forest** e **Gradient Boosting**, avaliando sua performance com métricas robustas.

    Transformamos dados desestruturados em previsões valiosas para o processo de recrutamento.
    """)

with st.container():
    st.subheader("Principais Insights")
    st.markdown("""
    Durante os testes, obtivemos resultados bastante promissores:
    
    - Modelos como **XGBoost** apresentou **AUC-ROC acima de 0.60**, indicando excelente capacidade de generalização.
    - Rankings gerados pelo modelo mostraram diferenças claras de performance por faixa:
    """)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.success("**Top 100 candidatos**: 51% de taxa de aprovação", icon="✅")
    with col2:
        st.warning("**Top 1000 candidatos**: 15% de taxa de aprovação", icon="⚠️")
    with col3:
        st.error("**Sem o modelo**: apenas 5% de taxa de aprovação", icon="🚨")

    st.markdown("Esses resultados reforçam o poder de um ranqueamento bem calibrado para acelerar a triagem de talentos com alto potencial.")
    st.markdown("### 📊 Gráficos de Desempenho dos Modelos")

    col3, col4 = st.columns(2)
    with col3:
        st.image("outputs/figures/roc_curve_xgboost.png", caption="Curva ROC - XGBoost", use_container_width =True)
    with col4:
        st.image("outputs/figures/precision_K_xgboost.png", caption="Precision@K - XGBoost", use_container_width =True)

with st.container():
    st.subheader("O que você verá neste app?")
    st.markdown("""
    Este aplicativo interativo apresenta toda a jornada do projeto — do desafio inicial aos resultados — e ainda permite que você **teste seus próprios dados** com o modelo treinado.

    Explore as páginas no menu lateral:

    - 📊 **Processo de análise**: gráficos, métricas e comparação entre algoritmos.
    - 📈 **Resultados dos Rankings**: análise por faixa de candidatos e decisões.
    - 🧪 **Teste com seus dados**: envie seu CSV e veja os candidatos ranqueados.
    """)
