import streamlit as st

st.set_page_config(page_title="Resultados Detalhados", layout='wide')

with open("assets/css/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

with st.container():
    st.markdown("""
        <h1 class="main-title" style="text-align: center;">
            Resultados detalhados do Modelo
        </h1>
    """, unsafe_allow_html=True)

st.markdown("---")

with st.container():
    st.subheader("Por que escolhemos o XGBoost?")
    st.markdown("""
    Após testar diferentes algoritmos, como **Random Forest** e **Gradient Boosting**, o modelo que apresentou o melhor desempenho foi o **XGBoost (Extreme Gradient Boosting)**.

    Ele foi escolhido por diversos motivos:

    - **Alta performance em tarefas tabulares**: especialmente eficaz com bases com mix de variáveis categóricas e numéricas.
    - **Velocidade de treinamento**: é otimizado para performance e aproveita paralelismo.
    - **Interpretação dos resultados**: permite extrair importância das variáveis.
    - **Regularização embutida**: previne overfitting automaticamente.
    - **Robustez**: mantém boa performance mesmo com dados com ruído ou colinearidade residual.

    Combinado à nossa engenharia de atributos, o XGBoost se mostrou ideal para o desafio de prever aprovações com base em múltiplas variáveis, muitas delas extraídas de texto (como tecnologias e experiência).
    """)

st.markdown("---")


with st.container():
    st.subheader("Métricas de Avaliação")
    st.markdown("""
    Avaliamos os modelos com métricas robustas de classificação, como AUC-ROC e Precision@K. Os resultados mostram que o modelo consegue distinguir bem os candidatos com maior potencial de aprovação.
    """)

    col1, col2 = st.columns(2)
    with col1:
        st.image("outputs/figures/roc_curve_xgboost.png", caption="Curva ROC - XGBoost", use_container_width=True)
    with col2:
        st.image("outputs/figures/precision_K_xgboost.png", caption="Precision@K - XGBoost", use_container_width=True)

# Insights
with st.container():
    st.subheader("Insights do Ranqueamento")
    st.markdown("""
    O modelo não apenas classifica os candidatos, como também ranqueia de acordo com a probabilidade de sucesso. Isso nos permitiu observar padrões claros por faixa:

    - ✅ **Top 100 candidatos**: 51% de taxa de aprovação.
    - ⚠️ **Top 1000 candidatos**: 15% de taxa de aprovação.
    - ❌ **Sem modelo (aleatório)**: apenas 5% de taxa de aprovação.

    Isso representa um ganho **de até 10x na assertividade** de aprovação no topo do ranking!
    """)

st.markdown("---")





with st.container():
    st.subheader("Agenda futura")
    st.markdown("""
    Nosso modelo demonstrou resultados promissores e abre oportunidade para novas funcionalidades que podem aumentar ainda mais a eficácia do processo de recrutamento. Como agenda futura e próximos passos, podemos destacar as seguintes ações:
    
    - Aprimoramento do ranking com palavras-chave: integrar técnicas de NLP (Processamento de Linguagem Natural) para identificar e comparar automaticamente tecnologias e competências citadas nos currículos com os requisitos das vagas.    
    - Análise de fit cultural: utilizar modelos treinados em dados comportamentais e valores da empresa para avaliar o alinhamento dos candidatos com a cultura organizacional.    
    - Geração automática de perguntas técnicas: com base nas lacunas detectadas entre o perfil do candidato e os requisitos da vaga, a plataforma poderá sugerir perguntas específicas para a entrevista.    
    - Integração com CRMs de recrutamento: permitir que os dados da aplicação se conectem a ferramentas já utilizadas pela Decision, facilitando o uso em ambientes reais.    
    - Manutenção dos modelos: com o uso da solução em produção, o modelo poderá ser constantemente refinado com base no resultado real dos processos seletivos.
    """)
    
st.markdown("---")    
    
with st.container():
    st.subheader("Conclusão")
    st.markdown("""
    A solução desenvolvida representa um avanço significativo no processo de recrutamento da Decision IT. Com base em dados históricos e técnicas modernas de machine learning, conseguimos construir um sistema capaz de:
    
    - Padronizar e automatizar parte do processo seletivo.
    - Aumentar a assertividade na triagem de talentos.
    - Reduzir o tempo e o retrabalho dos recrutadores.
    - Propor uma visão baseada em dados para decisões críticas de contratação.
    
    O uso do modelo XGBoost permitiu a criação de um ranking de candidatos com alta taxa de conversão. Em nossos testes, o top 100 do modelo obteve 51% de taxa de aprovação, um resultado 10 vezes melhor do que abordagens tradicionais aleatórias.
    
    Essa solução não apenas soluciona dores atuais da Decision, como também oferece um caminho concreto para a inovação e escalabilidade de processos seletivos da empresa.
    """)