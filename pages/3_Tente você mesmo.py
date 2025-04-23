import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from app.utils.feature_engineering import FeatureEngineer
from app.utils.predict import CandidateModelPipeline  

st.set_page_config(page_title="Ranking de Candidatos", layout="wide")
st.title("Ranking Inteligente de Candidatos")

st.markdown("""
Envie um arquivo CSV com os dados dos candidatos ou utilize dados simulados para testar a aplicação.

### Instruções:
- O arquivo CSV deve conter pelo menos **100 candidatos**.
- As colunas devem seguir o seguinte padrão:

| Coluna | Descrição |
|--------|-----------|
| `vacancy_contract_type` | Tipo de contrato da vaga (ex: CLT, PJ) |
| `vacancy_sap` | Vaga exige SAP? (Sim ou Não) |
| `vacancy_region` | Região da vaga (ex: São Paulo, Rio de Janeiro, Minas Gerais) |
| `vacancy_pcd` | Vaga destinada a PCD? (Sim ou Não) |
| `vacancy_professional_level` | Nível da vaga (ex: Pleno, Junior) |
| `vacancy_education_level` | Escolaridade exigida |
| `vacancy_english_level` | Nível de inglês exigido |
| `vacancy_spanish_level` | Nível de espanhol exigido |
| `prospect_candidate_status` | Status do candidato (ex: Aprovado, Rejeitado, etc.) |
| `prospect_application_date` | Data de candidatura |
| `candidate_ddd_mobile` | DDD do celular |
| `candidate_pcd` | Candidato é PCD? (Sim ou Não) |
| `candidate_certifications` | Certificações do candidato |
| `candidate_academic_level` | Escolaridade do candidato |
| `candidate_english_level` | Inglês do candidato |
| `candidate_spanish_level` | Espanhol do candidato |

**Dica:** Use o botão abaixo para baixar um exemplo editável.
""")

example_df = pd.DataFrame({
    "vacancy_contract_type": ["CLT"],
    "vacancy_sap": ["Sim"],
    "vacancy_region": ["São Paulo"],
    "vacancy_pcd": ["Não"],
    "vacancy_professional_level": ["Pleno"],
    "vacancy_education_level": ["Ensino Superior Completo"],
    "vacancy_english_level": ["Avançado"],
    "vacancy_spanish_level": ["Básico"],
    "prospect_candidate_status": ["Em processo seletivo"],
    "prospect_application_date": ["2023-06-15"],
    "candidate_ddd_mobile": [11],
    "candidate_pcd": [0],
    "candidate_certifications": [1],
    "candidate_academic_level": ["Ensino Superior Completo"],
    "candidate_english_level": ["Intermediário"],
    "candidate_spanish_level": ["Básico"],
})

csv_example = example_df.to_csv(index=False).encode("utf-8")
st.download_button("📥 Baixar CSV Exemplo", csv_example, "exemplo_candidatos.csv", "text/csv")

st.divider()

use_mock = st.checkbox("Usar dados simulados")

df = None

if use_mock:
    n = 100
    st.info("🔁 Gerando dados simulados para teste...")
    df = pd.DataFrame({
        "vacancy_contract_type": np.random.choice(["CLT", "PJ"], n),
        "vacancy_sap": np.random.choice(["Sim", "Não"], n),
        "vacancy_region": np.random.choice(["São Paulo", "Pernambuco", "Rio de Janeiro", "Minas Gerais"], n),
        "vacancy_pcd": np.random.choice(["Sim", "Não"], n),
        "vacancy_professional_level": np.random.choice(["Pleno", "Senior", "Junior", "Estágio"], n),
        "vacancy_education_level": np.random.choice(["Ensino Superior Completo", "Pós Graduação Incompleto", "Mestrado", "Doutorado"], n),
        "vacancy_english_level": np.random.choice(["Fluente", "Avançado", "Intermediário", "Básico", "Nenhum"], n),
        "vacancy_spanish_level": np.random.choice(["Fluente", "Avançado", "Intermediário", "Básico", "Nenhum"], n),
        "prospect_candidate_status": np.random.choice(["Em processo seletivo", "Aprovado", "Rejeitado"], n),
        "prospect_application_date": pd.to_datetime(np.random.choice(pd.date_range("2021-01-01", "2023-12-31", freq="D"), n)),
        "candidate_ddd_mobile": np.random.choice([11, 21, 31, 41, 51, 61], n),
        "candidate_pcd": np.random.choice([0, 1], n),
        "candidate_certifications": np.random.choice([0, 1, 2], n),
        "candidate_academic_level": np.random.choice(["Ensino Médio", "Ensino Superior Completo", "Pós Graduação Incompleto", "Mestrado"], n),
        "candidate_english_level": np.random.choice(["Nenhum", "Básico", "Intermediário", "Avançado"], n),
        "candidate_spanish_level": np.random.choice(["Nenhum", "Básico", "Intermediário", "Avançado"], n),
    })
else:
    uploaded_file = st.file_uploader("📎 Envie um arquivo CSV com os dados dos candidatos", type=["csv"])
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file, encoding="utf-8")
            if len(df) < 100:
                st.warning("⚠️ O arquivo deve conter pelo menos 100 candidatos.")
                df = None
            else:
                st.success("✅ Arquivo carregado com sucesso!")
        except Exception as e:
            st.error(f"Erro ao ler o arquivo: {e}")

if df is not None:
    st.subheader("🔍 Pré-visualização dos dados")
    st.dataframe(df.head(100))

    with st.spinner("🔄 Processando dados e gerando ranking..."):
        try:
            engineered_df = FeatureEngineer().preprocess(df)

            target_column = "prospect_candidate_status_Aprovado"
            if target_column not in engineered_df.columns and "prospect_candidate_status" in engineered_df.columns:
                engineered_df[target_column] = (engineered_df["prospect_candidate_status"] == "Aprovado").astype(int)

            pipeline = CandidateModelPipeline()
            model, ranked_candidates = pipeline.run(engineered_df, target_column=target_column, models_dir="models", plot_metrics=True)

            ranked_candidates["status"] = ranked_candidates["approved"].map({1: "Aprovado", 0: "Reprovado"})

            st.subheader("Ranking dos Candidatos")
            filtered_df = ranked_candidates[["approval_probability", "status"]]
            filtered_df["approval_probability"] = filtered_df["approval_probability"].apply(lambda x: f"{x * 100:.2f}%")
            st.dataframe(filtered_df.head(20))

            st.subheader("Distribuição de Aprovados e Reprovados")
            count_df = ranked_candidates["status"].value_counts().reset_index()
            count_df.columns = ["Status", "Quantidade"]
            fig = px.bar(count_df, x="Status", y="Quantidade", color="Status", text="Quantidade", title="Resumo do Ranking")
            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Ocorreu um erro durante o processamento: {e}")
else:
    st.warning("⚠️ Nenhum dado disponível. Selecione uma opção acima.")
