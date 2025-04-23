import pandas as pd
from sklearn.preprocessing import LabelEncoder

class FeatureEngineer:
    def __init__(self):
        self.label_encoders = {}
        self.one_hot_cols = []

    def match_exact(self, a, b):
        if pd.isna(a) or pd.isna(b):
            return 0
        return int(str(a).strip().lower() == str(b).strip().lower())

    def add_custom_features(self, df):
        df['match_education_level'] = df.apply(lambda x: self.match_exact(x['vacancy_education_level'], x['candidate_academic_level']), axis=1)
        df['match_english_level'] = df.apply(lambda x: self.match_exact(x['vacancy_english_level'], x['candidate_english_level']), axis=1)
        df['match_spanish_level'] = df.apply(lambda x: self.match_exact(x['vacancy_spanish_level'], x['candidate_spanish_level']), axis=1)
        df['match_pcd'] = df.apply(lambda x: int(x['vacancy_pcd'] == 'Sim' and x['candidate_pcd'] == 'Sim'), axis=1)

        sp_ddds = ['11', '12', '13', '14', '15', '16', '17', '18', '19']
        df['mobile_region_match'] = df.apply(lambda x: int(str(x['vacancy_region']).lower() == 'são paulo' and str(x['candidate_ddd_mobile']) in sp_ddds), axis=1)

        return df

    def encode_features(self, df):
        df = df.copy()
        df.drop(columns=['prospect_application_date', 'candidate_ddd_mobile'], inplace=True, errors='ignore')

        relevant_cols_for_onehot = [
            'vacancy_contract_type', 'vacancy_sap', 'vacancy_region', 'vacancy_english_level',
            'vacancy_professional_level', 'vacancy_education_level', 'vacancy_spanish_level',
            'vacancy_pcd', 'prospect_candidate_status', 'candidate_academic_level',
            'candidate_english_level', 'candidate_spanish_level', 'candidate_pcd'
        ]

        for col in relevant_cols_for_onehot:
            if col in df.columns:
                n_unique = df[col].nunique()
                if n_unique > 2:
                    self.one_hot_cols.append(col)
                else:
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col].astype(str))
                    self.label_encoders[col] = le
            else:
                print(f"Coluna '{col}' não encontrada no dataframe.")

        df = pd.get_dummies(df, columns=self.one_hot_cols)
        dummy_cols = [col for col in df.columns if any(prefix in col for prefix in self.one_hot_cols)]
        df[dummy_cols] = df[dummy_cols].astype(int)

        return df

    def preprocess(self, df):
        df = self.add_custom_features(df)
        df = self.encode_features(df)
        return df
