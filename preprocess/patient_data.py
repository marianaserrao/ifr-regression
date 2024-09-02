import pandas as pd
import json

def get_patient_data(patients_excel_path):
    df = pd.read_excel(patients_excel_path)
    df = df[['Nr', 'Sexo', 'Idade', 'iFR_valor', 'FFR_valor', 'Excluir (0_nao_1_sim_2_talvez',]]
    df = df.rename(columns={
        'Nr': "id", 
        'Sexo': "sex", 
        'Idade': "age", 
        'iFR_valor': "ifr", 
        'FFR_valor': "ffr", 
        'Excluir (0_nao_1_sim_2_talvez': "exclude"
    })
    df[['exclude']] = df[['exclude']].fillna(value=0)
    df = df[df['id'].notna()]
    df['id'] = df['id'].astype(int)
    df = df[(df['ifr'].notna()) & (df["exclude"]!=1)]
    df = df.drop_duplicates(subset='id', keep='last')
    df.set_index('id', drop=False, inplace=True)

    return df