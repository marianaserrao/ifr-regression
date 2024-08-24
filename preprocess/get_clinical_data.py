import pandas as pd
import json
import os
from pathlib import Path
from tqdm import tqdm
import re

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
    # TODO: manter o que não tiver flag excluir ou o que tiver mais valores ffr ifr
    df = df.drop_duplicates(subset='id', keep='last')
    df.set_index('id', drop=False, inplace=True)

    patients_json = df.to_json(orient='index')
    patients_data = json.loads(patients_json)

    return patients_data

def main():
    patients_excel_path = './clinical_data.xlsx'
    exam_videos_path = '/media/jlsstorage/masstorage/angiograms/Videos'
    exam_masks_path = '/media/jlsstorage/masstorage/angiograms/key_masks'
    output_json = './clinical_data.json'



    def get_kf_exams_data(exam_videos_path, exam_masks_path, patients_data):
        def get_frame_identifiers(file, primary_only=False):
            identifiers = []

            #remove extension
            last_dot_index = file.rfind('.')
            name = file[:last_dot_index]
            if name[-1].isalpha():
                name=name[:-1]

            parts = re.split(r'(?<![_-])-|_', name)
            primary = parts[:4]
            secundary = parts[4:]
            secundary = list(filter(lambda x: x.isdigit(), secundary))[:2]

            if primary_only:
                identifiers+=primary
            else:
                identifiers+=primary+secundary
                
            return "_".join(identifiers)
        
        def get_kf_dict(mask, patient_dir, primary_only_match=False):
            key_frame_identifiers = get_frame_identifiers(mask, primary_only=primary_only_match)           
            key_frame_id = int(mask.split('_')[3])
            key_frame_path = ""

            for root, dirs, files in os.walk(patient_dir):
                for file in files:
                    file_identifiers=get_frame_identifiers(file, primary_only=primary_only_match)
                    if key_frame_identifiers == file_identifiers:
                        key_frame_path = os.path.join(root, file)
                        exam_path = root
                        exam_id = root.split('/')[-1]
                        # if not mask[:mask.rfind('.')-1] in file:
                        #     print(mask)
                        #     print(file)
                        break
            
            if key_frame_path:
                return {
                    "id": exam_id,
                    "path": exam_path,
                    "key_frame": {
                        "id": key_frame_id,
                        "identifiers": key_frame_identifiers,
                        "path": key_frame_path,
                        "mask": mask_path,
                    }
                }
            else: 
                return None
        
        mask_files = os.listdir(exam_masks_path)
        exams = []
        masks_not_found = []

        for mask in tqdm(mask_files):
            #check if there i a c_mask duplicate -> better
            dot_index = mask.rfind('.')
            if mask[dot_index-1]=="a":
                c_mask = mask[:dot_index-1] + 'c' + mask[dot_index:]
                if c_mask in mask_files:
                    continue

            patient_id = mask.split("_")[0]
            patient_dir = os.path.join(exam_videos_path, patient_id)

            mask_path = os.path.join(exam_masks_path, mask)

            kf_dict = get_kf_dict(mask, patient_dir)
            if not kf_dict:
                kf_dict = get_kf_dict(mask, patient_dir, primary_only_match=True)
            
            if kf_dict:
                kf_dict["patient"]= patient_data[patient_id]
                exams.append(kf_dict)
            else:
                masks_not_found.append(mask_path)


        with open("exams.log", 'w') as file:
            for mask in masks_not_found:
                file.write(f"{mask}\n")

        return exams
    
    patient_data = get_patient_data(patients_excel_path)
    kf_exams_data = get_kf_exams_data(exam_videos_path, exam_masks_path, patient_data)

    full_json = {
        "kf_exams": kf_exams_data,
        "patients": patient_data
    }

    with open(output_json, 'w') as json_file:
        json.dump(full_json, json_file, indent=4)

    exams_total = len(kf_exams_data)
    exams_ifr = len([exam for exam in kf_exams_data if (exam['patient']['ifr']!=None and exam['patient']["exclude"]!=2)])
    exams_ffr = len([exam for exam in kf_exams_data if exam['patient']['ffr']!=None])

    print(f"exams total: {exams_total} \nexams w/ ifr: {exams_ifr} \nexams w/ ffr: {exams_ffr}")

if __name__ == "__main__":
    main()