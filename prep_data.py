from vessel_segmentation import VesselSegmentation
from preprocess.key_frame import *
from preprocess.patient_data import *
from common import get_config

import os
from tqdm import tqdm
import numpy as np
import re

def get_all_exam_paths(exams_dir, patient_ids, processed_exam_paths):
    exam_paths = []
    
    for patient_id in patient_ids:
        patient_exams_dir = os.path.join(exams_dir, str(patient_id))
        if not os.path.isdir(patient_exams_dir):
            continue

        exams = os.listdir(patient_exams_dir)
        exam_dirs = [os.path.join(patient_exams_dir, exam) for exam in exams if (
            os.path.isdir(os.path.join(patient_exams_dir, exam)) 
            and os.path.join(patient_exams_dir, exam) not in processed_exam_paths
        )]
        exam_paths.extend(exam_dirs)
    
    return exam_paths

def get_processed_exam_paths(processed_exams_log):
    log_lines=[]
    with open(processed_exams_log, 'r') as file:
        for line in file:
            log_lines.append(line.strip())
    return log_lines

def log_processed_exam(processed_exams_log, exam):
    with open(processed_exams_log, 'a') as file:
        file.write(exam + '\n')

def append_dict_to_json_file(json_file_path, new_dict):
    # Step 1: Read the existing JSON file
    if os.path.exists(json_file_path):
        with open(json_file_path, 'r') as file:
            try:
                data = json.load(file)
            except json.JSONDecodeError:
                data = []
    else:
        data = []

    # Step 2: Append the new JSON object to the list
    data.append(new_dict)

    # Step 3: Write the updated list back to the JSON file
    with open(json_file_path, 'w') as file:
        json.dump(data, file, indent=4)


def main():
    config = get_config("./config.yaml")
    log_file = config.preprocess.exams_log

    vs = VesselSegmentation()

    patient_df = get_patient_data(config.clinical_data_excel_path)
    patient_json = patient_df.to_json(orient='index')
    patient_data = json.loads(patient_json)

    with open(config.patient_json_path, 'w') as patient_json:
        json.dump(patient_data, patient_json, indent=4)

    processed_exam_paths = get_processed_exam_paths(log_file)
    exam_paths = get_all_exam_paths(config.exams_dir, list(patient_df['id'].unique()), processed_exam_paths)

    for exam_path in tqdm(exam_paths):
        kf_data = get_exam_keyframe(exam_path, config.kf_mask_dir, vs)
        if kf_data:
            kf_index, kf_path, kf_mask_path, kf_mask = kf_data
            
            exam_id = exam_path.split("/")[-1]
            patient_id = exam_path.split("/")[-2]

            exam_dict = {
                "id": exam_id,
                "patient_id": patient_id,
                "path": exam_path,
                "key_frame": { 
                    "id": str(kf_index+1),
                    "path": kf_path,
                    "mask": kf_mask_path
                }
            }

            append_dict_to_json_file(config.exam_json_path, exam_dict)
        log_processed_exam(log_file, exam_path)

if __name__ == "__main__":
    main()
    # exam = "/media/masstorage/angiograms/exams/250/47"
    # kf_mask_dir = "/media/masstorage/angiograms/key_masks"
    # vs = VesselSegmentation()
    # get_exam_keyframe(exam, kf_mask_dir, vs)

