import json, os, shutil
from tqdm import tqdm

def main():
    step_size = 3
    context_window = (5,5)  # (steps before, steps after)
    clinical_data_path = ""
    context_frames_path = ""

    def get_context_frame_ids(kf_index, total_exam_frames):

        start = kf_index - context_window[0] * step_size
        end = kf_index + context_window[1] * step_size

        context_frame_ids = list(range(start, end + 1, step_size))
        clamped_cf_ids = [max(0, min(x, total_exam_frames-1)) for x in context_frame_ids]

        return clamped_cf_ids

    with open(clinical_data_path, 'r') as file:
        clinical_data = json.load(file)

    for exam in tqdm(clinical_data['kf_exams']):
        frames = os.listdir(exam['path'])
        kf_index = next(i for i, s in enumerate(frames) if exam['key_frame']['name'] in s)

        clamped_cf_ids = get_context_frame_ids(kf_index, len(frames))

        destination_path = os.path.join(context_frames_path, exam['patient']['id'],exam["id"])
        os.makedirs(destination_path, exist_ok=True)

        for cf_id in clamped_cf_ids:
            shutil.copy(os.path.join(exam['path'], frames[cf_id]), destination_path)

if __name__ == "__main__":
    main()