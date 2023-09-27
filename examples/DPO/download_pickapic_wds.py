from datasets import load_dataset
import requests
import os
from tqdm import tqdm
import tarfile
from multiprocessing import Pool, cpu_count

"""
This script takes the filtered version of the PickAPic prompt dataset
and downloads the associated images, then tars them. This tar file can then
be moved to S3 or loaded directly if needed. Number of samples can be specified
"""

n_samples = 1000
data_root = "./pickapic_sample"
url = "CarperAI/pickapic_v1_no_images_training_sfw"
n_cpus = cpu_count()  # Detect the number of CPUs

base_name = os.path.basename(data_root).replace('.', '').replace('/', '')

def make_tarfile(output_filename, source_dir):
    with tarfile.open(output_filename, "w") as tar:
        tar.add(source_dir, arcname=os.path.basename(source_dir))

def download_image(args):
    url, filename = args
    response = requests.get(url)
    with open(filename, 'wb') as f:
        f.write(response.content)

if __name__ == "__main__":
    ds = load_dataset("CarperAI/pickapic_v1_no_images_training_sfw")['train']
    os.makedirs(data_root, exist_ok = True)

    id_counter = 0
    with Pool(n_cpus) as p:
        for row in tqdm(ds, total = n_samples):
            if id_counter >= n_samples:
                break
            if row['has_label']:
                id_str = str(id_counter).zfill(8)
                with open(os.path.join(data_root, f'{id_str}.prompt.txt'), 'w', encoding='utf-8') as f:
                    # Ensure the caption is in UTF-8 format
                    caption = row['caption'].encode('utf-8').decode('utf-8')
                    f.write(caption)
                if row['label_0']:
                    p.map(download_image, [(row['image_0_url'], os.path.join(data_root, f'{id_str}.chosen.png')), 
                                           (row['image_1_url'], os.path.join(data_root, f'{id_str}.rejected.png'))])
                else:
                    p.map(download_image, [(row['image_1_url'], os.path.join(data_root, f'{id_str}.chosen.png')), 
                                           (row['image_0_url'], os.path.join(data_root, f'{id_str}.rejected.png'))])
                id_counter += 1
    
    make_tarfile(f"{base_name}.tar", data_root)