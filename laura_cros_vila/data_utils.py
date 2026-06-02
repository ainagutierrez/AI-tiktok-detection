import numpy as np
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE_PATH = r"your/base/path"

def load_single_embedding(file):
    if file.endswith('.npy') and os.path.exists(file):
        return np.load(file, mmap_mode='r')
    return None

def load_embeddings(files):
    valid_files = [f for f in files if f.endswith('.npy')]
    
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(load_single_embedding, file) for file in valid_files]
        embeddings = [future.result() for future in as_completed(futures) if future.result() is not None]
    
    return np.array(embeddings)

def get_split(split, embedding, folders):
    files = []
    y = []
    for folder in folders:
        emb_folder = os.path.join(BASE_PATH, folder, split, 'embeddings', embedding)
        
        if not os.path.exists(emb_folder):
            print(f"[WARNING] Missing folder: {emb_folder}")
            continue
            
        for file_name in os.listdir(emb_folder):
            if file_name.endswith('.npy'):
                files.append(os.path.join(emb_folder, file_name))
                y.append(folder)
    
    X = load_embeddings(files)
    y = np.array(y)
    return X, y, files
