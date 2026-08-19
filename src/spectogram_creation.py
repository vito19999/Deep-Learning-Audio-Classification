import os
import json
import numpy as np
import librosa
from tqdm import tqdm
import gc

# Classi animali
ANIMALS = ['dog', 'cat', 'chirping_birds', 'rooster', 'pig', 'cow', 
           'frog', 'crickets', 'crow', 'hen', 'sheep', 'insects']

# Mapping numerico
CLASS_TO_ID = {animal: i for i, animal in enumerate(ANIMALS)}

# Path
BASE_PATH = "/media/neurone-pc5/Volume/AI & ML/Soundscapes/audio_sources"
MIXTURES_PATH = f"{BASE_PATH}/mixtures"
OUTPUT_PATH = f"{BASE_PATH}/spettrogrammi"

def load_audio_mono(audio_path):
    """Carica audio e converte a mono"""
    y, sr = librosa.load(audio_path, sr=16000, mono=True)
    return y

def check_existing(split):
    """Controllo mix processati"""
    split_output = f"{OUTPUT_PATH}/{split}"
    
    if not os.path.exists(split_output):
        return set()
    
    existing = set()
    for mix_folder in os.listdir(split_output):
        if mix_folder.startswith("mix_"):
            spec_file = f"{split_output}/{mix_folder}/spectrogram.npz"
            label_file = f"{split_output}/{mix_folder}/labels.npz"
            
            if os.path.exists(spec_file) and os.path.exists(label_file):
                existing.add(mix_folder)
    
    return existing

def process_batch(mix_batch, split):
    """Processa un batch di mix"""
    for mix_folder in tqdm(mix_batch, desc=f"Processing {split}", leave=False):
        try:
            # Path files
            audio_path = f"{MIXTURES_PATH}/{split}/{mix_folder}/mixture.wav"
            json_path = f"{MIXTURES_PATH}/{split}/{mix_folder}/mixture_info.json"
            
            # Carica audio (con conversione mono)
            audio = load_audio_mono(audio_path)
            
            # Genera spettrogramma
            spec = librosa.feature.melspectrogram(
                y=audio,
                sr=16000,
                n_mels=128,
                n_fft=2048,
                hop_length=512,
                fmax=8000
            )
            spec_db = librosa.power_to_db(spec, ref=np.max)
            
            # Estrai label
            with open(json_path) as f:
                info = json.load(f)
            
            event_class = info["event_info"][0]["Class"]
            class_id = CLASS_TO_ID[event_class]
            
            # Salva
            output_folder = f"{OUTPUT_PATH}/{split}/{mix_folder}"
            os.makedirs(output_folder, exist_ok=True)
            
            np.savez_compressed(f"{output_folder}/spectrogram.npz", spectrogram=spec_db)
            np.savez_compressed(f"{output_folder}/labels.npz", 
                              class_id=class_id, 
                              class_name=event_class)
            
        except Exception as e:
            print(f" Errore in {mix_folder}: {e}")
    
    gc.collect()

def generate_spectrograms():
    
    print(" GENERATORE SPETTROGRAMMI")
    print(f" Input: {MIXTURES_PATH}")
    print(f" Output: {OUTPUT_PATH}")
    print(f" Classi: {len(ANIMALS)} → {ANIMALS}")
    
    splits = ["train", "test", "validation"]
    batch_size = 1000
    
    # Salva mapping
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    np.savez_compressed(f"{OUTPUT_PATH}/class_mapping.npz", 
                       class_to_id=CLASS_TO_ID, 
                       animals=ANIMALS)
    
    for split in splits:
        print(f" Processando {split}...")
        
        # Trova mixtures
        split_path = f"{MIXTURES_PATH}/{split}"
        all_mix_folders = [f for f in os.listdir(split_path) if f.startswith("mix_")]
        all_mix_folders.sort(key=lambda x: int(x.split("_")[1]))
        
        # Resume: controlla già processate
        existing = check_existing(split)
        to_process = [m for m in all_mix_folders if m not in existing]
        
        print(f"Da processare: {len(to_process)} di {len(all_mix_folders)}")
        
        if not to_process:
            print(f"{split} già completato!")
            continue
        
        # Processa in batch
        for i in range(0, len(to_process), batch_size):
            batch = to_process[i:i + batch_size]
            print(f" Batch {i//batch_size + 1}: {len(batch)} files")
            
            process_batch(batch, split)
        
        print(f"{split} completato!")
    
    print(f"COMPLETATO! Output: {OUTPUT_PATH}")

try:
    generate_spectrograms()
except KeyboardInterrupt:
    print("Interrotto! Riavvia per continuare.")



        
