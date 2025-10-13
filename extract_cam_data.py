import os
import json
import numpy as np
from pathlib import Path
from collections import defaultdict

# ===== CONFIGURAZIONE =====
# Definisci qui le metriche da estrarre
METRICS_TO_EXTRACT = [
    "sum",
    "mean",
    "max",
    "std",
    "area_thr0.2",
    "area_thr0.5",
    "entropy"
]

# Percorsi base
BASE_DIR = Path("container_files/logs")
EXPERIMENTS = ["ATTENTION", "FULL"]
OUTPUT_DIR = BASE_DIR / "ATTENTION_vs_FULL"
CAM_TYPES = ["GRD", "SAT"]

# ===== FUNZIONI =====
def parse_json_filename(filename):
    """Estrae informazioni dal nome del file JSON"""
    # Formato: epoch2_iter149_GRD_cam_metric_0000029.json
    parts = filename.replace(".json", "").split("_")
    
    epoch = int(parts[0].replace("epoch", ""))
    iteration = int(parts[1].replace("iter", ""))
    cam_type = parts[2]
    image_code = parts[-1]
    
    return epoch, iteration, cam_type, image_code

def collect_metrics():
    """Raccoglie tutte le metriche da entrambi gli esperimenti"""
    # Struttura: data[cam_type][metric][image_code][experiment] = list of values by epoch
    data = {
        "GRD": defaultdict(lambda: defaultdict(lambda: {"ATTENTION": {}, "FULL": {}})),
        "SAT": defaultdict(lambda: defaultdict(lambda: {"ATTENTION": {}, "FULL": {}}))
    }
    
    print("Inizio raccolta dati...")
    
    for experiment in EXPERIMENTS:
        exp_path = BASE_DIR / experiment / "cam_metrics"
        
        if not exp_path.exists():
            print(f"ATTENZIONE: Cartella {exp_path} non trovata!")
            continue
        
        print(f"\nProcessando esperimento: {experiment}")
        
        # Itera attraverso tutte le cartelle cam_metrics_XXXXXXX
        for cam_folder in exp_path.glob("cam_metrics_*"):
            image_code = cam_folder.name.replace("cam_metrics_", "")
            
            # Itera attraverso tutti i file JSON nella cartella
            json_files = list(cam_folder.glob("*.json"))
            print(f"  Immagine {image_code}: {len(json_files)} file trovati")
            
            for json_file in json_files:
                try:
                    # Leggi il file JSON
                    with open(json_file, 'r') as f:
                        metrics = json.load(f)
                    
                    # Estrai informazioni dal nome del file
                    epoch, iteration, cam_type, _ = parse_json_filename(json_file.name)
                    
                    # Salva le metriche richieste
                    for metric in METRICS_TO_EXTRACT:
                        if metric in metrics:
                            if epoch not in data[cam_type][metric][image_code][experiment]:
                                data[cam_type][metric][image_code][experiment][epoch] = []
                            
                            data[cam_type][metric][image_code][experiment][epoch].append(metrics[metric])
                
                except Exception as e:
                    print(f"    ERRORE nel processare {json_file.name}: {e}")
    
    return data

def create_output_structure(data):
    """Crea la struttura di cartelle e salva i file NPY"""
    print("\n" + "="*50)
    print("Creazione struttura output...")
    
    # Crea le cartelle principali
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    for cam_type in CAM_TYPES:
        cam_dir = OUTPUT_DIR / cam_type
        cam_dir.mkdir(exist_ok=True)
        print(f"\nCartella creata: {cam_dir}")
        
        # Per ogni metrica
        for metric in METRICS_TO_EXTRACT:
            if metric not in data[cam_type]:
                continue
            
            # Per ogni immagine
            for image_code in data[cam_type][metric].keys():
                exp_data = data[cam_type][metric][image_code]
                
                # Raccogli i dati per entrambi gli esperimenti
                attention_values = []
                full_values = []
                
                # Ordina le epoche e crea array
                all_epochs = sorted(set(list(exp_data["ATTENTION"].keys()) + 
                                      list(exp_data["FULL"].keys())))
                
                for epoch in all_epochs:
                    # Per ATTENTION
                    if epoch in exp_data["ATTENTION"]:
                        # Media dei valori per quella epoca (se ci sono più iterazioni)
                        attention_values.append(np.mean(exp_data["ATTENTION"][epoch]))
                    else:
                        attention_values.append(np.nan)
                    
                    # Per FULL
                    if epoch in exp_data["FULL"]:
                        full_values.append(np.mean(exp_data["FULL"][epoch]))
                    else:
                        full_values.append(np.nan)
                
                # Crea array 2D: [2 righe x N epoche]
                result_array = np.array([attention_values, full_values])
                
                # Salva il file NPY
                filename = f"{metric.upper()}_{image_code}.npy"
                filepath = cam_dir / filename
                np.save(filepath, result_array)
                
            print(f"  Metrica '{metric}': {len(data[cam_type][metric])} file creati")

def print_summary(data):
    """Stampa un riepilogo dei dati raccolti"""
    print("\n" + "="*50)
    print("RIEPILOGO")
    print("="*50)
    
    for cam_type in CAM_TYPES:
        print(f"\n{cam_type}:")
        for metric in METRICS_TO_EXTRACT:
            if metric in data[cam_type]:
                n_images = len(data[cam_type][metric])
                print(f"  {metric}: {n_images} immagini")

# ===== MAIN =====
if __name__ == "__main__":
    print("="*50)
    print("PROCESSAMENTO METRICHE CAM")
    print("="*50)
    print(f"Metriche da estrarre: {', '.join(METRICS_TO_EXTRACT)}")
    print(f"Output directory: {OUTPUT_DIR}")
    
    # Raccoglie i dati
    data = collect_metrics()
    
    # Stampa riepilogo
    print_summary(data)
    
    # Crea la struttura output
    create_output_structure(data)
    
    print("\n" + "="*50)
    print("COMPLETATO!")
    print("="*50)
    print(f"\nI file NPY sono stati salvati in: {OUTPUT_DIR}")
    print("\nEsempio di utilizzo:")
    print(">>> import numpy as np")
    print(">>> data = np.load('logs/ATTENTION_vs_FULL/GRD/SUM_0000029.npy')")
    print(">>> print(data.shape)  # (2, num_epochs)")
    print(">>> attention_data = data[0]  # Prima riga: ATTENTION")
    print(">>> full_data = data[1]       # Seconda riga: FULL")