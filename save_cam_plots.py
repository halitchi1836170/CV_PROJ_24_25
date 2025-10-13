import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
from pathlib import Path
from scipy import stats
from PIL import Image

# Imposta stile
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (20, 12)
plt.rcParams['font.size'] = 10

# ===== CONFIGURAZIONE =====
BASE_DIR = Path("container_files/logs/ATTENTION_vs_FULL")
OUTPUT_BASE_DIR = Path("container_files/plots/ATTENTION_vs_FULL")
CAM_TYPE = "GRD"  # o "SAT"
IMAGE_CODE = "0000029"  # Cambia con l'immagine che vuoi analizzare

# Metriche da visualizzare
METRICS = {
    "sum": "Somma Totale",
    "mean": "Media",
    "std": "Deviazione Standard",
    "area_thr0.5": "Area > 0.5",
    "area_thr0.2": "Area > 0.2"
}

COLORS = {
    "ATTENTION": "#FF6B6B",  # Rosso
    "FULL": "#4ECDC4"        # Turchese
}

# ===== FUNZIONI =====
def load_metric_data(cam_type, metric, image_code):
    """Carica i dati di una metrica specifica"""
    filepath = BASE_DIR / cam_type / f"{metric.upper()}_{image_code}.npy"
    if filepath.exists():
        return np.load(filepath)
    return None

def normalize_for_radar(data_dict):
    """Normalizza i dati tra 0 e 1 per il radar plot"""
    normalized = {}
    for metric in data_dict.keys():
        att_val = data_dict[metric]["ATTENTION"]
        full_val = data_dict[metric]["FULL"]
        
        # Normalizza tra 0 e 1
        min_val = min(att_val, full_val)
        max_val = max(att_val, full_val)
        
        if max_val - min_val > 0:
            normalized[metric] = {
                "ATTENTION": (att_val - min_val) / (max_val - min_val),
                "FULL": (full_val - min_val) / (max_val - min_val)
            }
        else:
            normalized[metric] = {"ATTENTION": 0.5, "FULL": 0.5}
    
    return normalized

def calculate_scale_ranges(data_dict):
    """Calcola i range appropriati per ogni metrica"""
    ranges = {}
    for metric in data_dict.keys():
        att_val = data_dict[metric]["ATTENTION"]
        full_val = data_dict[metric]["FULL"]
        
        min_val = min(att_val, full_val)
        max_val = max(att_val, full_val)
        
        # Aggiungi un margine del 10% per avere spazio nel grafico
        margin = (max_val - min_val) * 0.1
        ranges[metric] = {
            "min": min_val - margin,
            "max": max_val + margin
        }
    
    return ranges

def create_radar_plot(ax, data_dict, title):
    """Crea un radar plot (spider plot) per confrontare le metriche"""
    
    # Numero di variabili
    categories = list(data_dict.keys())
    N = len(categories)
    
    # Calcola i range per ogni metrica
    ranges = calculate_scale_ranges(data_dict)
    
    # Angoli per ogni asse
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    values_attention = []
    values_full = []
    
    for cat in categories:
        att_val = data_dict[cat]["ATTENTION"]
        full_val = data_dict[cat]["FULL"]
        range_min = ranges[cat]["min"]
        range_max = ranges[cat]["max"]
        
        # Normalizza rispetto al range specifico della metrica
        if range_max - range_min > 0:
            values_attention.append((att_val - range_min) / (range_max - range_min))
            values_full.append((full_val - range_min) / (range_max - range_min))
        else:
            values_attention.append(0.5)
            values_full.append(0.5)
    
    values_attention += values_attention[:1]
    values_full += values_full[:1]
    
    # Plot
    ax.plot(angles, values_attention, 'o-', linewidth=2.5, 
            label='ATTENTION', color=COLORS["ATTENTION"], markersize=8)
    ax.fill(angles, values_attention, alpha=0.25, color=COLORS["ATTENTION"])
    
    ax.plot(angles, values_full, 'o-', linewidth=2.5, 
            label='FULL', color=COLORS["FULL"], markersize=8)
    ax.fill(angles, values_full, alpha=0.25, color=COLORS["FULL"])
    
    # Etichette delle metriche
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([METRICS.get(cat, cat) for cat in categories], size=9)
    
    # Griglia radiale
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels([])  # Nascondiamo le etichette radiali generiche
    ax.grid(True)
    
    # Aggiungi le scale specifiche per ogni asse
    for i, cat in enumerate(categories):
        angle = angles[i]
        
        range_min = ranges[cat]["min"]
        range_max = ranges[cat]["max"]
        
        # Ottieni i valori effettivi per calcolare la percentuale
        att_val = data_dict[cat]["ATTENTION"]
        full_val = data_dict[cat]["FULL"]
        
        # Calcola la differenza percentuale
        if att_val > full_val:
            pct_diff = ((att_val - full_val) / full_val) * 100
            winner = "ATT"
            winner_color = COLORS["ATTENTION"]
        else:
            pct_diff = ((full_val - att_val) / att_val) * 100
            winner = "FULL"
            winner_color = COLORS["FULL"]
        
        # Posiziona le etichette delle scale lungo ogni asse
        x_max = np.cos(angle)
        y_max = np.sin(angle)
        
        # Etichetta del valore massimo con percentuale
        label_text = f'{range_max:.2f}\n({winner}:{cat} +{pct_diff:.1f}%)'
        ax.text(angle+(2*np.pi*15)/360, 1.3, label_text, ha='center', va='center', size=7, bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor=winner_color, alpha=0.8, linewidth=1.5),color=winner_color, weight='bold', clip_on=False)
        
        # Valore minimo al centro
        #ax.text(1.15 * np.sin(angle), 1.15 * np.cos(angle), f'{range_min:.2f}', ha='center', va='center', size=6, alpha=0.6)
    
    ax.set_title(title, size=12, weight='bold', pad=40)
    ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1.1), fontsize=10)
    
    # Aggiungi note esplicative
    note_text = "Note: Each axes has its own scale"
    ax.text(0.5, -0.15, note_text, transform=ax.transAxes,
           ha='center', va='top', fontsize=9, style='italic', color='gray')

def create_accumulated_sum_plot(ax, metric, data, title):
    """Crea un plot dell'evoluzione nel tempo"""
    epochs = np.arange(1, data.shape[1] + 1)
    #print("Shape: ",epochs)
    
    acc_data = np.array([np.nancumsum(data[0]),np.nancumsum(data[1])])
    
    # Plot linee
    ax.plot(epochs, acc_data[0], marker='o', linewidth=2, markersize=4,label='ATTENTION', color=COLORS["ATTENTION"], alpha=0.8)
    ax.plot(epochs, acc_data[1], marker='s', linewidth=2, markersize=4,label='FULL', color=COLORS["FULL"], alpha=0.8)
    
    # Area di confidenza (se vuoi mostrare la variabilità)
    # ax.fill_between(epochs, data[0]*0.95, data[0]*1.05, alpha=0.2, color=COLORS["ATTENTION"])
    
    ax.set_xlabel('Epoca', fontsize=10)
    ax.set_ylabel(METRICS.get(metric, metric), fontsize=10)
    ax.set_title(title, fontsize=11, weight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

def create_evolution_plot(ax, metric, data, title):
    """Crea un plot dell'evoluzione nel tempo"""
    epochs = np.arange(1, data.shape[1] + 1)
    
    # Plot linee
    ax.plot(epochs, data[0], marker='o', linewidth=2, markersize=4,label='ATTENTION', color=COLORS["ATTENTION"], alpha=0.8)
    ax.plot(epochs, data[1], marker='s', linewidth=2, markersize=4,label='FULL', color=COLORS["FULL"], alpha=0.8)
    
    # Area di confidenza (se vuoi mostrare la variabilità)
    # ax.fill_between(epochs, data[0]*0.95, data[0]*1.05, alpha=0.2, color=COLORS["ATTENTION"])
    
    ax.set_xlabel('Epoca', fontsize=10)
    ax.set_ylabel(METRICS.get(metric, metric), fontsize=10)
    ax.set_title(title, fontsize=11, weight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

def create_difference_plot(ax, metric, data, title):
    """Crea un plot della differenza tra i due esperimenti"""
    epochs = np.arange(1, data.shape[1] + 1)
    difference = data[1] - data[0]  # FULL - ATTENTION
    
    # Plot con colori diversi per valori positivi e negativi
    colors = ['green' if d > 0 else 'red' for d in difference]
    ax.bar(epochs, difference, color=colors, alpha=0.6, edgecolor='black')
    
    # Linea dello zero
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
    
    ax.set_xlabel('Epoca', fontsize=10)
    ax.set_ylabel('Differenza (FULL - ATTENTION)', fontsize=10)
    ax.set_title(title, fontsize=11, weight='bold')
    ax.grid(True, alpha=0.3, axis='y')


def find_reference_image(base_dir, image_code):
    """Trova il file PNG che contiene il codice nel nome"""
    for file in Path(base_dir).glob("epoch32_iter*_grd_cam_*.png"):
        if image_code in file.stem:
            return file
    return None

def load_image_third_zoom(image_path, part=2, zoom=1.0):
    """
    Carica e ritaglia un terzo specifico di un'immagine verticale.
    
    Args:
        image_path (str or Path): percorso all'immagine PNG
        part (int): 0 = superiore, 1 = centrale, 2 = inferiore
        zoom (float): fattore di ingrandimento (1.0 = nessuno)
    """
    img = Image.open(image_path)
    width, height = img.size

    # Calcola i limiti del terzo scelto
    third = height // 3
    top = part * third
    bottom = (part + 1) * third

    # Ritaglia
    img_cropped = img.crop((0, top, width, bottom))

    # Applica zoom opzionale
    if zoom != 1.0:
        new_size = (int(width * zoom), int((bottom - top) * zoom))
        img_cropped = img_cropped.resize(new_size, Image.Resampling.LANCZOS)

    return img_cropped

def load_bottom_third_image(image_path):
    """Ritaglia solo il terzo inferiore di un'immagine verticale"""
    img = Image.open(image_path)
    width, height = img.size
    top = int(2 * height / 3)
    bottom = height
    return img.crop((0, top, width, bottom))

def create_stability_plot(ax, data_dict, title):
    """Crea un plot che mostra la stabilità (deviazione standard) delle metriche"""
    metrics_list = list(data_dict.keys())
    std_attention = [np.nanstd(data_dict[m][0]) for m in metrics_list]
    std_full = [np.nanstd(data_dict[m][1]) for m in metrics_list]
    
    x = np.arange(len(metrics_list))
    width = 0.35
    
    ax.bar(x - width/2, std_attention, width, label='ATTENTION', color=COLORS["ATTENTION"], alpha=0.8)
    ax.bar(x + width/2, std_full, width, label='FULL', color=COLORS["FULL"], alpha=0.8)
    
    ax.set_xlabel('Metrica', fontsize=10)
    ax.set_ylabel('Deviazione Standard', fontsize=10)
    ax.set_yscale("log")
    ax.set_title(title, fontsize=11, weight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([METRICS.get(m, m) for m in metrics_list], rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

def create_heatmap_plot(ax, data_dict, title):
    """Crea una heatmap delle metriche per epoca"""
    metrics_list = list(data_dict.keys())
    
    # Crea matrice: righe = metriche, colonne = epoche
    # Usa la differenza normalizzata (FULL - ATTENTION) / ATTENTION
    heatmap_data = []
    for metric in metrics_list:
        att = data_dict[metric][0]
        full = data_dict[metric][1]
        # Calcola variazione percentuale
        diff = ((full - att) / (att + 1e-8)) * 100
        heatmap_data.append(diff)
    
    heatmap_data = np.array(heatmap_data)
    
    # Plot heatmap
    im = ax.imshow(heatmap_data, cmap='RdYlGn', aspect='auto', vmin=-50, vmax=50)
    
    # Etichette
    ax.set_yticks(np.arange(len(metrics_list)))
    ax.set_yticklabels([METRICS.get(m, m) for m in metrics_list])
    ax.set_xlabel('Epoca', fontsize=10)
    ax.set_title(title, fontsize=11, weight='bold')
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Variazione % (FULL vs ATTENTION)', rotation=270, labelpad=15)


def create_global_visualization(cam_type, image_codes):
    """Crea una visualizzazione globale media su tutte le immagini."""
    data_all = {metric: [] for metric in METRICS.keys()}
    
    # Carica tutti i dati per ogni immagine
    for img_code in image_codes:
        print(f"Loading data for image: {img_code}")
        for metric in METRICS.keys():
            data = load_metric_data(cam_type, metric, img_code)
            if data is not None:
                data_all[metric].append(data)
    
    print(f"data_all dictionary has {len(data_all.keys())} keys")
    for key in data_all.keys():
        print(f"Metric of key {key} has shape: {len(data_all.get(key))}")
    
    # Calcola la media e deviazione standard su tutte le immagini
    data_mean = {}
    for metric, values in data_all.items():
        if len(values) == 0:
            continue
        stacked = np.stack(values, axis=0)  # [n_images, 2, n_epochs]
        print(f"Stacked values for metric: {metric} have shape: {stacked.shape}")
        data_mean[metric] = np.nanmean(stacked, axis=0)  # [2, n_epochs]

    if not data_mean:
        print("Nessun dato trovato per la visualizzazione globale.")
        return

    # ===== PLOT GLOBALE =====
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)

    # Radar (medie finali)
    final_values = {
        metric: {
            "ATTENTION": np.nanmean(data_mean[metric][0, -5:]),
            "FULL": np.nanmean(data_mean[metric][1, -5:])
        } for metric in data_mean.keys()
    }

    #RIGA 1

    ax_radar = fig.add_subplot(gs[0, 0:2], projection='polar')
    create_radar_plot(ax_radar, final_values, f'Confronto Metriche Globali - {cam_type}')

    # Stabilità media
    ax_stab = fig.add_subplot(gs[0, 2:4])
    create_stability_plot(ax_stab, data_mean, 'Stabilità media delle metriche')

    #RIGA 2

    # Evoluzione di una metrica chiave
    key_metric = list(data_mean.keys())[0]
    ax_evo = fig.add_subplot(gs[1, 0:2])
    create_evolution_plot(ax_evo, key_metric, data_mean[key_metric],f'Evoluzione media: {METRICS.get(key_metric, key_metric)}')
    
    ax_cumulated_evo=fig.add_subplot(gs[1, 2:4])
    create_accumulated_sum_plot(ax_cumulated_evo,key_metric,data_mean[key_metric],f'Cumulated SUM')
    
    #RIGA 3
    # Differenza
    ax_diff = fig.add_subplot(gs[2, 0])
    create_difference_plot(ax_diff, key_metric, data_mean[key_metric],f'Differenza media: {METRICS.get(key_metric, key_metric)}')

    key_metric = list(data_mean.keys())[3]
    ax_evo = fig.add_subplot(gs[2, 1:2])
    create_evolution_plot(ax_evo, key_metric, data_mean[key_metric],f'Evoluzione media: {METRICS.get(key_metric, key_metric)}')
    
    key_metric = list(data_mean.keys())[4]
    ax_evo = fig.add_subplot(gs[2, 2:3])
    create_evolution_plot(ax_evo, key_metric, data_mean[key_metric],f'Evoluzione media: {METRICS.get(key_metric, key_metric)}')
    
    # Heatmap globale
    ax_heat = fig.add_subplot(gs[2, 3:4])
    create_heatmap_plot(ax_heat, data_mean, 'Heatmap variazioni % medie per epoca')

    fig.suptitle(f'Analisi Globale GradCAM - {cam_type}', fontsize=16, weight='bold', y=0.995)

    output_path = OUTPUT_BASE_DIR / f"analysis_GLOBAL_{cam_type}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Grafico globale salvato in: {output_path}") 


def create_comprehensive_visualization(cam_type, image_code):
    """Crea la visualizzazione completa"""
    
    # Carica tutti i dati
    data_dict = {}
    final_values = {}  # Per il radar plot (usa valori finali)
    
    for metric in METRICS.keys():
        data = load_metric_data(cam_type, metric, image_code)
        if data is not None:
            data_dict[metric] = data
            # Usa la media delle ultime 5 epoche per il radar
            final_values[metric] = {
                "ATTENTION": np.nanmean(data[0,:]),
                "FULL": np.nanmean(data[1,:])
            }
    
    if not data_dict:
        print("Nessun dato trovato!")
        return
    
    # Crea figura con layout complesso
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)
    
    # ===== ROW 1: RADAR + STABILITY + STATS =====
    # Radar plot (occupa 2 colonne)
    ax_radar = fig.add_subplot(gs[0, 0], projection='polar')
    create_radar_plot(ax_radar, final_values, f'Confronto Metriche Finali - {cam_type} - {image_code}')
    
    ax_ref = fig.add_subplot(gs[0, 1:3])
    ax_ref.axis('off')

    # Cerca immagine di riferimento
    ref_img_path = find_reference_image(f"./container_files/plots/FULL/GIF_{image_code}", image_code)
    if ref_img_path:
        ref_img = load_image_third_zoom(ref_img_path,part=0)
        zoom = 1.0  # ingrandisce visivamente l'immagine nel plot
        ax_ref.imshow(ref_img, extent=[0, ref_img.size[0]*zoom, 0, ref_img.size[1]*zoom])
        ax_ref.set_xlim(0, ref_img.size[0])
        ax_ref.set_ylim(0, ref_img.size[1])
        ax_ref.set_title("Last Ground CAM for the reference image", fontsize=11, weight='bold')
    else:
        ax_ref.text(0.5, 0.5, "Immagine non trovata", ha='center', va='center', fontsize=12)
    
    # Stability plot
    ax_stability = fig.add_subplot(gs[0, 3:4])
    create_stability_plot(ax_stability, data_dict, 'Stabilità delle Metriche')
    
    # # Box plot comparativo
    # ax_box = fig.add_subplot(gs[0, 3])
    # # Esempio con una metrica chiave
    # key_metric = list(data_dict.keys())[0]
    # box_data = [data_dict[key_metric][0], data_dict[key_metric][1]]
    # bp = ax_box.boxplot(box_data, labels=['ATTENTION', 'FULL'],patch_artist=True, widths=0.6)
    # bp['boxes'][0].set_facecolor(COLORS["ATTENTION"])
    # bp['boxes'][1].set_facecolor(COLORS["FULL"])
    # ax_box.set_ylabel(METRICS.get(key_metric, key_metric))
    # ax_box.set_title(f'Distribuzione: {METRICS.get(key_metric, key_metric)}',fontsize=11, weight='bold')
    # ax_box.grid(True, alpha=0.3, axis='y')
    
    # ===== ROW 2: EVOLUTION PLOTS (4 metriche) =====
    metrics_to_plot = list(data_dict.keys())[:5]
    
    metric=metrics_to_plot[0]
    ax = fig.add_subplot(gs[1, 0:2])
    create_evolution_plot(ax, metric, data_dict[metric], f'Evoluzione: {METRICS.get(metric, metric)}')
    
    metric=metrics_to_plot[0]
    ax = fig.add_subplot(gs[1, 2:4])
    create_accumulated_sum_plot(ax, metric, data_dict[metric], f'Evoluzione: ACCUMULATED SUM')
    
    # for i, metric in enumerate(metrics_to_plot):
    #     ax = fig.add_subplot(gs[1, i])
    #     create_evolution_plot(ax, metric, data_dict[metric], f'Evoluzione: {METRICS.get(metric, metric)}')
    
    # ===== ROW 3: DIFFERENCE PLOTS + HEATMAP =====
    # Difference plots (prime 2 metriche)
    metric=metrics_to_plot[0]
    ax = fig.add_subplot(gs[2, 0])
    create_difference_plot(ax, metric, data_dict[metric], f'Differenza: {METRICS.get(metric, metric)}')
    
    metric=metrics_to_plot[3]
    ax = fig.add_subplot(gs[2, 1])
    create_evolution_plot(ax, metric, data_dict[metric], f'Evoluzione: {METRICS.get(metric, metric)}')
    
    # Heatmap (occupa 2 colonne)
    ax_heatmap = fig.add_subplot(gs[2, 2:])
    create_heatmap_plot(ax_heatmap, data_dict, 
                       'Heatmap Variazioni % per Epoca')
    
    # Titolo generale
    fig.suptitle(f'Analisi Comparativa GradCAM - {cam_type} - Immagine {image_code}', 
                 fontsize=16, weight='bold', y=0.995)
    
    # Salva
    output_path = OUTPUT_BASE_DIR / f"analysis_{cam_type}_{image_code}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Grafico salvato in: {output_path}")
    
    #plt.show()

# ===== ESECUZIONE =====
if __name__ == "__main__":
    for IMAGE_CODE_ITER in ["0000029","0001124","0003157", "0017660", "0031663","0032354","0006134","0010078","0032488","0011435"]:
         create_comprehensive_visualization(CAM_TYPE, IMAGE_CODE_ITER)
    
    #create_comprehensive_visualization(CAM_TYPE, IMAGE_CODE)
    
    # Analisi globale
    IMAGE_CODES = ["0000029","0001124","0003157", "0017660", "0031663","0032354","0006134","0010078","0032488","0011435"]
    create_global_visualization(CAM_TYPE, IMAGE_CODES)