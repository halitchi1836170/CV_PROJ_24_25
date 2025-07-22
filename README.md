# Ground-to-Aerial Image Matching with Attention and Sky Removal

Questo progetto implementa in PyTorch un sistema di image retrieval tra immagini ground-level e immagini satellitari, originariamente sviluppato in TensorFlow. Include moduli opzionali per attenzione spaziale e rimozione del cielo.

## Architettura

Siamese-like con backbone VGG16

Input: immagini ground e satellite (RGB), opzionalmente in base alla configurazione selezionata attention maps o sky-masked

Output: descrittori globali normalizzati

Loss: Triplet Loss e integrazione con Saliency loss in configurazione SKYREMOVAL o FULL

## Dataset

Il dataset utilizzato è una versione ridotta del CVUSA, organizzato tramite file CSV:

train-19zl.csv

val-19zl.csv

## Addestramento

python main.py --mode TRAIN --experiments ALL

Opzioni:
 
--mode: TRAIN o TEST

--experiments: configurazione ALL (vengono testate tutte le configurazioni) o FULL


I modelli vengono salvati in ./models/{EXP_NAME}/epoch{N}/model.pth

## Valutazione

python Evaluation.py

Comportamento:

Carica tutti i modelli in ./models/* /epoch*/model.pth

Estrae descrittori per immagini ground e satellite

Calcola: Top-1, Top-5, Top-10, Top-1% recall

Genera: top-5 retrieval, curva CMC

## Output generati

plots/{EXP_NAME}_top_5_matches.png: visualizzazione retrieval

plots/CMC_all_models.png: confronto curve CMC tra modelli

## Requisiti

requirements.txt

.
├── logs/
    ├── logs/
    ├── logs_bu1/
├── plots/
    ├── plots/
    ├── plots_bu1/
├── CVUSA_subset.rar
├── CV_24_25_Slides.pdf
├── Data.py
├── Dockerfile
├── Evaluation.py
├── Globals.py
├── Network.py
├── README.md
├── Train.py
├── Utils.py
├── command.txt
├── logger.py
├── mian.py
├── requirements.txt
├── save_plots.py
├── sky_removal.py




