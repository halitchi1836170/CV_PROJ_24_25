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

## Struttura

.
├── logs/

    ├── logs/
    ├── logs_bu1/
├── plots/
    
    ├── plots/
    ├── plots_bu1/

├── CVUSA_subset.rar : subset of CVUSA to be saved in ./Data/

├── CV_24_25_Slides.pdf

├── Data.py : defines all method to unload train and test data

├── Dockerfile : Dockerfile used to create the container for the project (to bu run on a host with CUDA)

├── Evaluation.py

├── Globals.py

├── Network.py

├── README.md

├── Train.py

├── Utils.py

├── commands.txt : useful txt file containing most of the docker (and podman for the host machine) to manage dcoker images and container

├── download_from_container TBP.sh : script .sh che automatizza il download di file o cartelle dal container alla machcina locale, tutti i parametri impostabili all'intenro

├── logger.py

├── mian.py

├── requirements.txt

├── save_plots.py : used to create most of the plots

├── sky_removal.py : used to define the method to be called when removing the sky using the mask2former pre-trained model

├── upload_to_container TBP.sh : script.sh che automatizza il caricamento di file o cartelledalla macchina locale al container in un certo percorso, tutti i parametri impostabili all'intenro




