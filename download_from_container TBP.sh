#!/bin/bash

# ====== PARAMETRI DA CONFIGURARE ======
# Cartella comune a tutti i file che si vogliono estrarre
TARGET_FOLDER="/logs"

# Info macchina remota
REMOTE_USER=""
REMOTE_HOST=""
REMOTE_PORT=
REMOTE_SSH="${REMOTE_USER}@${REMOTE_HOST}"

# ID o nome del container Docker remoto
CONTAINER_ID="container_cv_proj_image"

# Percorso nel container da cui scaricare i file
CONTAINER_PATH="/workspace"

# Percorso temporaneo nella macchina remota dove salvare i file prima dello scp
REMOTE_TEMP_DIR="/home/halitchi/CV_PROJ_24_25/tmp_download"

# Percorso locale (la cartella dove si trova questo script)
LOCAL_DEST="./container_files"

# ====== INIZIO SCRIPT ======

echo "Inizio download file(s) dal container..."

echo "Creazione cartella di download temporanea remota..."
# Crea la directory temporanea sulla macchina remota
ssh $REMOTE_SSH "mkdir -p $REMOTE_TEMP_DIR"

echo "Copia della cartella dal container alla macchina remota..."
ssh $REMOTE_SSH "podman cp $CONTAINER_ID:$CONTAINER_PATH/$TARGET_FOLDER $REMOTE_TEMP_DIR/"

echo "Download della cartella sulla macchina locale..."
mkdir -p "$LOCAL_DEST"
scp -r "$REMOTE_SSH:$REMOTE_TEMP_DIR/$(basename $TARGET_FOLDER)" "$LOCAL_DEST/"

echo "Pulizia dei file temporanei sulla macchina remota..."
ssh $REMOTE_SSH "rm -rf $REMOTE_TEMP_DIR"

echo "Download completato con successo!"