#!/bin/bash

# ====== PARAMETRI DA CONFIGURARE ======
# Array di file da copiare (aggiungi i nomi che ti servono)
FILES=("Evaluation.py")

# Info macchina remota
REMOTE_USER=""
REMOTE_HOST=""
REMOTE_PORT=                     
REMOTE_SSH="${REMOTE_USER}@${REMOTE_HOST}"

# ID o nome del container Docker remoto
CONTAINER_ID="container_cv_proj_image"

# Percorso nel container dove copiare i file
CONTAINER_PATH="/workspace"

# Percorso temporaneo nella macchina remota dove depositare i file prima della copia nel container
REMOTE_TEMP_DIR="/home/halitchi/CV_PROJ_24_25/tmp_upload"

# ====== INIZIO SCRIPT ======

echo "Inizio trasferimento file..."

# Crea la directory temporanea sulla macchina remota
ssh $REMOTE_SSH "mkdir -p $REMOTE_TEMP_DIR"

# Copia i file nella macchina remota
for file in "${FILES[@]}"; do
    echo "Copio $file sulla macchina remota..."
    scp "$file" "$REMOTE_SSH:$REMOTE_TEMP_DIR/"
done

# Copia i file nel container
echo "Copio i file nel container $CONTAINER_ID..."
ssh $REMOTE_SSH "podman cp $REMOTE_TEMP_DIR/. $CONTAINER_ID:$CONTAINER_PATH"

# (Opzionale) Rimuovi i file temporanei dalla macchina remota
echo "Rimuovo i file temporanei dalla macchina remota..."
ssh $REMOTE_SSH "rm -rf $REMOTE_TEMP_DIR"

echo "Tutto completato con successo!"
