#!/usr/bin/env bash

# Copyright 2024 Pavel Denisov
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;
. ./db.sh || exit 1;

# general configuration
SECONDS=0

 . utils/parse_options.sh || exit 1;

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

if [ -z "${COMMONVOICE}" ]; then
    log "Fill the value of 'FLEURS' of db.sh"
    exit 1
fi

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

log "Preparing FLEURS data"

python local/create_fleurs.py

log "Preparing Common Voice data"

if [ ! -f downloads/cv12_train_utts.lst ]; then
    log "Downloading the list of Common Voice utterances"
    curl 'https://zenodo.org/records/10900287/files/cv12_train_utts.lst.gz?download=1' | \
        gunzip > downloads/cv12_train_utts.lst
fi

python local/prepare_cv.py ${COMMONVOICE}

log "Combining FLEURS and Common Voice training data"

utils/combine_data.sh data/train_t data/train_fleurs data/train_cv

log "Exporting BLOOMZ token embeddings"
python local/export_hf_embed_tokens.py bigscience/bloomz-7b1 downloads/bloomz_token_embeddings.pth

log "Successfully finished. [elapsed=${SECONDS}s]"
