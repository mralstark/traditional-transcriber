#!/usr/bin/env bash
# Нормализация исходных записей в audio/wav/NN.wav (16 кГц, моно, pcm16).
# Исходники клади в audio/raw/ и называй по плановому порядку бесед: 01.*, 02.* … 15.*
# Деноиз НЕ добавляем намеренно: агрессивная чистка роняет качество ASR.
set -euo pipefail
mkdir -p audio/wav
shopt -s nullglob
for f in audio/raw/*; do
  base="$(basename "$f")"
  nn="${base%.*}"
  echo "→ $base"
  ffmpeg -y -loglevel error -i "$f" -ac 1 -ar 16000 -af loudnorm -c:a pcm_s16le "audio/wav/${nn}.wav"
done
echo "готово → audio/wav/"
