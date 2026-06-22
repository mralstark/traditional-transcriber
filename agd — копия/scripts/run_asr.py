#!/usr/bin/env python3
"""
ASR на GigaAM-v3 + диаризация (pyannote) + привязка кластеров к ИМЕНАМ спикеров.

Движок распознавания — GigaAM (Sber), SOTA для русского, ставится без ctranslate2,
поэтому работает в т.ч. на Python 3.14. Whisper/whisperx больше не используются.

Запускать ЛОКАЛЬНО: нужен GPU (желательно), HF_TOKEN и принятые условия моделей pyannote:
    pyannote/segmentation-3.0, pyannote/speaker-diarization-3.1, pyannote/embedding

Вход:
    audio/wav/NN.wav                         (16k mono pcm16; см. prep_audio.sh)
    refs/ref_malofeev.wav, ref_dugin.wav, ref_tkachev.wav   (по 20-30с чистой речи)
Выход:
    out/asr/NN_raw.json      (сегменты: start/end/speaker/text)
    out/asr/NN_speaker.txt   ("[мм:сс] Имя: текст реплики")

Запуск:
    python scripts/run_asr.py            # все audio/wav/NN.wav
    python scripts/run_asr.py 01         # только 01.wav (калибровка)
    python scripts/run_asr.py 01 02 03

ВНИМАНИЕ: API gigaam/pyannote чувствителен к версиям. Если что-то отвалилось —
агент Claude Code должен прочитать трейс и поправить вызовы под установленные версии.
"""
import os
import sys
import json
import glob
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

import torch
import gigaam
from pyannote.audio import Pipeline, Model, Inference
from pyannote.core import Segment

# ----------------------------- конфиг -----------------------------
HF_TOKEN = os.environ.get("HF_TOKEN")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GIGAAM_MODEL = "v3_e2e_rnnt"   # пунктуация+нормализация; запасной вариант: "v2_rnnt"
N_SPK = 3                      # знаем точно: трое
WAV_DIR = Path("audio/wav")
OUT_DIR = Path("out/asr")

REFS = {
    "Малофеев": "refs/ref_malofeev.wav",
    "Дугин":    "refs/ref_dugin.wav",
    "Ткачёв":   "refs/ref_tkachev.wav",
}
# -------------------------------------------------------------------


def fmt_ts(sec):
    sec = int(sec or 0)
    return f"{sec // 60:02d}:{sec % 60:02d}"


def from_pretrained(cls, name, token, **kw):
    """pyannote менял kwarg use_auth_token -> token между версиями."""
    try:
        return cls.from_pretrained(name, use_auth_token=token, **kw)
    except TypeError:
        return cls.from_pretrained(name, token=token, **kw)


def load_asr():
    """gigaam.load_model в разных версиях принимает или не принимает device."""
    try:
        return gigaam.load_model(GIGAAM_MODEL, device=DEVICE)
    except TypeError:
        m = gigaam.load_model(GIGAAM_MODEL)
        try:
            m.to(DEVICE)
        except Exception:
            pass
        return m


def seg_fields(u):
    """Достаём start/end/text из объекта сегмента GigaAM (имена полей зависят от версии)."""
    start = getattr(u, "start", None)
    end = getattr(u, "end", None)
    if start is None or end is None:
        b = getattr(u, "boundaries", None)
        if b is not None and len(b) >= 2:
            start, end = b[0], b[1]
    text = None
    for attr in ("transcription", "text", "norm_text"):
        v = getattr(u, attr, None)
        if isinstance(v, str) and v.strip():
            text = v.strip()
            break
    if text is None:
        text = str(u).strip()
    return float(start or 0.0), float(end or 0.0), text


def speaker_embeddings(emb_inf, audio_path, diar, topk=5, min_dur=2.0):
    """Средний эмбеддинг по нескольким самым длинным сегментам каждого кластера."""
    out = {}
    for spk in diar.labels():
        segs = sorted(((s.end - s.start, s.start, s.end) for s in diar.label_timeline(spk)),
                      reverse=True)
        vecs = []
        for dur, s, e in segs:
            if dur < min_dur:
                continue
            try:
                vecs.append(np.asarray(emb_inf.crop(audio_path, Segment(s, e))).ravel())
            except Exception:
                continue
            if len(vecs) >= topk:
                break
        if vecs:
            out[spk] = np.mean(vecs, axis=0)
    return out


def assign_names(spk_embs, ref_embs):
    """Взаимно-однозначное сопоставление кластер->имя (венгерский алгоритм)."""
    if not spk_embs or not ref_embs:
        return {}
    spks = list(spk_embs.keys())
    names = list(ref_embs.keys())
    D = cdist(np.stack([spk_embs[s] for s in spks]),
              np.stack([ref_embs[n] for n in names]), metric="cosine")
    rows, cols = linear_sum_assignment(D)
    mapping = {spks[r]: names[c] for r, c in zip(rows, cols)}
    for s in spks:
        mapping.setdefault(s, s)
    return mapping


def speaker_at(diar, start, end):
    """Спикер с наибольшим перекрытием отрезка [start, end]."""
    best, best_ov = "?", 0.0
    for turn, _, spk in diar.itertracks(yield_label=True):
        ov = max(0.0, min(turn.end, end) - max(turn.start, start))
        if ov > best_ov:
            best_ov, best = ov, spk
    return best


def select_files():
    args = sys.argv[1:]
    if not args:
        return [Path(p) for p in sorted(glob.glob(str(WAV_DIR / "[0-9][0-9].wav")))]
    files = []
    for a in args:
        p = Path(a)
        if not p.suffix:
            p = WAV_DIR / f"{a}.wav"
        files.append(p)
    return files


def main():
    if not HF_TOKEN:
        sys.exit("HF_TOKEN не задан. Нужен для pyannote (VAD/диаризация). "
                 "Положи в окружение/.env и прими условия моделей "
                 "pyannote/segmentation-3.0, speaker-diarization-3.1, embedding.")
    os.environ["HF_TOKEN"] = HF_TOKEN  # gigaam.transcribe_longform читает токен из окружения
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    files = select_files()
    if not files:
        sys.exit(f"Нет файлов в {WAV_DIR}. Сначала прогони prep_audio.sh.")

    print(f"device={DEVICE} model={GIGAAM_MODEL} файлов={len(files)}")
    asr = load_asr()
    diar_pipe = from_pretrained(Pipeline, "pyannote/speaker-diarization-3.1", HF_TOKEN)
    if DEVICE == "cuda":
        try:
            diar_pipe.to(torch.device("cuda"))
        except Exception:
            pass
    emb_inf = Inference(from_pretrained(Model, "pyannote/embedding", HF_TOKEN), window="whole")

    ref_embs = {n: np.asarray(emb_inf(p)).ravel()
                for n, p in REFS.items() if Path(p).exists()}
    if len(ref_embs) < len(REFS):
        missing = [n for n in REFS if n not in ref_embs]
        print(f"WARN: нет эталонов для {missing} — привязка к именам будет частичной.")

    for wav in files:
        if not wav.exists():
            print(f"SKIP: нет файла {wav}")
            continue
        nn = wav.stem

        print(f"[{nn}] ASR (GigaAM longform)…")
        utterances = asr.transcribe_longform(str(wav))   # VAD-сегменты: start/end/text

        print(f"[{nn}] диаризация (pyannote)…")
        diar_output = diar_pipe(str(wav), num_speakers=N_SPK)
        # Безопасно достаем объект Annotation, сохраняя совместимость со старыми версиями
        diar = getattr(diar_output, "speaker_diarization", diar_output)

        name_map = assign_names(speaker_embeddings(emb_inf, str(wav), diar), ref_embs)

        segments = []
        for u in utterances:
            start, end, text = seg_fields(u)
            spk = name_map.get(speaker_at(diar, start, end), "?")
            segments.append({"start": start, "end": end, "speaker": spk, "text": text})

        json.dump(segments, open(OUT_DIR / f"{nn}_raw.json", "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)
        with open(OUT_DIR / f"{nn}_speaker.txt", "w", encoding="utf-8") as f:
            for s in segments:
                f.write(f"[{fmt_ts(s['start'])}] {s['speaker']}: {s['text']}\n")
        print(f"[{nn}] готово → {OUT_DIR}/{nn}_speaker.txt")


if __name__ == "__main__":
    main()