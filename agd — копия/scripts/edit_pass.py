#!/usr/bin/env python3
"""
Стадия B — литературная редакция по чанкам с КОНТРОЛЕМ УСАДКИ.

Зачем по чанкам: на маленьком фрагменте модель не может "зарезюмировать всё",
а значит не возникает эффекта "10 страниц -> 3". После каждого чанка считаем
ratio = (длина out без пробелов) / (длина in без пробелов); если < 0.85 — флаг REVIEW.

Вход:  out/clean/NN_clean.md   (результат стадии A — точный verbatim, источник истины)
Выход: out/final/NN_final.md, out/final/NN_ratio.txt

Нужен ANTHROPIC_API_KEY в .env. Запуск после стадии A.
Альтернатива: ту же правку можно делать прямо в сессии Claude Code, без этого скрипта.

    python scripts/edit_pass.py            # все out/clean/*_clean.md
    python scripts/edit_pass.py 01         # только 01
"""
import os
import re
import sys
import glob
from pathlib import Path

from anthropic import Anthropic

MODEL = "claude-opus-4-8"      # качество; для скорости/цены — "claude-sonnet-4-6"
CHUNK_CHARS = 4000             # ~1000-1500 токенов на чанк
RATIO_MIN = 0.85
TEMPERATURE = 0.2

SYSTEM = (
    "Ты — научный редактор беседы по традиционным ценностям (философия, право, богословие). "
    "Участники: протоиерей Андрей Ткачёв (священник), Константин Малофеев (бизнесмен, монархист), "
    "Александр Дугин (философ-традиционалист). "
    "Преобразуй фрагмент в связный, литературно выверенный текст, сохранив ВЕСЬ смысл, "
    "все содержательные детали, авторский стиль каждого говорящего и порядок мыслей. "
    "Запрещено: добавлять то, чего не было; удалять содержание; менять термины и имена. "
    "Сохраняй метки спикеров и таймкоды в начале реплик. Верни ТОЛЬКО отредактированный текст."
)

TURN_RE = re.compile(r"^\[\d{2}:\d{2}\]")


def chunk_md(text, budget=CHUNK_CHARS):
    """Режем по границам реплик (строки '[мм:сс] Имя:'), не разрывая реплику."""
    chunks, cur = [], ""
    for ln in text.splitlines(keepends=True):
        if TURN_RE.match(ln) and len(cur) >= budget:
            chunks.append(cur)
            cur = ""
        cur += ln
    if cur.strip():
        chunks.append(cur)
    return chunks


def nospace_len(s):
    return len(re.sub(r"\s+", "", s))


def edit_chunk(client, chunk):
    msg = client.messages.create(
        model=MODEL,
        max_tokens=4096,
        temperature=TEMPERATURE,
        system=SYSTEM,
        messages=[{"role": "user", "content": chunk}],
    )
    return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")


def select_files():
    args = sys.argv[1:]
    if not args:
        return sorted(glob.glob("out/clean/*_clean.md"))
    return [a if a.endswith(".md") else f"out/clean/{a}_clean.md" for a in args]


def main():
    client = Anthropic()  # ключ берётся из ANTHROPIC_API_KEY
    out_dir = Path("out/final")
    out_dir.mkdir(parents=True, exist_ok=True)

    for path in select_files():
        path = Path(path)
        if not path.exists():
            print(f"SKIP: нет {path}")
            continue
        nn = path.stem.replace("_clean", "")
        text = path.read_text(encoding="utf-8")
        chunks = chunk_md(text)
        edited, report = [], []
        for i, ch in enumerate(chunks, 1):
            out = edit_chunk(client, ch)
            ratio = nospace_len(out) / max(1, nospace_len(ch))
            flagged = ratio < RATIO_MIN
            report.append(f"чанк {i:02d}: ratio={ratio:.2f}" + ("  ⚠ REVIEW" if flagged else ""))
            if flagged:
                out += f"\n\n<!-- ⚠ REVIEW ratio={ratio:.2f}: возможна усадка, проверить вручную -->"
            edited.append(out if out.endswith("\n") else out + "\n")
            print(f"[{nn}] {report[-1]}")
        total = nospace_len("".join(edited)) / max(1, nospace_len(text))
        (out_dir / f"{nn}_final.md").write_text("".join(edited), encoding="utf-8")
        (out_dir / f"{nn}_ratio.txt").write_text(
            "\n".join(report) + f"\n\nИТОГО ratio={total:.2f}\n", encoding="utf-8")
        verdict = "OK" if total >= RATIO_MIN else "⚠ ПРОВЕРИТЬ УСАДКУ"
        print(f"[{nn}] итог ratio={total:.2f} [{verdict}] → out/final/{nn}_final.md")


if __name__ == "__main__":
    main()
