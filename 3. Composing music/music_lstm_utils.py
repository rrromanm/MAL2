"""Helpers and model classes for Portfolio 3 (RNN language models for music).

This module is imported by ``assignment_music_rnn.ipynb``. It is intentionally
self-contained and depends only on numpy, scipy, torch and pretty_midi so the
notebook runs on a plain CPU machine (macOS / Windows / Linux) inside VS Code.

Token representation
--------------------
Each non-drum MIDI note becomes a ``(pitch, duration_bin)`` pair ("character").
Durations are binned against a fixed eighth-note base (``BASE_DUR`` seconds).
The vocabulary is every unique pair seen in the corpus, preceded by three
special tokens: ``<pad>`` (id 0), ``<unk>`` (id 1), ``<bos>`` (id 2).
``itos`` maps an id to either a special-token *string* or a ``(pitch, dur)``
*tuple*; ``stoi`` is the inverse.
"""

from pathlib import Path
import os
import shutil
import subprocess

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

import pretty_midi

# Eighth-note base (seconds) used for both binning durations on the way in and
# reconstructing note lengths on the way out, so encode/decode stay consistent.
BASE_DUR = 0.125
MAX_DUR_BIN = 15            # duration bins are clamped to [0, MAX_DUR_BIN]
SPECIALS = ["<pad>", "<unk>", "<bos>"]
PAD, UNK, BOS = 0, 1, 2


# --------------------------------------------------------------------------- #
# Data: MIDI <-> token sequences
# --------------------------------------------------------------------------- #
def list_midi_files(directory):
    """Return a sorted list of .mid/.midi file paths under ``directory``."""
    d = Path(directory)
    if not d.exists():
        return []
    files = list(d.rglob("*.mid")) + list(d.rglob("*.midi"))
    return [str(p) for p in sorted(files)]


def make_sequences_from_midis(paths, base_dur=BASE_DUR, max_dur_bin=MAX_DUR_BIN):
    """Turn each MIDI file into a list of ``(pitch, duration_bin)`` tuples.

    Non-drum notes are sorted in time; each note's duration is binned against
    ``base_dur``. Files that fail to parse or contain no usable notes are
    skipped. Returns a list (one entry per usable file) of token sequences.
    """
    seqs = []
    for p in paths:
        try:
            pm = pretty_midi.PrettyMIDI(str(p))
        except Exception:
            continue
        notes = []
        for inst in pm.instruments:
            if inst.is_drum:
                continue
            notes.extend(inst.notes)
        if not notes:
            continue
        notes.sort(key=lambda n: (n.start, n.pitch))
        seq = []
        for n in notes:
            dur_bin = int(round((n.end - n.start) / base_dur)) - 1
            dur_bin = max(0, min(max_dur_bin, dur_bin))
            seq.append((int(n.pitch), int(dur_bin)))
        if len(seq) >= 2:
            seqs.append(seq)
    return seqs


def build_vocab(seqs):
    """Build ``stoi, itos, V`` from token sequences (specials first)."""
    stoi = {tok: i for i, tok in enumerate(SPECIALS)}
    itos = {i: tok for i, tok in enumerate(SPECIALS)}
    nxt = len(SPECIALS)
    for s in seqs:
        for tok in s:
            if tok not in stoi:
                stoi[tok] = nxt
                itos[nxt] = tok
                nxt += 1
    return stoi, itos, nxt


def encode_sequence(seq, stoi):
    """Map a ``(pitch, dur)`` sequence to a list of token ids (UNK fallback)."""
    return [stoi.get(tok, UNK) for tok in seq]


class MusicTokenDataset(Dataset):
    """Sliding windows of length ``seq_len`` for next-step prediction.

    Each item is ``(x, y)`` of shape ``(seq_len,)`` where ``y`` is ``x`` shifted
    one step. Sequences shorter than ``seq_len + 1`` are right-padded with PAD
    (id 0); the training loss ignores PAD via ``ignore_index=0``.
    """

    def __init__(self, token_seqs, seq_len, step=None):
        self.seq_len = seq_len
        step = step if step is not None else max(1, seq_len // 8)
        self.windows = []
        for seq in token_seqs:
            if len(seq) < 2:
                continue
            need = seq_len + 1
            i = 0
            while True:
                chunk = seq[i:i + need]
                if len(chunk) < need:
                    chunk = chunk + [PAD] * (need - len(chunk))
                    self.windows.append(chunk)
                    break
                self.windows.append(chunk)
                if i + need >= len(seq):
                    break
                i += step

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        chunk = self.windows[idx]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
class OneHotGRUModel(nn.Module):
    """One-hot input (dim V) -> GRU -> linear. A classic char-RNN baseline."""

    def __init__(self, vocab_size, hidden, num_layers=1, dropout=0.1):
        super().__init__()
        self.V = vocab_size
        self.gru = nn.GRU(
            vocab_size, hidden, num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(hidden, vocab_size)

    def forward(self, x):
        oh = F.one_hot(x, self.V).float()
        h, _ = self.gru(oh)
        return self.out(self.drop(h))


class EmbLSTMModel(nn.Module):
    """Learnable embedding -> LSTM -> linear."""

    def __init__(self, vocab_size, emb_dim=64, hidden=256, num_layers=1, dropout=0.1):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD)
        self.lstm = nn.LSTM(
            emb_dim, hidden, num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(hidden, vocab_size)

    def forward(self, x):
        e = self.emb(x)
        h, _ = self.lstm(e)
        return self.out(self.drop(h))


# --------------------------------------------------------------------------- #
# Sampling
# --------------------------------------------------------------------------- #
@torch.no_grad()
def sample_sequence(model, device, stoi, itos, V, seed_ids, n_new,
                    temperature=1.0, repetition_penalty=1.0,
                    context=256, generator=None):
    """Autoregressively sample ``n_new`` tokens after ``seed_ids``.

    Multinomial sampling on softmax(logits / temperature) of the last step.
    A CTRL-style ``repetition_penalty`` discourages already-used tokens, and
    special tokens (PAD/UNK/BOS) are never sampled.
    """
    model.eval()
    model = model.to(device)
    ids = list(seed_ids)
    for _ in range(n_new):
        ctx = ids[-context:]
        x = torch.tensor([ctx], dtype=torch.long, device=device)
        logits = model(x)[0, -1].float()
        if repetition_penalty and repetition_penalty != 1.0:
            for t in set(ids):
                if logits[t] > 0:
                    logits[t] /= repetition_penalty
                else:
                    logits[t] *= repetition_penalty
        logits = logits / max(1e-6, temperature)
        logits[PAD] = float("-inf")
        logits[UNK] = float("-inf")
        logits[BOS] = float("-inf")
        probs = F.softmax(logits, dim=-1)
        nxt = int(torch.multinomial(probs, 1, generator=generator).item())
        ids.append(nxt)
    return ids


# --------------------------------------------------------------------------- #
# Tokens -> MIDI -> audio
# --------------------------------------------------------------------------- #
def token_ids_to_midi(ids, itos, base_dur=BASE_DUR, program=0):
    """Turn a list of token ids into a monophonic ``pretty_midi.PrettyMIDI``."""
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=program)
    t = 0.0
    for tid in ids:
        tok = itos.get(int(tid))
        if not isinstance(tok, tuple) or len(tok) != 2:
            continue  # skip special tokens
        pitch, dur_bin = tok
        if not (0 < pitch < 128):
            continue
        dur = (dur_bin + 1) * base_dur
        inst.notes.append(
            pretty_midi.Note(velocity=90, pitch=int(pitch), start=t, end=t + dur)
        )
        t += dur
    pm.instruments.append(inst)
    return pm


def simple_render_pm(pm, timbre="gru", sr=22050):
    """Render a PrettyMIDI object to a normalized mono float32 waveform.

    Uses pretty_midi's pure-numpy ``synthesize`` (no fluidsynth/soundfonts).
    The ``timbre`` argument swaps the oscillator so the two models sound
    distinct: a sine for the GRU, a softer triangle-ish wave for the LSTM.
    """
    if timbre == "lstm":
        wave = lambda x: (2.0 / np.pi) * np.arcsin(np.sin(x))  # triangle
    else:
        wave = np.sin
    try:
        audio = pm.synthesize(fs=sr, wave=wave)
    except Exception:
        audio = np.zeros(sr, dtype=np.float32)
    audio = np.asarray(audio, dtype=np.float32)
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 0:
        audio = audio / peak * 0.9
    return audio


def write_wav(path, audio, sr=22050):
    """Write a float waveform to a 16-bit PCM WAV file."""
    from scipy.io import wavfile
    a = np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0)
    wavfile.write(str(path), int(sr), (a * 32767.0).astype(np.int16))


def to_mp3(wav_path, mp3_path):
    """Convert a WAV to MP3 via ffmpeg if available; return True on success.

    Returns False (and writes nothing) when no ffmpeg binary is found, so the
    notebook can fall back to WAV cleanly without leaving 0-byte mp3 files.
    """
    exe = os.environ.get("FFMPEG_BINARY") or shutil.which("ffmpeg")
    if not exe:
        return False
    try:
        subprocess.run(
            [exe, "-y", "-i", str(wav_path), str(mp3_path)],
            check=True, capture_output=True,
        )
        return True
    except Exception:
        return False


# --------------------------------------------------------------------------- #
# Synthetic data generation (so the notebook runs without a MIDI corpus)
# --------------------------------------------------------------------------- #
_SCALES = {
    "lofi": [0, 3, 5, 7, 10],                 # minor pentatonic
    "anthems": [0, 2, 4, 5, 7, 9, 11],        # major scale
}


def _write_procedural_midi(path, rng, scale, root, n_notes, base_dur=BASE_DUR):
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=0)
    t = 0.0
    for _ in range(n_notes):
        degree = int(rng.choice(scale))
        octave = root + 12 * int(rng.choice([0, 0, 1]))
        pitch = int(np.clip(octave + degree, 21, 108))
        dur = base_dur * int(rng.choice([1, 2, 2, 4]))
        inst.notes.append(pretty_midi.Note(90, pitch, t, t + dur))
        t += dur
    pm.instruments.append(inst)
    pm.write(str(path))


def generate_sample_data(lofi_dir, anthems_dir, n_lofi=120, n_anthems=120,
                         min_notes=160, max_notes=260, seed=0):
    """Create simple procedural MIDI files so the pipeline is runnable.

    Drop your own .mid files into the same folders to use a real corpus
    instead; this only writes files when the folders are empty.
    """
    rng = np.random.default_rng(seed)
    made = 0
    for folder, name, n_files, root in (
        (lofi_dir, "lofi", n_lofi, 48),
        (anthems_dir, "anthems", n_anthems, 60),
    ):
        d = Path(folder)
        d.mkdir(parents=True, exist_ok=True)
        if list_midi_files(d):
            continue  # already has MIDI; leave it alone
        scale = _SCALES[name]
        for i in range(n_files):
            n_notes = int(rng.integers(min_notes, max_notes + 1))
            _write_procedural_midi(
                d / f"{name}_{i:03d}.mid", rng, scale, root, n_notes
            )
            made += 1
    return made
