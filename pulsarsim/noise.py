"""
noise.py — Ruído branco e interferência de rádio-frequência (RFI).

Dois tipos de RFI são modelados, cobrindo os casos mais comuns em dados
reais de radiotelescópios:

  * narrowband RFI ("birdie"): um tom periódico presente em um ou poucos
    canais de frequência específicos (ex.: contaminação de um transmissor
    de rádio/celular/satélite).
  * broadband impulsive RFI: pulsos curtos e intensos que aparecem em TODOS
    os canais simultaneamente (ex.: radares, raios/atividade elétrica).
"""

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


def white_noise(shape, std: float, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Ruído branco gaussiano com desvio-padrão `std`."""
    rng = rng or np.random.default_rng()
    return rng.normal(loc=0.0, scale=std, size=shape)


@dataclass
class NarrowbandRFI:
    channel_frac: float      # posição relativa do canal afetado (0=freq mais baixa, 1=mais alta)
    period_s: float          # período do tom de interferência (s)
    amplitude: float         # amplitude do tom
    duty_cycle: float = 0.5  # fração do tempo em que o tom fica "ligado"


@dataclass
class BroadbandImpulseRFI:
    rate_hz: float           # taxa média de ocorrência de impulsos (eventos/s)
    amplitude: float         # amplitude típica do impulso
    width_s: float = 0.001   # duração típica de cada impulso (s)


@dataclass
class RFIConfig:
    narrowband: List[NarrowbandRFI] = field(default_factory=list)
    broadband: List[BroadbandImpulseRFI] = field(default_factory=list)


def apply_narrowband_rfi(filterbank: np.ndarray, freqs_mhz: np.ndarray, t: np.ndarray,
                          rfi: NarrowbandRFI) -> None:
    """Adiciona um tom periódico narrowband a um único canal (in-place)."""
    n_channels = len(freqs_mhz)
    ch_idx = int(np.clip(rfi.channel_frac, 0.0, 1.0) * (n_channels - 1))
    tone_phase = np.mod(t / rfi.period_s, 1.0)
    tone = np.where(tone_phase < rfi.duty_cycle, rfi.amplitude, 0.0)
    filterbank[ch_idx, :] += tone


def apply_broadband_rfi(filterbank: np.ndarray, t: np.ndarray, rfi: BroadbandImpulseRFI,
                         rng: Optional[np.random.Generator] = None) -> None:
    """Adiciona impulsos broadband (mesmo padrão temporal em todos os canais), in-place."""
    rng = rng or np.random.default_rng()
    duration = t[-1] - t[0] if len(t) > 1 else 0.0
    n_events = rng.poisson(rfi.rate_hz * duration)
    dt = t[1] - t[0] if len(t) > 1 else 1.0
    width_samples = max(int(rfi.width_s / dt), 1)
    event_times = rng.uniform(t[0], t[-1], size=n_events)
    impulse_track = np.zeros_like(t)
    for et in event_times:
        idx = int((et - t[0]) / dt)
        lo, hi = max(idx - width_samples // 2, 0), min(idx + width_samples // 2 + 1, len(t))
        impulse_track[lo:hi] += rfi.amplitude
    filterbank += impulse_track[np.newaxis, :]
