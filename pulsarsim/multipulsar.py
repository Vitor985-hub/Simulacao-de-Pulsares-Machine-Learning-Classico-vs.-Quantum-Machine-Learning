"""
multipulsar.py — Utilitários para combinar sinais de múltiplos pulsares
observados simultaneamente no mesmo campo de visada do radiotelescópio.

Cada pulsar contribui de forma independente e aditiva para o filterbank
final (aproximação válida: os sinais de rádio se somam linearmente em
potência/intensidade, já que vêm de fontes não correlacionadas).
"""

from typing import List

import numpy as np


def combine_filterbanks(filterbanks: List[np.ndarray]) -> np.ndarray:
    """Soma elemento a elemento vários filterbanks (mesma forma), representando
    a superposição de múltiplas fontes no mesmo feixe do radiotelescópio."""
    if not filterbanks:
        raise ValueError("Nenhum filterbank fornecido para combinar.")
    total = np.zeros_like(filterbanks[0])
    for fb in filterbanks:
        total += fb
    return total
