"""
glitch.py — Injeção de glitches: saltos abruptos no período de rotação,
seguidos de recuperação parcial exponencial. Fenômeno real observado em
pulsares jovens (ex.: Vela, Crab), associado a rearranjos internos da
crosta/núcleo superfluido da estrela de nêutrons.

Modelo (padrão na literatura de timing de pulsares):

    Um glitch causa uma diminuição súbita do período (o pulsar "acelera"),
    seguida de uma recuperação parcial exponencial de volta a um valor
    próximo (mas geralmente não idêntico) ao período pré-glitch.

    P(t) = P0                                                  , t < t_g
    P(t) = P0 - ΔP_perm - ΔP_transient * exp(-(t - t_g) / tau)  , t >= t_g

Calculamos a fase por integração numérica de 1/P(t), para não assumir
período constante durante o evento.
"""

from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class GlitchEvent:
    time_s: float              # instante do glitch (s), relativo ao início da observação
    delta_p_over_p: float      # fração do salto TOTAL no período (positivo; período diminui)
    transient_fraction: float = 0.7  # fração do salto que é transiente (recupera com o tempo)
    recovery_timescale_s: float = 50.0  # tau da recuperação exponencial (s)


def apply_glitches_to_period(period_base: np.ndarray, t_s: np.ndarray, glitches: List[GlitchEvent]) -> np.ndarray:
    """
    Aplica uma lista de glitches sobre um período-base (que pode já variar no
    tempo, ex.: por spin-down). Cada glitch subtrai um salto (parcialmente
    permanente, parcialmente transiente com recuperação exponencial) a partir
    do seu instante de ocorrência.
    """
    period = period_base.copy()
    for glitch in glitches:
        period0_at_glitch = np.interp(glitch.time_s, t_s, period_base)
        delta_p_total = glitch.delta_p_over_p * period0_at_glitch
        delta_p_perm = (1.0 - glitch.transient_fraction) * delta_p_total
        delta_p_transient = glitch.transient_fraction * delta_p_total

        mask = t_s >= glitch.time_s
        dt_since_glitch = t_s[mask] - glitch.time_s
        period[mask] -= delta_p_perm + delta_p_transient * np.exp(-dt_since_glitch / glitch.recovery_timescale_s)
    return period


def instantaneous_period(t_s: np.ndarray, period0_s: float, glitch: GlitchEvent) -> np.ndarray:
    """Período instantâneo P(t) considerando um único evento de glitch (caso simples)."""
    period_base = np.full_like(t_s, period0_s, dtype=float)
    return apply_glitches_to_period(period_base, t_s, [glitch])


def phase_with_glitch(t_s: np.ndarray, period0_s: float, glitch: GlitchEvent) -> np.ndarray:
    """
    Fase de rotação acumulada [0, 1) ao longo do tempo, considerando o glitch.
    Integra numericamente 1/P(t) (frequência de rotação instantânea) para
    obter a fase acumulada de forma fisicamente consistente.
    """
    period_t = instantaneous_period(t_s, period0_s, glitch)
    return np.mod(unwrapped_phase_from_period(t_s, period_t), 1.0)


def unwrapped_phase_from_period(t_s: np.ndarray, period_t: np.ndarray) -> np.ndarray:
    """
    Fase acumulada NÃO reduzida ao intervalo [0,1) (ou seja, cresce
    monotonicamente com o tempo). Necessária para poder interpolar a fase em
    tempos deslocados (ex.: atraso de dispersão) sem os artefatos de
    descontinuidade que o módulo 1 introduziria antes da interpolação.
    """
    freq_t = 1.0 / period_t
    if len(t_s) > 1:
        dt = np.diff(t_s, prepend=2 * t_s[0] - t_s[1])
    else:
        dt = np.zeros_like(t_s)
    return np.cumsum(freq_t * dt)


def period_with_pdot_and_glitches(t_s: np.ndarray, period0_s: float, pdot: float,
                                   glitches: List[GlitchEvent]) -> np.ndarray:
    """Período instantâneo combinando deriva de spin-down (linear) e glitches."""
    period_base = period0_s + pdot * t_s
    if glitches:
        period_base = apply_glitches_to_period(period_base, t_s, glitches)
    return period_base
