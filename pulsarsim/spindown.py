"""
spindown.py — Relações físicas entre período (P), sua derivada (Ṗ), campo
magnético (B) e perda de energia rotacional (spin-down), no modelo padrão de
freio magneto-dipolar (braking index n=3).

Fórmulas padrão de astrofísica de pulsares (Lorimer & Kramer):

    B_surface [G]  ≈ 3.2e19 * sqrt(P * Pdot)          (P em s, Pdot adimensional)
    Idade característica τ_c = P / (2 * Pdot)          (s)
    Luminosidade de spin-down: Ė = 4*pi^2 * I * Pdot / P^3   (erg/s)

I (momento de inércia) ≈ 1e45 g*cm^2 é o valor canônico para uma estrela de
nêutrons de ~1.4 massas solares.
"""

import numpy as np

I_MOMENT_OF_INERTIA = 1.0e45  # g * cm^2, valor canônico
_B_CONSTANT = 3.2e19  # G * s^-1 (constante empírica do modelo de freio magneto-dipolar)
SECONDS_PER_YEAR = 365.25 * 24 * 3600.0


def magnetic_field_from_p_pdot(period_s: float, pdot: float) -> float:
    """Campo magnético de superfície estimado (Gauss) a partir de P e Ṗ."""
    if pdot <= 0:
        return 0.0
    return _B_CONSTANT * np.sqrt(period_s * pdot)


def pdot_from_magnetic_field(period_s: float, b_field_gauss: float) -> float:
    """Ṗ implícito, dado um campo magnético alvo e o período atual."""
    if b_field_gauss <= 0:
        return 0.0
    return (b_field_gauss / _B_CONSTANT) ** 2 / period_s


def characteristic_age_years(period_s: float, pdot: float) -> float:
    """Idade característica do pulsar (anos), τ_c = P / (2 Ṗ)."""
    if pdot <= 0:
        return np.inf
    return (period_s / (2.0 * pdot)) / SECONDS_PER_YEAR


def spindown_luminosity(period_s: float, pdot: float, moment_of_inertia: float = I_MOMENT_OF_INERTIA) -> float:
    """Luminosidade de spin-down Ė (erg/s): energia rotacional perdida por segundo."""
    if pdot <= 0 or period_s <= 0:
        return 0.0
    return 4.0 * np.pi ** 2 * moment_of_inertia * pdot / period_s ** 3


def period_with_spindown(t_s: np.ndarray, period0_s: float, pdot: float) -> np.ndarray:
    """
    Período instantâneo ao longo do tempo, assumindo Ṗ aproximadamente
    constante durante a janela de observação simulada (válido pois Pdot é
    extremamente pequeno em escalas de segundos/minutos, mas incluído para
    completude física e para datasets de longa duração/anos simulados).
    """
    return period0_s + pdot * t_s


def flux_scaling_from_distance(amplitude_ref: float, distance_kpc: float, distance_ref_kpc: float = 1.0) -> float:
    """
    Escala a amplitude do pulso pela lei do inverso do quadrado da distância:
    S ∝ 1 / d^2. `amplitude_ref` é a amplitude na distância de referência.
    """
    distance_kpc = max(distance_kpc, 1e-3)
    return amplitude_ref * (distance_ref_kpc / distance_kpc) ** 2
