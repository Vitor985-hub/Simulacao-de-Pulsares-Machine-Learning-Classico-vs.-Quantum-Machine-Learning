"""
pulse.py — Perfil de pulso e geração do trem de pulsos no domínio da fase.

O perfil de um pulsar é modelado como uma função periódica da fase de rotação
(phi em [0, 1)), usando uma distribuição de von Mises (o análogo circular da
gaussiana). Isso evita descontinuidades artificiais nas bordas da fase 0/1,
que uma gaussiana comum produziria.
"""

from dataclasses import dataclass, field
from typing import List

import numpy as np

# Constante para converter FWHM (em radianos) em kappa de von Mises,
# assumindo aproximação gaussiana para kappa grande:
# FWHM_rad = 2*sqrt(2*ln(2)) * sigma ,  sigma = 1/sqrt(kappa)
_FWHM_TO_SIGMA = 2.0 * np.sqrt(2.0 * np.log(2.0))


def _width_to_kappa(width_fraction: float) -> float:
    """Converte a largura do pulso (fração do período, ~FWHM) em kappa (von Mises)."""
    width_fraction = max(width_fraction, 1e-4)
    fwhm_rad = 2.0 * np.pi * width_fraction
    sigma = fwhm_rad / _FWHM_TO_SIGMA
    kappa = 1.0 / (sigma ** 2)
    return kappa


def von_mises_component(phase: np.ndarray, center: float, width: float, amplitude: float) -> np.ndarray:
    """
    Um componente de pulso (ex.: pulso principal ou interpulso).

    Parameters
    ----------
    phase : array de fase de rotação em [0, 1)
    center : fase central do pico do pulso (0 a 1)
    width : largura do pulso como fração do período (duty cycle), ~FWHM
    amplitude : amplitude de pico (unidade arbitrária de intensidade)
    """
    kappa = _width_to_kappa(width)
    theta = 2.0 * np.pi * (phase - center)
    profile = np.exp(kappa * (np.cos(theta) - 1.0))
    return amplitude * profile


@dataclass
class PulseComponent:
    """Um componente individual do perfil (pulso principal, interpulso, etc.)."""
    center: float       # fase central (0-1)
    width: float        # largura como fração do período
    amplitude: float    # amplitude relativa


@dataclass
class PulseProfile:
    """
    Perfil completo de um pulsar, podendo ter múltiplos componentes
    (ex.: pulso principal + interpulso, comum em pulsares como o Crab).
    """
    components: List[PulseComponent] = field(default_factory=lambda: [PulseComponent(0.5, 0.05, 1.0)])

    def evaluate(self, phase: np.ndarray) -> np.ndarray:
        """Avalia o perfil somado de todos os componentes numa array de fases."""
        total = np.zeros_like(phase, dtype=float)
        for comp in self.components:
            total += von_mises_component(phase, comp.center, comp.width, comp.amplitude)
        return total


def phase_from_time(t: np.ndarray, period: float, t0: float = 0.0) -> np.ndarray:
    """Converte tempo (s) em fase de rotação [0, 1), dado período constante."""
    return np.mod((t - t0) / period, 1.0)
