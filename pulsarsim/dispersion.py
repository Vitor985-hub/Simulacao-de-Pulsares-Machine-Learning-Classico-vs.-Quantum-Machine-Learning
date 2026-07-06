"""
dispersion.py — Dispersão do meio interestelar (DM) e scattering.

O meio interestelar é um plasma tênue e ionizado. Frequências mais baixas
viajam mais devagar através dele, causando um atraso que depende de DM
(medida de dispersão, em pc/cm^3) e da frequência de observação.

Fórmula padrão (ex.: Lorimer & Kramer, "Handbook of Pulsar Astronomy"):

    Δt(ν) [s] = k_DM * DM * (1/ν_ref^2 - 1/ν^2)

com ν em MHz, DM em pc/cm^3, e k_DM = 4148.808 (constante de dispersão).

Usamos como referência a frequência mais alta da banda (ν_ref = ν_max):
canais de frequência mais baixa chegam com atraso POSITIVO em relação a ela.
"""

import numpy as np

K_DM = 4148.808  # s * MHz^2 * cm^3 / pc  (constante de dispersão padrão)


def dispersion_delay_relative(freq_mhz: np.ndarray, dm: float, freq_ref_mhz: float) -> np.ndarray:
    """
    Atraso de dispersão (em segundos) de cada canal de frequência em relação
    à frequência de referência (tipicamente o topo da banda, freq_ref_mhz).
    Canais abaixo de freq_ref_mhz têm atraso positivo (chegam depois).
    """
    return K_DM * dm * (1.0 / freq_mhz ** 2 - 1.0 / freq_ref_mhz ** 2)


def scattering_timescale(freq_mhz: np.ndarray, dm: float, ref_freq_mhz: float = 1000.0,
                          tau_ref_s: float = 0.0) -> np.ndarray:
    """
    Escala de tempo de espalhamento (scattering) por canal, escalando como
    tau ~ freq^-4 (aproximação empírica padrão usada na literatura de pulsares).
    Se tau_ref_s = 0, o efeito de scattering é desprezado (retorna zeros).
    """
    if tau_ref_s <= 0:
        return np.zeros_like(freq_mhz)
    return tau_ref_s * (freq_mhz / ref_freq_mhz) ** (-4.0)


def apply_scattering(channel_signal: np.ndarray, dt: float, tau_s: float) -> np.ndarray:
    """
    Convolui um canal com um kernel exponencial unilateral (modelo simples de
    espalhamento por múltiplos caminhos no ISM). tau_s é a escala de tempo
    de espalhamento naquele canal, em segundos.
    """
    if tau_s <= 0:
        return channel_signal
    n_kernel = max(int(10 * tau_s / dt), 1)
    t_kernel = np.arange(n_kernel) * dt
    kernel = np.exp(-t_kernel / tau_s)
    kernel /= kernel.sum()
    convolved = np.convolve(channel_signal, kernel, mode="full")[: len(channel_signal)]
    return convolved
