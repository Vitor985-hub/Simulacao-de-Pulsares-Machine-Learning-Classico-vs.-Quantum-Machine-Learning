"""
plotting.py — Visualizações do sinal simulado: gráfico "waterfall"
(intensidade x tempo x frequência) e perfil de pulso dobrado (folded).
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_waterfall(filterbank: np.ndarray, freqs_mhz: np.ndarray, t: np.ndarray,
                    title: str = "Filterbank simulado", ax=None, cmap="viridis"):
    """Plota o diagrama waterfall (intensidade em função de tempo e frequência)."""
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(9, 5))
    extent = [t[0], t[-1], freqs_mhz[0], freqs_mhz[-1]]
    im = ax.imshow(filterbank, aspect="auto", origin="lower", extent=extent, cmap=cmap)
    ax.set_xlabel("Tempo (s)")
    ax.set_ylabel("Frequência (MHz)")
    ax.set_title(title)
    if own_fig:
        fig.colorbar(im, ax=ax, label="Intensidade")
        return fig, ax
    return ax


def plot_dedispersed_and_profile(dedispersed: np.ndarray, t: np.ndarray, dt: float,
                                  folded_profile: np.ndarray, title_prefix: str = ""):
    """Plota lado a lado: série temporal de-dispersada + perfil médio dobrado."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(t, dedispersed, lw=0.8, color="#2b6cb0")
    axes[0].set_xlabel("Tempo (s)")
    axes[0].set_ylabel("Intensidade somada")
    axes[0].set_title(f"{title_prefix}Série temporal de-dispersada")

    phase_bins = np.linspace(0, 1, len(folded_profile), endpoint=False)
    axes[1].plot(phase_bins, folded_profile, color="#c05621", lw=1.5)
    axes[1].set_xlabel("Fase de rotação")
    axes[1].set_ylabel("Intensidade média")
    axes[1].set_title(f"{title_prefix}Perfil dobrado (folded)")

    fig.tight_layout()
    return fig, axes
