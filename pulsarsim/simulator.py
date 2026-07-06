"""
simulator.py — Orquestrador de alto nível do pulsarsim.

Classes principais:
    PulsarConfig      -> parâmetros físicos de UM pulsar
    ObservationConfig -> parâmetros do "instrumento" (radiotelescópio simulado)
    PulsarSimulator    -> gera o filterbank final (intensidade x tempo x freq)
                          combinando um ou mais pulsares, dispersão, scattering,
                          ruído branco e RFI.
"""

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from . import dispersion, spindown, noise as noise_mod
from .pulse import PulseProfile, PulseComponent
from .glitch import GlitchEvent, period_with_pdot_and_glitches, unwrapped_phase_from_period
from .multipulsar import combine_filterbanks


@dataclass
class PulsarConfig:
    """Parâmetros físicos de um único pulsar simulado."""

    name: str = "PSR_SIM"

    # Rotação
    period_s: float = 1.0
    pdot: float = 0.0                      # Ṗ (s/s); se 0 e b_field_gauss for dado, é calculado
    b_field_gauss: Optional[float] = None  # campo magnético de superfície (G)

    # Propagação / distância
    dm: float = 30.0                       # medida de dispersão (pc/cm^3)
    distance_kpc: float = 1.0              # distância (kpc)

    # Perfil de pulso
    amplitude: float = 10.0                # intensidade de referência do feixe (a 1 kpc)
    duty_cycle: float = 0.05               # largura do pulso (fração do período)
    interpulse: bool = False               # incluir um interpulso?
    interpulse_amplitude_frac: float = 0.3  # amplitude do interpulso relativa ao pulso principal
    interpulse_offset: float = 0.5         # deslocamento de fase do interpulso

    # Espectro e propagação no ISM
    spectral_index: float = -1.6           # S(freq) ~ freq^spectral_index (pulsares são + brilhantes em baixa freq)
    scattering_tau_ref_s: float = 0.0      # timescale de scattering na freq de referência (0 = desliga)

    # Eventos
    glitches: List[GlitchEvent] = field(default_factory=list)

    # Fase inicial (útil para múltiplos pulsares não alinhados)
    phase0: float = 0.0

    def effective_pdot(self) -> float:
        """Retorna Ṗ efetivo: usa `pdot` diretamente, ou o calcula a partir de B se fornecido."""
        if self.pdot > 0:
            return self.pdot
        if self.b_field_gauss:
            return spindown.pdot_from_magnetic_field(self.period_s, self.b_field_gauss)
        return 0.0

    def effective_b_field(self) -> float:
        """Campo magnético (G): usa `b_field_gauss` diretamente, ou o infere de P e Ṗ."""
        if self.b_field_gauss:
            return self.b_field_gauss
        return spindown.magnetic_field_from_p_pdot(self.period_s, self.effective_pdot())

    def build_profile(self) -> PulseProfile:
        components = [PulseComponent(center=0.5, width=self.duty_cycle, amplitude=1.0)]
        if self.interpulse:
            ip_center = np.mod(0.5 + self.interpulse_offset, 1.0)
            components.append(
                PulseComponent(center=ip_center, width=self.duty_cycle, amplitude=self.interpulse_amplitude_frac)
            )
        return PulseProfile(components=components)


@dataclass
class ObservationConfig:
    """Parâmetros do 'instrumento': o radiotelescópio simulado."""

    n_channels: int = 128
    freq_low_mhz: float = 1200.0
    freq_high_mhz: float = 1500.0
    sample_time_s: float = 0.001           # resolução temporal (s) -> aqui, 1 ms
    duration_s: float = 2.0                # duração da observação simulada (s)
    noise_std: float = 1.0                 # desvio-padrão do ruído branco
    rfi_config: noise_mod.RFIConfig = field(default_factory=noise_mod.RFIConfig)
    seed: Optional[int] = None


class PulsarSimulator:
    """
    Orquestra a geração de um filterbank sintético (intensidade x tempo x
    frequência), combinando um ou mais pulsares, dispersão interestelar,
    scattering, ruído branco e RFI — tudo configurável via PulsarConfig /
    ObservationConfig.
    """

    def __init__(self, obs_config: ObservationConfig):
        self.obs = obs_config
        self.pulsars: List[PulsarConfig] = []
        self._rng = np.random.default_rng(obs_config.seed)

    def add_pulsar(self, pulsar_config: PulsarConfig) -> "PulsarSimulator":
        self.pulsars.append(pulsar_config)
        return self  # permite encadear chamadas (.add_pulsar(...).add_pulsar(...))

    # ------------------------------------------------------------------ #
    # Geração de um único pulsar (dispersão + espectro + scattering)
    # ------------------------------------------------------------------ #
    def _generate_single_pulsar(self, cfg: PulsarConfig, t: np.ndarray, freqs_mhz: np.ndarray) -> np.ndarray:
        dt = self.obs.sample_time_s
        freq_ref = freqs_mhz.max()  # canal de referência: topo da banda (sem atraso)

        # Atraso máximo possível (na frequência mais baixa) -> define quanto
        # precisamos estender o array de tempo para trás antes de t=0.
        max_delay = dispersion.dispersion_delay_relative(freqs_mhz.min(), cfg.dm, freq_ref)
        pad_samples = int(np.ceil(max_delay / dt)) + 2
        t_ext = np.arange(-pad_samples, len(t)) * dt + t[0]

        pdot = cfg.effective_pdot()
        period_t_ext = period_with_pdot_and_glitches(t_ext, cfg.period_s, pdot, cfg.glitches)
        phase_unwrapped_ext = unwrapped_phase_from_period(t_ext, period_t_ext) + cfg.phase0

        profile = cfg.build_profile()
        filterbank = np.zeros((len(freqs_mhz), len(t)))

        for i, freq_ch in enumerate(freqs_mhz):
            delay = dispersion.dispersion_delay_relative(freq_ch, cfg.dm, freq_ref)
            sample_times = t - delay
            phase_ch = np.mod(np.interp(sample_times, t_ext, phase_unwrapped_ext), 1.0)

            channel_signal = profile.evaluate(phase_ch)

            # Escala espectral: pulsares tipicamente mais brilhantes em freq baixas
            spectral_scale = (freq_ch / freq_ref) ** cfg.spectral_index
            # Escala por distância (lei do inverso do quadrado)
            flux_scale = spindown.flux_scaling_from_distance(cfg.amplitude, cfg.distance_kpc)
            channel_signal = channel_signal * spectral_scale * flux_scale

            if cfg.scattering_tau_ref_s > 0:
                tau_ch = dispersion.scattering_timescale(freq_ch, cfg.dm, tau_ref_s=cfg.scattering_tau_ref_s)
                channel_signal = dispersion.apply_scattering(channel_signal, dt, tau_ch)

            filterbank[i, :] = channel_signal

        return filterbank

    # ------------------------------------------------------------------ #
    # API principal
    # ------------------------------------------------------------------ #
    def generate(self):
        """
        Gera o filterbank completo (todos os pulsares + ruído + RFI).

        Retorna
        -------
        filterbank : np.ndarray, shape (n_channels, n_samples)
        freqs_mhz  : np.ndarray, shape (n_channels,)
        t          : np.ndarray, shape (n_samples,) — tempo em segundos
        metadata   : dict — parâmetros de "ground truth" de cada pulsar (útil para rótulos de ML)
        """
        if not self.pulsars:
            raise ValueError("Nenhum pulsar adicionado. Use .add_pulsar(PulsarConfig(...)) antes de generate().")

        t = np.arange(0.0, self.obs.duration_s, self.obs.sample_time_s)
        freqs_mhz = np.linspace(self.obs.freq_low_mhz, self.obs.freq_high_mhz, self.obs.n_channels)

        per_pulsar_fb = [self._generate_single_pulsar(cfg, t, freqs_mhz) for cfg in self.pulsars]
        filterbank = combine_filterbanks(per_pulsar_fb)

        # Ruído branco
        filterbank += noise_mod.white_noise(filterbank.shape, self.obs.noise_std, rng=self._rng)

        # RFI
        for nb in self.obs.rfi_config.narrowband:
            noise_mod.apply_narrowband_rfi(filterbank, freqs_mhz, t, nb)
        for bb in self.obs.rfi_config.broadband:
            noise_mod.apply_broadband_rfi(filterbank, t, bb, rng=self._rng)

        metadata = {
            "pulsars": [
                {
                    "name": cfg.name,
                    "period_s": cfg.period_s,
                    "pdot": cfg.effective_pdot(),
                    "b_field_gauss": cfg.effective_b_field(),
                    "dm": cfg.dm,
                    "distance_kpc": cfg.distance_kpc,
                    "duty_cycle": cfg.duty_cycle,
                    "n_glitches": len(cfg.glitches),
                    "spindown_luminosity_erg_s": spindown.spindown_luminosity(cfg.period_s, cfg.effective_pdot()),
                }
                for cfg in self.pulsars
            ],
            "observation": {
                "n_channels": self.obs.n_channels,
                "freq_low_mhz": self.obs.freq_low_mhz,
                "freq_high_mhz": self.obs.freq_high_mhz,
                "sample_time_s": self.obs.sample_time_s,
                "duration_s": self.obs.duration_s,
                "noise_std": self.obs.noise_std,
            },
        }
        return filterbank, freqs_mhz, t, metadata

    # ------------------------------------------------------------------ #
    # Utilitários de verificação / pós-processamento
    # ------------------------------------------------------------------ #
    @staticmethod
    def dedisperse(filterbank: np.ndarray, freqs_mhz: np.ndarray, t: np.ndarray, dm: float) -> np.ndarray:
        """
        De-dispersão simples: desloca cada canal de volta pelo atraso previsto
        para o DM fornecido, e soma todos os canais numa série temporal 1D.
        Útil como verificação (o pulso deve ficar nítido quando DM = DM real).
        """
        dt = t[1] - t[0]
        freq_ref = freqs_mhz.max()
        summed = np.zeros_like(t)
        for i, freq_ch in enumerate(freqs_mhz):
            delay = dispersion.dispersion_delay_relative(freq_ch, dm, freq_ref)
            shift_samples = int(round(delay / dt))
            summed += np.roll(filterbank[i, :], -shift_samples)
        return summed

    @staticmethod
    def fold_profile(timeseries: np.ndarray, dt: float, period_s: float, n_bins: int = 64) -> np.ndarray:
        """Dobra (folding) uma série temporal 1D no período fornecido, retornando o perfil médio."""
        n = len(timeseries)
        t = np.arange(n) * dt
        phase = np.mod(t / period_s, 1.0)
        bins = np.floor(phase * n_bins).astype(int)
        bins = np.clip(bins, 0, n_bins - 1)
        profile = np.zeros(n_bins)
        counts = np.zeros(n_bins)
        np.add.at(profile, bins, timeseries)
        np.add.at(counts, bins, 1)
        counts[counts == 0] = 1
        return profile / counts
