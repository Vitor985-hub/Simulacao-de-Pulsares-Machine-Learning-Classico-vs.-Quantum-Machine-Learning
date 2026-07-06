"""
demo.py — Demonstração do pacote pulsarsim: 4 cenários de teste.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pulsarsim import PulsarConfig, ObservationConfig, PulsarSimulator
from pulsarsim.glitch import GlitchEvent
from pulsarsim.noise import RFIConfig, NarrowbandRFI, BroadbandImpulseRFI
from pulsarsim.plotting import plot_waterfall, plot_dedispersed_and_profile

OUT = "./demo_outputs"
import os
os.makedirs(OUT, exist_ok=True)


def scenario_1_single_pulsar_dispersion():
    print("== Cenário 1: pulsar único com dispersão ==")
    obs = ObservationConfig(
        n_channels=128, freq_low_mhz=1200, freq_high_mhz=1500,
        sample_time_s=0.0005, duration_s=3.0, noise_std=0.3, seed=42,
    )
    psr = PulsarConfig(name="PSR_A", period_s=0.5, dm=80.0, duty_cycle=0.04, amplitude=15.0)
    sim = PulsarSimulator(obs).add_pulsar(psr)
    fb, freqs, t, meta = sim.generate()
    print("Metadata:", meta["pulsars"][0])
    assert fb.shape == (128, len(t))
    assert not np.isnan(fb).any()

    fig, ax = plt.subplots(figsize=(9, 5))
    plot_waterfall(fb, freqs, t, title="Cenário 1: pulsar único (DM=80)", ax=ax)
    fig.colorbar(ax.images[0], ax=ax, label="Intensidade")
    fig.savefig(f"{OUT}/cenario1_waterfall.png", dpi=120)
    plt.close(fig)

    dedispersed = PulsarSimulator.dedisperse(fb, freqs, t, dm=80.0)
    folded = PulsarSimulator.fold_profile(dedispersed, obs.sample_time_s, psr.period_s, n_bins=64)
    fig2, _ = plot_dedispersed_and_profile(dedispersed, t, obs.sample_time_s, folded, title_prefix="Cenário 1: ")
    fig2.savefig(f"{OUT}/cenario1_dedispersed.png", dpi=120)
    plt.close(fig2)

    # Sanity check: de-dispersão com DM errado deve produzir perfil mais "achatado"
    dedispersed_wrong = PulsarSimulator.dedisperse(fb, freqs, t, dm=0.0)
    folded_wrong = PulsarSimulator.fold_profile(dedispersed_wrong, obs.sample_time_s, psr.period_s, n_bins=64)
    snr_correct = (folded.max() - folded.mean()) / folded.std()
    snr_wrong = (folded_wrong.max() - folded_wrong.mean()) / folded_wrong.std()
    print(f"SNR do perfil com DM correto: {snr_correct:.2f} | DM errado (DM=0): {snr_wrong:.2f}")
    assert snr_correct > snr_wrong, "De-dispersão correta deveria dar um pulso mais nítido!"
    print("OK: de-dispersão correta produz pulso mais nítido que DM incorreto.\n")


def scenario_2_noise_and_rfi():
    print("== Cenário 2: ruído branco + RFI (narrowband e broadband) ==")
    rfi = RFIConfig(
        narrowband=[NarrowbandRFI(channel_frac=0.7, period_s=0.02, amplitude=8.0, duty_cycle=0.5)],
        broadband=[BroadbandImpulseRFI(rate_hz=2.0, amplitude=20.0, width_s=0.01)],
    )
    obs = ObservationConfig(
        n_channels=96, freq_low_mhz=1200, freq_high_mhz=1500,
        sample_time_s=0.001, duration_s=4.0, noise_std=1.0, seed=7, rfi_config=rfi,
    )
    psr = PulsarConfig(name="PSR_B", period_s=0.8, dm=50.0, duty_cycle=0.05, amplitude=10.0)
    sim = PulsarSimulator(obs).add_pulsar(psr)
    fb, freqs, t, meta = sim.generate()
    assert fb.shape == (96, len(t))
    assert not np.isnan(fb).any()

    fig, ax = plt.subplots(figsize=(9, 5))
    plot_waterfall(fb, freqs, t, title="Cenário 2: pulsar + ruído + RFI", ax=ax)
    fig.colorbar(ax.images[0], ax=ax, label="Intensidade")
    fig.savefig(f"{OUT}/cenario2_waterfall_rfi.png", dpi=120)
    plt.close(fig)
    print("OK: cenário com RFI gerado sem erros.\n")


def scenario_3_multiple_pulsars():
    print("== Cenário 3: múltiplos pulsares simultâneos ==")
    obs = ObservationConfig(
        n_channels=100, freq_low_mhz=1200, freq_high_mhz=1500,
        sample_time_s=0.0008, duration_s=3.0, noise_std=0.4, seed=1,
    )
    psr1 = PulsarConfig(name="PSR_C1", period_s=0.3, dm=40.0, duty_cycle=0.04, amplitude=12.0, phase0=0.0)
    psr2 = PulsarConfig(name="PSR_C2", period_s=0.9, dm=120.0, duty_cycle=0.06, amplitude=9.0, phase0=0.3,
                         interpulse=True, interpulse_amplitude_frac=0.4)
    sim = PulsarSimulator(obs).add_pulsar(psr1).add_pulsar(psr2)
    fb, freqs, t, meta = sim.generate()
    assert fb.shape == (100, len(t))
    print("Pulsares simulados:", [p["name"] for p in meta["pulsars"]])

    fig, ax = plt.subplots(figsize=(9, 5))
    plot_waterfall(fb, freqs, t, title="Cenário 3: dois pulsares sobrepostos (DMs diferentes)", ax=ax)
    fig.colorbar(ax.images[0], ax=ax, label="Intensidade")
    fig.savefig(f"{OUT}/cenario3_multi_pulsar.png", dpi=120)
    plt.close(fig)
    print("OK: múltiplos pulsares combinados sem erros.\n")


def scenario_4_glitch_and_spindown():
    print("== Cenário 4: glitch + campo magnético/spin-down ==")
    obs = ObservationConfig(
        n_channels=64, freq_low_mhz=1300, freq_high_mhz=1500,
        sample_time_s=0.0005, duration_s=6.0, noise_std=0.3, seed=3,
    )
    glitch = GlitchEvent(time_s=3.0, delta_p_over_p=0.01, transient_fraction=0.6, recovery_timescale_s=0.8)
    psr = PulsarConfig(
        name="PSR_D", period_s=0.4, dm=60.0, duty_cycle=0.05, amplitude=14.0,
        b_field_gauss=1e12, glitches=[glitch],
    )
    sim = PulsarSimulator(obs).add_pulsar(psr)
    fb, freqs, t, meta = sim.generate()
    print("Metadata (inclui Pdot inferido de B e luminosidade de spin-down):")
    print(meta["pulsars"][0])
    assert fb.shape == (64, len(t))
    assert not np.isnan(fb).any()

    dedispersed = PulsarSimulator.dedisperse(fb, freqs, t, dm=60.0)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t, dedispersed, lw=0.7, color="#805ad5")
    ax.axvline(glitch.time_s, color="red", linestyle="--", label="instante do glitch")
    ax.set_xlabel("Tempo (s)")
    ax.set_ylabel("Intensidade somada (de-dispersada)")
    ax.set_title("Cenário 4: salto de período (glitch) em t=3s")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{OUT}/cenario4_glitch.png", dpi=120)
    plt.close(fig)
    print("OK: glitch injetado sem erros.\n")


if __name__ == "__main__":
    scenario_1_single_pulsar_dispersion()
    scenario_2_noise_and_rfi()
    scenario_3_multiple_pulsars()
    scenario_4_glitch_and_spindown()
    print("Todos os cenários rodaram com sucesso. Imagens salvas em", OUT)
