# Simulacao-de-Pulsares-Machine-Learning-Classico-vs.-Quantum-Machine-Learning

# pulsarsim — Simulador de sinais de pulsar (Fase 1)

Pacote Python modular que gera dados sintéticos de "filterbank" (intensidade
× tempo × frequência), semelhantes ao que um radiotelescópio real produziria,
com física de pulsares implementada com fórmulas padrão da literatura
(Lorimer & Kramer, *Handbook of Pulsar Astronomy*).

## Instalação
Sem dependências externas além de `numpy`, `scipy` e `matplotlib` (já
inclusos em qualquer ambiente científico padrão).

```bash
unzip pulsarsim.zip
cd <pasta onde extraiu>
python demo.py   # roda 4 cenários de teste e salva imagens de verificação
```

## Uso básico

```python
from pulsarsim import PulsarConfig, ObservationConfig, PulsarSimulator

obs = ObservationConfig(
    n_channels=128,
    freq_low_mhz=1200, freq_high_mhz=1500,
    sample_time_s=0.001,     # resolução temporal
    duration_s=3.0,          # duração da observação simulada
    noise_std=0.5,           # ruído branco
    seed=42,
)

pulsar = PulsarConfig(
    name="PSR_TESTE",
    period_s=0.8,            # período de rotação
    dm=60.0,                 # dispersão do meio interestelar
    duty_cycle=0.05,         # largura do pulso (fração do período)
    amplitude=10.0,          # intensidade do feixe
    distance_kpc=2.0,        # distância do observador
    b_field_gauss=1e12,      # campo magnético (opcional; calcula Ṗ automaticamente)
)

sim = PulsarSimulator(obs).add_pulsar(pulsar)
filterbank, freqs_mhz, t, metadata = sim.generate()
# filterbank.shape == (n_channels, n_samples)
```

## Parâmetros físicos suportados (todos os pedidos originalmente)

| Parâmetro | Onde configurar |
|---|---|
| Período de rotação | `PulsarConfig.period_s` |
| Campo magnético | `PulsarConfig.b_field_gauss` (ou `pdot` diretamente) |
| Intensidade do feixe | `PulsarConfig.amplitude` |
| Largura do pulso | `PulsarConfig.duty_cycle` |
| Distância do observador | `PulsarConfig.distance_kpc` (escala de fluxo 1/d²) |
| Ruído branco | `ObservationConfig.noise_std` |
| Interferências (RFI) | `ObservationConfig.rfi_config` (narrowband e broadband) |
| Dispersão do meio interestelar | `PulsarConfig.dm` |
| Decaimento da energia | calculado automaticamente via `spindown_luminosity` (modelo de freio magneto-dipolar) |
| Vários pulsares simultâneos | `sim.add_pulsar(...)` várias vezes |

Bônus incluídos (aumentam o realismo, não foram pedidos explicitamente mas
são padrão em dados reais): índice espectral (pulsares mais brilhantes em
frequências baixas), interpulso opcional, scattering (espalhamento
multi-caminho no ISM), e **glitches** (saltos de período com recuperação
exponencial — essencial para a Fase de detecção de anomalias).

## Validação física já realizada

- De-dispersão com o DM correto produz um perfil de pulso nitidamente mais
  concentrado (maior SNR) do que com um DM incorreto — testado
  automaticamente em `demo.py` (`scenario_1`).
- Campo magnético inferido de `B = 3.2e19 √(P·Ṗ)` e luminosidade de
  spin-down `Ė = 4π²IṖ/P³` retornam ordens de grandeza consistentes com
  pulsares reais catalogados (B ~10¹²G, Ė ~10³³ erg/s para um pulsar jovem
  típico).

## Estrutura do pacote

```
pulsarsim/
├── __init__.py       # expõe PulsarConfig, ObservationConfig, PulsarSimulator
├── pulse.py           # perfil de pulso (von Mises), suporte a interpulso
├── dispersion.py       # atraso de dispersão (DM) e scattering
├── spindown.py         # relações P–Ṗ–B–energia (freio magneto-dipolar)
├── glitch.py            # injeção de glitches (salto + recuperação exponencial)
├── noise.py              # ruído branco e RFI (narrowband/broadband)
├── multipulsar.py         # combinação de múltiplos pulsares
├── simulator.py            # PulsarSimulator: classe principal
└── plotting.py              # waterfall e perfil dobrado
```

## Próximo passo (Fase 2)
Envolver este simulador em um script de geração em lote (variando
parâmetros via *Latin Hypercube Sampling*) para produzir o dataset de
milhares de exemplos, já rotulado com o *ground truth* de cada anomalia
injetada.