"""
pulsarsim
=========

Simulador físico de sinais de pulsar semelhantes aos recebidos por um
radiotelescópio (dados de "filterbank": intensidade x tempo x frequência).

Módulos:
    pulse        -> perfil de pulso (von Mises), trem de pulsos
    dispersion   -> atraso de dispersão interestelar (DM) por canal de frequência
    spindown     -> relações físicas entre período, campo magnético e perda de energia
    glitch       -> injeção de glitches (saltos de período com recuperação exponencial)
    noise        -> ruído branco e interferências de rádio-frequência (RFI)
    multipulsar  -> combinação de múltiplos pulsares simultâneos
    simulator    -> classe de alto nível PulsarSimulator, que orquestra tudo
    plotting     -> visualizações (waterfall, perfil dobrado)
"""

from .simulator import PulsarConfig, ObservationConfig, PulsarSimulator

__all__ = ["PulsarConfig", "ObservationConfig", "PulsarSimulator"]
