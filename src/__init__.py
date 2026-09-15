"""
JARVIS - Sistema de Resposta por Palmas
Módulo principal do projeto
"""

__version__ = "1.0.0"
__author__ = "Pedro E. de Sales"
__email__ = "pedroedusales@gmail.com"

from .jarvis import JARVIS
from .config import Config

__all__ = ['JARVIS', 'Config']
