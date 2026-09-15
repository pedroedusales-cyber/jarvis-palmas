"""
Detector de palmas - Módulo principal
Detecta padrões de palmas através de análise de áudio
"""

import numpy as np
import sounddevice as sd
import logging
from typing import List, Tuple, Optional
from collections import deque
from dataclasses import dataclass
from enum import Enum


class ClapPattern(Enum):
    """Enumeração de padrões de palmas."""
    SINGLE = 1      # 👏
    DOUBLE = 2      # 👏👏
    TRIPLE = 3      # 👏👏👏
    QUAD = 4        # 👏👏👏👏
    SOS = 5          # 👏👏👏👏👏
    CUSTOM = 0


@dataclass
class ClapDetection:
    """Resultado de detecção de palma."""
    pattern: ClapPattern
    confidence: float
    intensity: float
    timestamp: float
    frequency_components: Optional[List[float]] = None
    
    def __str__(self) -> str:
        return f"Padrão: {self.pattern.name}, Confiança: {self.confidence:.2%}, Intensidade: {self.intensity:.2%}"


class PalmsDetector:
    """Detecta padrões de palmas em áudio em tempo real."""
    
    def __init__(
        self,
        sample_rate: int = 44100,
        chunk_size: int = 2048,
        sensitivity: float = 0.7,
        clap_duration: float = 0.2,
        silence_threshold: float = 0.3
    ):
        """
        Inicializar detector de palmas.
        
        Args:
            sample_rate: Taxa de amostragem em Hz
            chunk_size: Tamanho do buffer
            sensitivity: Sensibilidade de detecção (0-1)
            clap_duration: Duração esperada de uma palma em segundos
            silence_threshold: Limite de silêncio entre palmas
        """
        self.logger = logging.getLogger('PalmsDetector')
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.sensitivity = sensitivity
        self.clap_duration = clap_duration
        self.silence_threshold = silence_threshold
        
        # Buffer de áudio
        self.audio_buffer = deque(maxlen=sample_rate * 2)  # 2 segundos
        
        # Parâmetros de detecção
        self.clap_threshold = 1.0 - sensitivity  # Inverter sensibilidade
        self.min_clap_samples = int(sample_rate * clap_duration)
        self.max_clap_samples = int(sample_rate * clap_duration * 2)
        
        # Histórico de palmas
        self.clap_history: deque = deque(maxlen=10)
        
        self.logger.info(f"Detector inicializado - Taxa: {sample_rate}Hz, Sensibilidade: {sensitivity}")
    
    def detect_clap(self, audio_chunk: np.ndarray) -> Optional[ClapDetection]:
        """
        Detectar uma palma individual em um chunk de áudio.
        
        Args:
            audio_chunk: Array de áudio
            
        Returns:
            ClapDetection se uma palma foi detectada, None caso contrário
        """
        # Normalizar áudio
        audio_chunk = self._normalize_audio(audio_chunk)
        
        # Calcular RMS (Root Mean Square)
        rms = np.sqrt(np.mean(audio_chunk ** 2))
        
        # Calcular energia no domínio da frequência
        fft = np.fft.fft(audio_chunk)
        frequencies = np.fft.fftfreq(len(audio_chunk), 1 / self.sample_rate)
        magnitude = np.abs(fft)
        
        # Palmas típicas estão entre 2kHz e 8kHz
        clap_freq_mask = (frequencies > 2000) & (frequencies < 8000)
        clap_energy = np.sum(magnitude[clap_freq_mask])
        total_energy = np.sum(magnitude)
        
        clap_ratio = clap_energy / (total_energy + 1e-6)
        
        # Verificar se é uma palma
        is_clap = (rms > self.clap_threshold) and (clap_ratio > 0.3)
        
        if is_clap:
            confidence = min(1.0, rms * clap_ratio)
            intensity = rms
            
            detection = ClapDetection(
                pattern=ClapPattern.SINGLE,
                confidence=confidence,
                intensity=intensity,
                timestamp=0.0,
                frequency_components=[clap_energy, clap_ratio]
            )
            
            self.logger.debug(f"Palma detectada: {detection}")
            return detection
        
        return None
    
    def detect_pattern(self, audio_data: np.ndarray, timeout: float = 2.0) -> Optional[ClapPattern]:
        """
        Detectar padrão de palmas múltiplas.
        
        Args:
            audio_data: Array de áudio
            timeout: Tempo máximo entre palmas em segundos
            
        Returns:
            ClapPattern detectado
        """
        claps = []
        chunk_duration = self.chunk_size / self.sample_rate
        chunks_per_second = int(1.0 / chunk_duration)
        
        # Processar áudio em chunks
        for i in range(0, len(audio_data), self.chunk_size):
            chunk = audio_data[i:i + self.chunk_size]
            
            if len(chunk) < self.chunk_size:
                chunk = np.pad(chunk, (0, self.chunk_size - len(chunk)))
            
            detection = self.detect_clap(chunk)
            if detection and detection.confidence > self.sensitivity:
                claps.append(detection)
        
        # Agrupar palmas próximas
        if claps:
            grouped_claps = self._group_claps(claps, timeout)
            pattern = self._classify_pattern(grouped_claps)
            return pattern
        
        return None
    
    def _normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        """Normalizar áudio para [-1, 1]."""
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            return audio / max_val
        return audio
    
    def _group_claps(self, claps: List[ClapDetection], timeout: float) -> List[List[ClapDetection]]:
        """Agrupar palmas em sequências."""
        if not claps:
            return []
        
        groups = [[claps[0]]]
        
        for clap in claps[1:]:
            # Se está dentro do timeout, adicionar ao grupo atual
            if groups[-1] and (clap.timestamp - groups[-1][-1].timestamp) < timeout:
                groups[-1].append(clap)
            else:
                # Novo grupo
                groups.append([clap])
        
        return groups
    
    def _classify_pattern(self, groups: List[List[ClapDetection]]) -> Optional[ClapPattern]:
        """Classificar padrão de palmas."""
        if not groups:
            return None
        
        # Contar palmas no primeiro grupo (mais recente)
        num_claps = len(groups[-1])
        
        try:
            return ClapPattern(num_claps)
        except ValueError:
            return ClapPattern.CUSTOM
    
    def listen_for_claps(self, duration: float = 5.0, device: Optional[int] = None) -> List[ClapDetection]:
        """
        Escutar por palmas durante um período.
        
        Args:
            duration: Duração em segundos
            device: Índice do dispositivo de áudio
            
        Returns:
            Lista de detecções de palmas
        """
        self.logger.info(f"Escutando por palmas por {duration} segundos...")
        
        detections = []
        
        try:
            # Registrar áudio
            audio_data = sd.rec(
                int(duration * self.sample_rate),
                samplerate=self.sample_rate,
                channels=1,
                device=device
            )
            sd.wait()
            
            audio_data = audio_data.flatten()
            
            # Detectar palmas
            pattern = self.detect_pattern(audio_data, timeout=duration)
            
            if pattern:
                detection = ClapDetection(
                    pattern=pattern,
                    confidence=0.85,
                    intensity=0.75,
                    timestamp=0.0
                )
                detections.append(detection)
                self.logger.info(f"Padrão detectado: {pattern.name}")
            else:
                self.logger.info("Nenhuma palma detectada")
        
        except Exception as e:
            self.logger.error(f"Erro ao detectar palmas: {e}")
        
        return detections
    
    def test_audio_device(self, device: Optional[int] = None) -> bool:
        """
        Testar dispositivo de áudio.
        
        Args:
            device: Índice do dispositivo
            
        Returns:
            True se o dispositivo funciona
        """
        self.logger.info("Testando dispositivo de áudio...")
        
        try:
            # Registrar 1 segundo de áudio
            print("Falando/Batendo palmas agora...")
            audio = sd.rec(self.sample_rate, samplerate=self.sample_rate, channels=1, device=device)
            sd.wait()
            
            # Verificar se há áudio
            rms = np.sqrt(np.mean(audio ** 2))
            
            if rms > 0.01:
                self.logger.info(f"✓ Dispositivo funcionando. RMS: {rms:.4f}")
                return True
            else:
                self.logger.warning("✗ Nenhum áudio detectado. Verifique o microfone.")
                return False
        
        except Exception as e:
            self.logger.error(f"✗ Erro ao testar dispositivo: {e}")
            return False
    
    def list_devices(self) -> None:
        """Listar dispositivos de áudio disponíveis."""
        print("\nDispositivos de áudio disponíveis:")
        print(sd.query_devices())
