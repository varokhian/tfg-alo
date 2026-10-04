import os
import math
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split

class OthelloDataset(Dataset):
    """
    Dataset de PyTorch para partidas de Othello / Reversi / juegos de tablero.
    
    Transforma cada fila del CSV en:
      - Tensor espacial de entrada (2, N, N) tipo torch.float32:
          * Canal 0: Matriz binaria con 1.0 en las fichas del jugador actual (Turno_Actual).
          * Canal 1: Matriz binaria con 1.0 en las fichas del rival.
      - Etiqueta objetivo (target): Ganador_Final (escalar float32 entre -1.0 y 1.0 respecto a MAX).
    """
    def __init__(self, ruta_csv, transform=None):
        """
        :param ruta_csv: Ruta al archivo CSV con los datos de las partidas.
        :param transform: Transformaciones opcionales a aplicar a los tensores de entrada.
        """
        if not os.path.exists(ruta_csv):
            raise FileNotFoundError(f"No se encontró el archivo CSV en: {ruta_csv}")
            
        self.df = pd.read_csv(ruta_csv)
        self.transform = transform
        
        columnas_requeridas = {'Tablero', 'Turno_Actual', 'Ganador_Final', 'Algoritmo'}
        if not columnas_requeridas.issubset(self.df.columns):
            raise ValueError(f"El CSV debe contener las columnas: {columnas_requeridas}. Encontradas: {set(self.df.columns)}")

    def __len__(self):
        return len(self.df)

    def _obtener_rival(self, jugador):
        """Obtiene la pieza rival correspondiente al jugador actual."""
        if jugador == 'N':
            return 'B'
        elif jugador == 'B':
            return 'N'
        elif jugador == 'X':
            return 'O'
        elif jugador == 'O':
            return 'X'
        return None

    def __getitem__(self, idx):
        fila = self.df.iloc[idx]
        tablero_str = str(fila['Tablero'])
        turno_actual = str(fila['Turno_Actual']).strip()
        algoritmo = str(fila['Algoritmo']).lower()
        ganador_final = float(fila['Ganador_Final'])
        
        longitud = len(tablero_str)
        n = int(math.isqrt(longitud))
        if n * n != longitud:
            raise ValueError(f"La longitud de la cadena del tablero ({longitud}) no es un cuadrado perfecto.")

        rival = self._obtener_rival(turno_actual)
        
        # Canal 0: Fichas del jugador activo (Turno_Actual)
        # Canal 1: Fichas del rival
        canal_jugador = np.zeros((n, n), dtype=np.float32)
        canal_rival = np.zeros((n, n), dtype=np.float32)
        
        for i, char in enumerate(tablero_str):
            r = i // n
            c = i % n
            if char == turno_actual:
                canal_jugador[r, c] = 1.0
            elif char == rival or (char != ' ' and char != turno_actual):
                canal_rival[r, c] = 1.0
                
        # Tensor espacial de dimensión (2, N, N) tipo torch.float32
        tensor_estado = torch.from_numpy(np.stack([canal_jugador, canal_rival], axis=0))
        
        # Marco canónico para el target:
        # Ganador_Final en el CSV está expresado respecto a la máquina MAX (pesos).
        # Si el turno corresponde a la máquina MIN (movilidad), se invierte el signo para que
        # 1.0 siempre signifique victoria para el jugador activo del Canal 0.
        if 'movilidad' in algoritmo:
            target_val = -ganador_final
        else:
            target_val = ganador_final

        target = torch.tensor(target_val, dtype=torch.float32)
        
        if self.transform:
            tensor_estado = self.transform(tensor_estado)
            
        return tensor_estado, target

def obtener_dataloaders(ruta_o_dataset, batch_size=64, train_ratio=0.8, shuffle=True, seed=42, num_workers=0):
    """
    Divide un OthelloDataset en subconjuntos de entrenamiento y validación y devuelve sus respectivos DataLoaders.
    
    :param ruta_o_dataset: Ruta al archivo CSV (str) o una instancia de OthelloDataset.
    :param batch_size: Tamaño del lote (batch size).
    :param train_ratio: Proporción para entrenamiento (por defecto 0.8 para 80% train / 20% val).
    :param shuffle: Si se deben mezclar los datos de entrenamiento.
    :param seed: Semilla para reproducibilidad.
    :param num_workers: Número de subprocesos para DataLoader.
    :return: (train_loader, val_loader)
    """
    if isinstance(ruta_o_dataset, str):
        dataset = OthelloDataset(ruta_o_dataset)
    else:
        dataset = ruta_o_dataset
        
    total_muestras = len(dataset)
    if total_muestras == 0:
        raise ValueError("El dataset está vacío. No se pueden generar DataLoaders.")
        
    train_size = int(total_muestras * train_ratio)
    val_size = total_muestras - train_size
    
    generator = torch.Generator().manual_seed(seed) if seed is not None else None
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    return train_loader, val_loader

if __name__ == "__main__":
    import glob
    archivos_csv = glob.glob("pruebas/**/*.csv", recursive=True) + glob.glob("*.csv")
    if archivos_csv:
        archivo_ejemplo = archivos_csv[0]
        print(f"Probando OthelloDataset con: {archivo_ejemplo}")
        ds = OthelloDataset(archivo_ejemplo)
        print(f"Total de registros: {len(ds)}")
        if len(ds) > 0:
            estado, target = ds[0]
            print(f"Forma del tensor de entrada: {estado.shape} (Canales, Filas, Columnas)")
            print(f"Tipo del tensor: {estado.dtype}")
            print(f"Etiqueta objetivo (Ganador_Final): {target.item()} ({target.dtype})")
            print("Canal 0 (Jugador actual):\n", estado[0])
            print("Canal 1 (Rival):\n", estado[1])
            
            train_loader, val_loader = obtener_dataloaders(ds, batch_size=32)
            print(f"\nDataLoaders creados:")
            print(f"  - Batches de entrenamiento: {len(train_loader)}")
            print(f"  - Batches de validación: {len(val_loader)}")
    else:
        print("No se encontraron archivos .csv en el directorio para la prueba directa.")
