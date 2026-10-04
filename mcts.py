import os
import sys
import copy
import math
import time
import random
import numpy as np
import torch
import torch.nn as nn

# Asegurar importación de módulos principales
directorio_raiz = os.path.dirname(os.path.abspath(__file__))
if directorio_raiz not in sys.path:
    sys.path.insert(0, directorio_raiz)

from juegos import Othello
from entrenamiento_supervisado import RedDeValorCNN

class NodoMCTS:
    """
    Representa un nodo dentro del árbol de búsqueda Monte Carlo (MCTS).
    Compartido por MCTS Clásico y MCTS Neuronal.
    """
    def __init__(self, estado, jugador_actual, movimiento_origen=None, padre=None, juego=None):
        """
        :param estado: Matriz bidimensional del tablero.
        :param jugador_actual: Jugador al que le corresponde mover en este estado.
        :param movimiento_origen: Movimiento (fila, col) o None que llevó a este nodo desde el padre.
        :param padre: NodoMCTS padre.
        :param juego: Instancia del juego para obtener los movimientos legales pendientes.
        """
        self.estado = estado
        self.jugador_actual = jugador_actual
        self.movimiento_origen = movimiento_origen
        self.padre = padre
        self.hijos = {} # Diccionario {movimiento: NodoMCTS}
        
        self.visitas = 0           # N
        self.valor_acumulado = 0.0 # W
        self.valor_promedio = 0.0  # Q = W / N
        
        if juego is not None and not juego.es_terminal(estado):
            movs = juego.obtener_movimientos_legales(estado, jugador_actual)
            self.movimientos_no_probados = list(movs)
        else:
            self.movimientos_no_probados = []

    def esta_totalmente_expandido(self):
        """Devuelve True si ya no quedan movimientos legales pendientes por explorar desde este nodo."""
        return len(self.movimientos_no_probados) == 0

    def ucb1(self, c_param=1.414):
        """
        Calcula el valor UCB1 (Upper Confidence Bound) para la fase de selección.
        
        :param c_param: Factor de exploración (por defecto sqrt(2) ≈ 1.414).
        :return: Valor UCB1 desde la perspectiva del jugador del nodo padre.
        """
        if self.visitas == 0:
            return float('inf')
        
        if self.padre is not None:
            # Si el turno cambió entre padre e hijo, el valor del hijo se invierte para el padre
            if self.jugador_actual == self.padre.jugador_actual:
                explotacion = self.valor_promedio
            else:
                explotacion = -self.valor_promedio
                
            exploracion = c_param * math.sqrt(math.log(self.padre.visitas) / self.visitas)
            return explotacion + exploracion
        else:
            return self.valor_promedio


class MCTSClasico:
    """
    Monte Carlo Tree Search (MCTS) clásico.
    Evalúa estados mediante simulaciones aleatorias (rollouts) completas
    hasta alcanzar un estado terminal.
    """
    def __init__(self, juego, c_param=1.414):
        self.juego = juego
        self.c_param = c_param

    def rollout(self, estado_inicial, jugador_inicial):
        """
        Ejecuta un rollout aleatorio completo desde estado_inicial hasta estado terminal.
        Retorna la recompensa (+1.0, -1.0, 0.0) respecto a jugador_inicial.
        """
        tablero = [fila[:] for fila in estado_inicial]
        turno = jugador_inicial
        rival = self.juego.obtener_rival(turno)
        
        while not self.juego.es_terminal(tablero):
            movs = self.juego.obtener_movimientos_legales(tablero, turno)
            if movs != [None]:
                f, c = random.choice(movs)
                tablero[f][c] = turno
                for vf, vc in self.juego._fichas_a_voltear(tablero, f, c, turno):
                    tablero[vf][vc] = turno
            turno, rival = rival, turno
            
        return float(self.juego.obtener_utilidad(tablero, jugador_inicial))

    def simular_busqueda(self, raiz):
        """
        Ejecuta una iteración de las 4 fases de MCTS Clásico:
        1. Selección: Desciende por el árbol eligiendo nodos según UCB1.
        2. Expansión: Genera un nuevo nodo hijo a partir de movimientos pendientes.
        3. Evaluación: Ejecuta un rollout aleatorio completo.
        4. Retropropagación: Actualiza visitas (N), valor acumulado (W) y promedio (Q).
        """
        nodo = raiz
        
        # 1. SELECCIÓN
        while not self.juego.es_terminal(nodo.estado) and nodo.esta_totalmente_expandido() and len(nodo.hijos) > 0:
            nodo = max(nodo.hijos.values(), key=lambda h: h.ucb1(self.c_param))
            
        # 2. EXPANSIÓN
        if not self.juego.es_terminal(nodo.estado) and len(nodo.movimientos_no_probados) > 0:
            movimiento = nodo.movimientos_no_probados.pop()
            nuevo_estado = self.juego.hacer_movimiento(nodo.estado, movimiento, nodo.jugador_actual)
            siguiente_jugador = self.juego.obtener_rival(nodo.jugador_actual)
            
            hijo = NodoMCTS(
                estado=nuevo_estado,
                jugador_actual=siguiente_jugador,
                movimiento_origen=movimiento,
                padre=nodo,
                juego=self.juego
            )
            nodo.hijos[movimiento] = hijo
            nodo = hijo
            
        # 3. EVALUACIÓN (Rollout aleatorio o utilidad terminal directa)
        if self.juego.es_terminal(nodo.estado):
            v = float(self.juego.obtener_utilidad(nodo.estado, nodo.jugador_actual))
        else:
            v = self.rollout(nodo.estado, nodo.jugador_actual)
            
        # 4. RETROPROPAGACIÓN
        jugador_evaluado = nodo.jugador_actual
        actual = nodo
        while actual is not None:
            actual.visitas += 1
            if actual.jugador_actual == jugador_evaluado:
                actual.valor_acumulado += v
            else:
                actual.valor_acumulado -= v
            actual.valor_promedio = actual.valor_acumulado / actual.visitas
            actual = actual.padre

    def mejor_movimiento(self, estado, jugador, num_simulaciones=50):
        """
        Ejecuta num_simulaciones iteraciones de MCTS y devuelve el movimiento más visitado.
        Interfaz homogénea: (movimiento, raiz)
        """
        raiz = NodoMCTS(estado, jugador, padre=None, juego=self.juego)
        if raiz.movimientos_no_probados == [None]:
            return None, raiz
            
        for _ in range(num_simulaciones):
            self.simular_busqueda(raiz)
            
        if not raiz.hijos:
            if raiz.movimientos_no_probados:
                return raiz.movimientos_no_probados[0], raiz
            return None, raiz
            
        mejor_hijo = max(raiz.hijos.values(), key=lambda h: h.visitas)
        return mejor_hijo.movimiento_origen, raiz


class MCTSNeuronal:
    """
    Monte Carlo Tree Search (MCTS) guiado por Red de Valor Convolucional (CNN).
    Sustituye los rollouts aleatorios por inferencia directa de la red neuronal.
    """
    def __init__(self, juego, modelo_o_ruta, dispositivo=None, c_param=1.414):
        self.juego = juego
        self.c_param = c_param
        
        if dispositivo is None:
            self.dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.dispositivo = torch.device(dispositivo)
            
        if isinstance(modelo_o_ruta, str):
            if not os.path.exists(modelo_o_ruta):
                raise FileNotFoundError(f"No se encontró el archivo de pesos del modelo en: {modelo_o_ruta}")
            self.modelo = RedDeValorCNN().to(self.dispositivo)
            state_dict = torch.load(modelo_o_ruta, map_location=self.dispositivo, weights_only=True)
            self.modelo.load_state_dict(state_dict)
        elif isinstance(modelo_o_ruta, nn.Module):
            self.modelo = modelo_o_ruta.to(self.dispositivo)
        else:
            raise TypeError("modelo_o_ruta debe ser una ruta de archivo (str) o una instancia de nn.Module.")
            
        self.modelo.eval()

    def estado_a_tensor(self, estado, jugador):
        """
        Transforma la matriz 2D del tablero en un tensor de PyTorch de dimensión (1, 2, N, N)
        tipo float32 con marco canónico:
          - Canal 0: Fichas del jugador activo (jugador).
          - Canal 1: Fichas del rival.
        """
        n = len(estado)
        rival = self.juego.obtener_rival(jugador)
        
        canal_0 = np.zeros((n, n), dtype=np.float32)
        canal_1 = np.zeros((n, n), dtype=np.float32)
        
        for r in range(n):
            for c in range(n):
                casilla = estado[r][c]
                if casilla == jugador:
                    canal_0[r, c] = 1.0
                elif casilla == rival:
                    canal_1[r, c] = 1.0
                    
        tensor_np = np.stack([canal_0, canal_1], axis=0)
        tensor_pt = torch.from_numpy(tensor_np).unsqueeze(0).to(self.dispositivo)
        return tensor_pt

    def simular_busqueda(self, raiz):
        """
        Ejecuta una iteración de las 4 fases de MCTS Neuronal:
        1. Selección: Desciende por el árbol eligiendo nodos según UCB1.
        2. Expansión: Añade un nuevo nodo hijo a partir de movimientos pendientes.
        3. Evaluación: Inferencia con la Red de Valor CNN (o utilidad exacta si es terminal).
        4. Retropropagación: Actualiza visitas y valores hacia la raíz.
        """
        nodo = raiz
        
        # 1. SELECCIÓN
        while not self.juego.es_terminal(nodo.estado) and nodo.esta_totalmente_expandido() and len(nodo.hijos) > 0:
            nodo = max(nodo.hijos.values(), key=lambda h: h.ucb1(self.c_param))
            
        # 2. EXPANSIÓN
        if not self.juego.es_terminal(nodo.estado) and len(nodo.movimientos_no_probados) > 0:
            movimiento = nodo.movimientos_no_probados.pop()
            nuevo_estado = self.juego.hacer_movimiento(nodo.estado, movimiento, nodo.jugador_actual)
            siguiente_jugador = self.juego.obtener_rival(nodo.jugador_actual)
            
            hijo = NodoMCTS(
                estado=nuevo_estado,
                jugador_actual=siguiente_jugador,
                movimiento_origen=movimiento,
                padre=nodo,
                juego=self.juego
            )
            nodo.hijos[movimiento] = hijo
            nodo = hijo
            
        # 3. EVALUACIÓN (Red de Valor CNN o utilidad terminal exacta)
        if self.juego.es_terminal(nodo.estado):
            v = float(self.juego.obtener_utilidad(nodo.estado, nodo.jugador_actual))
        else:
            tensor_entrada = self.estado_a_tensor(nodo.estado, nodo.jugador_actual)
            with torch.no_grad():
                v = self.modelo(tensor_entrada).item()
                v = max(-1.0, min(1.0, v))
                
        # 4. RETROPROPAGACIÓN
        jugador_evaluado = nodo.jugador_actual
        actual = nodo
        while actual is not None:
            actual.visitas += 1
            if actual.jugador_actual == jugador_evaluado:
                actual.valor_acumulado += v
            else:
                actual.valor_acumulado -= v
            actual.valor_promedio = actual.valor_acumulado / actual.visitas
            actual = actual.padre

    def mejor_movimiento(self, estado, jugador, num_simulaciones=50):
        """
        Ejecuta num_simulaciones iteraciones de MCTS y devuelve el movimiento más visitado.
        Interfaz homogénea: (movimiento, raiz)
        """
        raiz = NodoMCTS(estado, jugador, padre=None, juego=self.juego)
        if raiz.movimientos_no_probados == [None]:
            return None, raiz
            
        for _ in range(num_simulaciones):
            self.simular_busqueda(raiz)
            
        if not raiz.hijos:
            if raiz.movimientos_no_probados:
                return raiz.movimientos_no_probados[0], raiz
            return None, raiz
            
        mejor_hijo = max(raiz.hijos.values(), key=lambda h: h.visitas)
        return mejor_hijo.movimiento_origen, raiz
