from abc import ABC, abstractmethod
import copy

class Juego(ABC):
    @abstractmethod
    def estado_inicial(self):
        pass

    @abstractmethod
    def obtener_movimientos_legales(self, estado, jugador):
        pass

    @abstractmethod
    def hacer_movimiento(self, estado, movimiento, jugador):
        pass

    @abstractmethod
    def es_terminal(self, estado):
        pass

    @abstractmethod
    def obtener_utilidad(self, estado, jugador):
        pass

    @abstractmethod
    def obtener_rival(self, jugador):
        pass

    @abstractmethod
    def evaluar_estado(self, estado, jugador, heuristica=None):
        pass

class TresEnRaya(Juego):
    def estado_inicial(self):
        tablero = [[' ']*3 for _ in range(3)]
        return tablero, 'X'
        
    def obtener_movimientos_legales(self, estado, jugador):
        casillas_vacias = []
        for f in range(3):
            for c in range(3):
                if estado[f][c] == ' ':
                    casillas_vacias.append((f, c))
        return casillas_vacias

    def hacer_movimiento(self, estado, movimiento, jugador):
        nuevo_estado = copy.deepcopy(estado)
        fila, col = movimiento
        nuevo_estado[fila][col] = jugador
        return nuevo_estado

    def es_terminal(self, estado):
        return self.obtener_utilidad(estado, 'X') != 0 or len(self.obtener_movimientos_legales(estado, 'X')) == 0

    def obtener_rival(self, jugador):
        return 'O' if jugador == 'X' else 'X'

    def obtener_utilidad(self, estado, jugador):
        rival = self.obtener_rival(jugador)
        if self._verificar_ganador(estado, jugador):
            return 1
        elif self._verificar_ganador(estado, rival):
            return -1
        else:
            return 0

    def evaluar_estado(self, estado, jugador, heuristica=None):
        return self.obtener_utilidad(estado, jugador)
            
    def _verificar_ganador(self, estado, jugador):
        for fila in estado:
            if fila[0] == fila[1] == fila[2] == jugador:
                return True
        for col in range(3):
            if estado[0][col] == estado[1][col] == estado[2][col] == jugador:
                return True
        if estado[0][0] == estado[1][1] == estado[2][2] == jugador:
            return True
        if estado[0][2] == estado[1][1] == estado[2][0] == jugador:
            return True
        return False

class Othello(Juego):
    def __init__(self, tamano=8):
        self.n = tamano
        self.pesos = [
            [ 100, -20,  10,   5,   5,  10, -20, 100],
            [ -20, -50,  -2,  -2,  -2,  -2, -50, -20],
            [  10,  -2,  -1,  -1,  -1,  -1,  -2,  10],
            [   5,  -2,  -1,  -1,  -1,  -1,  -2,   5],
            [   5,  -2,  -1,  -1,  -1,  -1,  -2,   5],
            [  10,  -2,  -1,  -1,  -1,  -1,  -2,  10],
            [ -20, -50,  -2,  -2,  -2,  -2, -50, -20],
            [ 100, -20,  10,   5,   5,  10, -20, 100]
        ]
        
    def estado_inicial(self):
        tablero = [[' ']*self.n for _ in range(self.n)]
        mitad = self.n // 2
        tablero[mitad-1][mitad-1] = 'B'
        tablero[mitad][mitad] = 'B'
        tablero[mitad-1][mitad] = 'N'
        tablero[mitad][mitad-1] = 'N'
        return tablero, 'N'
        
    def obtener_rival(self, jugador):
        return 'B' if jugador == 'N' else 'N'
        
    def _en_tablero(self, f, c):
        return 0 <= f < self.n and 0 <= c < self.n

    def _fichas_a_voltear(self, estado, f, c, jugador):
        rival = self.obtener_rival(jugador)
        fichas_voltear = []
        direcciones = [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1)]
        for df, dc in direcciones:
            f_act, c_act = f + df, c + dc
            camino = []
            while self._en_tablero(f_act, c_act) and estado[f_act][c_act] == rival:
                camino.append((f_act, c_act))
                f_act += df
                c_act += dc
            if self._en_tablero(f_act, c_act) and estado[f_act][c_act] == jugador:
                fichas_voltear.extend(camino)
        return fichas_voltear

    def obtener_movimientos_legales(self, estado, jugador):
        movimientos = []
        for f in range(self.n):
            for c in range(self.n):
                if estado[f][c] == ' ':
                    if self._fichas_a_voltear(estado, f, c, jugador):
                        movimientos.append((f, c))
        if not movimientos:
            return [None]
        return movimientos

    def hacer_movimiento(self, estado, movimiento, jugador):
        nuevo_estado = copy.deepcopy(estado)
        if movimiento is None:
            return nuevo_estado 
            
        f, c = movimiento
        fichas = self._fichas_a_voltear(estado, f, c, jugador)
        nuevo_estado[f][c] = jugador
        for vf, vc in fichas:
            nuevo_estado[vf][vc] = jugador
        return nuevo_estado

    def es_terminal(self, estado):
        movs_n = self.obtener_movimientos_legales(estado, 'N')
        movs_b = self.obtener_movimientos_legales(estado, 'B')
        return movs_n == [None] and movs_b == [None]

    def obtener_utilidad(self, estado, jugador):
        puntos_jugador = 0
        puntos_rival = 0
        rival = self.obtener_rival(jugador)
        for fila in estado:
            for casilla in fila:
                if casilla == jugador:
                    puntos_jugador += 1
                elif casilla == rival:
                    puntos_rival += 1
                    
        if puntos_jugador > puntos_rival:
            return 1
        elif puntos_jugador < puntos_rival:
            return -1
        else:
            return 0

    def evaluar_estado(self, estado, jugador, heuristica=None):
        rival = self.obtener_rival(jugador)
        valor_pesos = 0
        if heuristica != "movilidad":
            for f in range(self.n):
                for c in range(self.n):
                    if estado[f][c] == jugador:
                        valor_pesos += self.pesos[f][c] if self.n == 8 else 1
                    elif estado[f][c] == rival:
                        valor_pesos -= self.pesos[f][c] if self.n == 8 else 1

        if heuristica == "movilidad":
            movs_jug_legales = self.obtener_movimientos_legales(estado, jugador)
            movs_riv_legales = self.obtener_movimientos_legales(estado, rival)
            movs_jugador = 0 if movs_jug_legales == [None] else len(movs_jug_legales)
            movs_rival = 0 if movs_riv_legales == [None] else len(movs_riv_legales)
            return (movs_jugador - movs_rival) * 10
        elif heuristica == "hibrida":
            movs_jug_legales = self.obtener_movimientos_legales(estado, jugador)
            movs_riv_legales = self.obtener_movimientos_legales(estado, rival)
            movs_jugador = 0 if movs_jug_legales == [None] else len(movs_jug_legales)
            movs_rival = 0 if movs_riv_legales == [None] else len(movs_riv_legales)
            return valor_pesos + (movs_jugador - movs_rival) * 10
        else:
            return valor_pesos