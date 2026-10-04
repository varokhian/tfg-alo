def minimax_universal(juego, estado, jugador_max, jugador_actual):
    """
    Minimax universal sin poda.
    Devuelve: (mejor_puntaje, mejor_movimiento, total_nodos_evaluados)
    """
    if juego.es_terminal(estado):
        return juego.obtener_utilidad(estado, jugador_max), None, 1

    es_turno_max = (jugador_actual == jugador_max)
    rival = juego.obtener_rival(jugador_actual)
    mejor_movimiento = None
    total_nodos = 1

    if es_turno_max:
        mejor_puntaje = -float('inf')
        for movimiento in juego.obtener_movimientos_legales(estado, jugador_actual):
            nuevo_estado = juego.hacer_movimiento(estado, movimiento, jugador_actual)
            puntaje, _, nodos_hijo = minimax_universal(juego, nuevo_estado, jugador_max, rival)
            total_nodos += nodos_hijo
            
            if puntaje > mejor_puntaje:
                mejor_puntaje = puntaje
                mejor_movimiento = movimiento
        return mejor_puntaje, mejor_movimiento, total_nodos
    else:
        peor_puntaje = float('inf')
        for movimiento in juego.obtener_movimientos_legales(estado, jugador_actual):
            nuevo_estado = juego.hacer_movimiento(estado, movimiento, jugador_actual)
            puntaje, _, nodos_hijo = minimax_universal(juego, nuevo_estado, jugador_max, rival)
            total_nodos += nodos_hijo
            
            if puntaje < peor_puntaje:
                peor_puntaje = puntaje
                mejor_movimiento = movimiento
        return peor_puntaje, mejor_movimiento, total_nodos

def minimax_alfa_beta(juego, estado, jugador_max, jugador_actual, profundidad, alfa=float('-inf'), beta=float('inf'), heuristica=None):
    """
    Minimax con Poda Alfa-Beta y soporte para heurísticas.
    Devuelve: (mejor_puntaje, mejor_movimiento, total_nodos_evaluados)
    """
    if juego.es_terminal(estado):
        return juego.obtener_utilidad(estado, jugador_max) * 10000, None, 1
        
    if profundidad == 0:
        return juego.evaluar_estado(estado, jugador_max, heuristica), None, 1

    es_turno_max = (jugador_actual == jugador_max)
    rival = juego.obtener_rival(jugador_actual)
    mejor_movimiento = None
    total_nodos = 1

    if es_turno_max:
        mejor_puntaje = float('-inf')
        for movimiento in juego.obtener_movimientos_legales(estado, jugador_actual):
            nuevo_estado = juego.hacer_movimiento(estado, movimiento, jugador_actual)
            puntaje, _, nodos_hijo = minimax_alfa_beta(juego, nuevo_estado, jugador_max, rival, profundidad - 1, alfa, beta, heuristica)
            total_nodos += nodos_hijo
            
            if puntaje > mejor_puntaje:
                mejor_puntaje = puntaje
                mejor_movimiento = movimiento
                
            alfa = max(alfa, mejor_puntaje)
            if beta <= alfa:
                break
                
        return mejor_puntaje, mejor_movimiento, total_nodos
    else:
        peor_puntaje = float('inf')
        for movimiento in juego.obtener_movimientos_legales(estado, jugador_actual):
            nuevo_estado = juego.hacer_movimiento(estado, movimiento, jugador_actual)
            puntaje, _, nodos_hijo = minimax_alfa_beta(juego, nuevo_estado, jugador_max, rival, profundidad - 1, alfa, beta, heuristica)
            total_nodos += nodos_hijo
            
            if puntaje < peor_puntaje:
                peor_puntaje = puntaje
                mejor_movimiento = movimiento
                
            beta = min(beta, peor_puntaje)
            if beta <= alfa:
                break
                
        return peor_puntaje, mejor_movimiento, total_nodos