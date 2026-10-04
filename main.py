# Importamos las clases creadas en el otro archivo
import os
import glob
import time
import random
import csv
from juegos import TresEnRaya, Othello
from minimax import minimax_universal, minimax_alfa_beta
from mcts import MCTSNeuronal

_agente_mcts_global = None

def obtener_agente_mcts(juego):
    """Instancia o reutiliza el agente MCTS Neuronal cargando el modelo de valor de forma segura."""
    global _agente_mcts_global
    if _agente_mcts_global is None:
        directorio_script = os.path.dirname(os.path.abspath(__file__))
        
        candidatos = [
            os.path.join(directorio_script, "Capítulo_6", "artefactos_memoria", "modelos", "mejor_red_valor_8x8.pth"),
            os.path.join(directorio_script, "Capítulo_6", "Fase_1_Entrenamiento_Supervisado", "modelos", "mejor_red_valor_8x8.pth"),
            os.path.join(directorio_script, "_Capitulo_6_uso_interno", "artefactos_memoria", "modelos", "mejor_red_valor_8x8.pth"),
            os.path.join(directorio_script, "_Capitulo_6_uso_interno", "Fase_1_Entrenamiento_Supervisado", "modelos", "mejor_red_valor_8x8.pth"),
            os.path.join(directorio_script, "pruebas", "prueba_max_pesos", "mejor_red_valor_8x8.pth"),
            os.path.join(directorio_script, "mejor_red_valor_8x8.pth"),
            "pruebas/prueba_max_pesos/mejor_red_valor_8x8.pth",
            "mejor_red_valor_8x8.pth"
        ]
        ruta_modelo = None
        for c in candidatos:
            if os.path.exists(c):
                ruta_modelo = c
                break
                
        if ruta_modelo is None:
            patron_pruebas = os.path.join(directorio_script, "pruebas", "**", "*.pth")
            patron_raiz = os.path.join(directorio_script, "*.pth")
            pths = glob.glob(patron_pruebas, recursive=True) + glob.glob(patron_raiz) + glob.glob("pruebas/**/*.pth", recursive=True) + glob.glob("*.pth")
            if pths:
                ruta_modelo = pths[0]
            else:
                raise FileNotFoundError(
                    f"No se encontró ningún modelo .pth para MCTS Neuronal. "
                    f"Buscado en '{directorio_script}'. "
                    f"Asegúrate de haber entrenado el modelo previamente (ejecutando entrenamiento_supervisado.py)."
                )
        _agente_mcts_global = MCTSNeuronal(juego=juego, modelo_o_ruta=ruta_modelo)
    else:
        _agente_mcts_global.juego = juego
    return _agente_mcts_global

def imprimir_tablero(estado):
    """Función que saca el tablero por pantalla"""
    print("\n")
    # Imprimir números de columnas para orientarse mejor 
    columnas = "   " + "   ".join([str(i+1) for i in range(len(estado[0]))])
    print(columnas)
    
    for i, fila in enumerate(estado):
        separador = "-" * (len(fila) * 4 - 1)
        # Añadir número de fila al principio
        print(f"{i+1} | " + " | ".join(fila))
        if i < len(estado) - 1:
            print("  " + separador)
    print("\n")

def pedir_movimiento(movimientos_validos):
    if movimientos_validos == [None]:
        print("No tienes movimientos válidos. Pulsa Enter para pasar el turno.")
        input()
        return None
        
    movimiento_elegido = None
    while movimiento_elegido not in movimientos_validos:
        try:
            entrada = input("Elige fila y columna separados por espacio (ej. '1 1' o '1 2'): ")
            fila, col = map(int, entrada.split())
            movimiento_elegido = (fila - 1, col - 1)

            if movimiento_elegido not in movimientos_validos:
                print("Movimiento inválido. Intenta de nuevo.")
        except ValueError:
            print("Formato incorrecto. Introduce dos números separados por espacio.")
            
    return movimiento_elegido

def seleccionar_algoritmo(nombre_maquina, juego):
    print(f"\n--- Algoritmo para {nombre_maquina} ---")
    print("1. Minimax (Sin poda)")
    print("2. Minimax con Poda Alfa-Beta")
    print("3. Aleatorio ")
    
    puede_mcts = isinstance(juego, Othello) and juego.n == 8
    if puede_mcts:
        print("4. MCTS Neuronal (CNN)")
        
    while True:
        max_op = '4' if puede_mcts else '3'
        opcion = input(f"Elige el algoritmo (1-{max_op}): ")
        if opcion == '1':
            return "universal"
        elif opcion == '2':
            if isinstance(juego, TresEnRaya):
                prof_def = 9
            elif isinstance(juego, Othello) and juego.n == 4:
                prof_def = 15
            else:
                prof_def = 4
                
            if isinstance(juego, Othello):
                print(f"\n--- Heurística para {nombre_maquina} (Alfa-Beta) ---")
                print("1. Pesos Posicionales ")
                print("2. Movilidad ")
                print("3. Híbrida (Pesos + Movilidad)")
                while True:
                    opcion_heur = input("Elige la heurística (1-3): ")
                    if opcion_heur == '1':
                        tipo_alfa = "alfa_beta_pesos"
                        break
                    elif opcion_heur == '2':
                        tipo_alfa = "alfa_beta_movilidad"
                        break
                    elif opcion_heur == '3':
                        tipo_alfa = "alfa_beta_hibrida"
                        break
                    print("Opción inválida.")
            else:
                tipo_alfa = "alfa_beta"
                
            entrada_prof = input(f"Profundidad máxima de búsqueda [Enter = {prof_def}]: ").strip()
            if entrada_prof == "":
                prof = prof_def
            else:
                try:
                    prof = int(entrada_prof)
                except ValueError:
                    print(f"Valor no numérico. Usando profundidad {prof_def} por defecto.")
                    prof = prof_def
                    
            return f"{tipo_alfa}|{prof}"
            
        elif opcion == '3':
            return "random"
        elif opcion == '4' and puede_mcts:
            entrada_sims = input("Número de simulaciones MCTS por turno [Enter = 50]: ").strip()
            if entrada_sims == "":
                num_sims = 50
            else:
                try:
                    num_sims = int(entrada_sims)
                except ValueError:
                    print("Valor no numérico. Usando 50 simulaciones por defecto.")
                    num_sims = 50
            return f"mcts_neuronal|{num_sims}"
            
        print("Opción inválida.")

def obtener_movimiento_maquina(juego, estado, jugador, algoritmo, silencioso=False, numero_turno=1, turnos_aleatorios_inicio=3, epsilon=0.10):
    if not silencioso:
        print("Pensando...")
        time.sleep(0.5)
    
    inicio_tiempo = time.perf_counter()
    nodos_evaluados = 0
    
    movimientos_validos = juego.obtener_movimientos_legales(estado, jugador)
    if movimientos_validos == [None]:
        tiempo_calculo = time.perf_counter() - inicio_tiempo
        return None, tiempo_calculo, 0

    # 1. Aperturas aleatorias durante los primeros N turnos
    if numero_turno <= turnos_aleatorios_inicio:
        movimiento_elegido = random.choice(movimientos_validos)
        nodos_evaluados = 1
    # 2. Política epsilon-greedy a partir de N+1
    elif epsilon > 0 and random.random() < epsilon:
        movimiento_elegido = random.choice(movimientos_validos)
        nodos_evaluados = 1
    # 3. Algoritmo seleccionado (explotación)
    else:
        if algoritmo == "random":
            movimiento_elegido = random.choice(movimientos_validos)
            nodos_evaluados = 1
        elif algoritmo == "universal":
            puntaje, movimiento_elegido, nodos_evaluados = minimax_universal(juego, estado, jugador, jugador)
        elif algoritmo.startswith("mcts_neuronal"):
            partes = algoritmo.split("|")
            num_sims = int(partes[1]) if len(partes) > 1 else 50
            agente_mcts = obtener_agente_mcts(juego)
            movimiento_elegido, _ = agente_mcts.mejor_movimiento(estado, jugador, num_simulaciones=num_sims)
            nodos_evaluados = num_sims
        elif algoritmo.startswith("mcts_clasico"):
            partes = algoritmo.split("|")
            num_sims = int(partes[1]) if len(partes) > 1 else 50
            from mcts import MCTSClasico
            agente_mcts = MCTSClasico(juego=juego)
            movimiento_elegido, _ = agente_mcts.mejor_movimiento(estado, jugador, num_simulaciones=num_sims)
            nodos_evaluados = num_sims
        elif algoritmo.startswith("alfa_beta"):
            if isinstance(juego, TresEnRaya):
                prof_def = 9
            elif isinstance(juego, Othello) and juego.n == 4:
                prof_def = 15
            else:
                prof_def = 4
                
            partes = algoritmo.split("|")
            tipo_alfa_beta = partes[0]
            if len(partes) > 1:
                try:
                    prof = int(partes[1])
                except ValueError:
                    prof = prof_def
            else:
                prof = prof_def
                
            heuristica = None
            if tipo_alfa_beta == "alfa_beta_movilidad":
                heuristica = "movilidad"
            elif tipo_alfa_beta == "alfa_beta_pesos":
                heuristica = "pesos"
            elif tipo_alfa_beta == "alfa_beta_hibrida":
                heuristica = "hibrida"
                
            puntaje, movimiento_elegido, nodos_evaluados = minimax_alfa_beta(juego, estado, jugador, jugador, profundidad=prof, heuristica=heuristica)
        else:
            raise ValueError(f"Algoritmo no reconocido: {algoritmo}")
            
    tiempo_calculo = time.perf_counter() - inicio_tiempo
    return movimiento_elegido, tiempo_calculo, nodos_evaluados

def jugar_humano_vs_humano(juego):
    estado, turno_actual = juego.estado_inicial()
    
    print(f"--- Juego: Humano vs Humano ---")

    while not juego.es_terminal(estado):
        imprimir_tablero(estado)
        print(f"\n Turno del Jugador {turno_actual}")

        movimientos_validos = juego.obtener_movimientos_legales(estado, turno_actual)
        movimiento_elegido = pedir_movimiento(movimientos_validos)

        estado = juego.hacer_movimiento(estado, movimiento_elegido, turno_actual)
        turno_actual = juego.obtener_rival(turno_actual)

    imprimir_tablero(estado)
    # Mostramos ganador
    jugador_inicial = juego.estado_inicial()[1]
    utilidad = juego.obtener_utilidad(estado, jugador_inicial)
    
    if utilidad == 1:
        print(f"Ha ganado el Jugador {jugador_inicial}!")
    elif utilidad == -1:
        print(f"Ha ganado el Jugador {juego.obtener_rival(jugador_inicial)}!")
    else:
        print("Ha sido un empate!")

def jugar_humano_vs_maquina(juego, algoritmo_maquina):
    print("\nLa máquina será el jugador MAX. ¿Qué piezas quieres llevar tú (Humano)?")
    if isinstance(juego, TresEnRaya):
        print("1. X (Empiezas tú)")
        print("2. O (Empieza la máquina)")
        opciones_validas = {'1': 'X', '2': 'O'}
    else:
        print("1. N (Negras - Empiezas tú)")
        print("2. B (Blancas - Empieza la máquina)")
        opciones_validas = {'1': 'N', '2': 'B'}
        
    while True:
        opcion = input("Elige (1-2): ")
        if opcion in opciones_validas:
            jugador_humano = opciones_validas[opcion]
            break
        print("Opción inválida.")

    jugador_maquina = juego.obtener_rival(jugador_humano)
    estado, turno_actual = juego.estado_inicial()
    numero_turno = 1
    
    print(f"--- Juego: Humano ({jugador_humano}) vs Máquina MAX ({jugador_maquina}) [{algoritmo_maquina}] ---")

    while not juego.es_terminal(estado):
        imprimir_tablero(estado)
        print(f"\n Turno de {turno_actual}")

        if turno_actual == jugador_humano:
            movimientos_validos = juego.obtener_movimientos_legales(estado, turno_actual)
            movimiento_elegido = pedir_movimiento(movimientos_validos)
        else:
            movimientos_validos = juego.obtener_movimientos_legales(estado, turno_actual)
            
            if movimientos_validos == [None]:
                print("La máquina tiene que pasar su turno.")
                time.sleep(1)
                movimiento_elegido = None
            else:
                movimiento_elegido, tiempo_calc, nodos_calc = obtener_movimiento_maquina(juego, estado, turno_actual, algoritmo_maquina, turnos_aleatorios_inicio=0, epsilon=0.0)
                print(f"La máquina elige la fila {movimiento_elegido[0]+1}, columna {movimiento_elegido[1]+1} (calculado en {tiempo_calc:.4f}s | {nodos_calc} nodos)")

        estado = juego.hacer_movimiento(estado, movimiento_elegido, turno_actual)
        turno_actual = juego.obtener_rival(turno_actual)

    imprimir_tablero(estado)
    utilidad = juego.obtener_utilidad(estado, jugador_maquina)
    
    if utilidad == 1:
        print(f"Ha ganado la máquina MAX ({jugador_maquina})!")
    elif utilidad == -1:
        print(f"Ha ganado el humano ({jugador_humano})!")
    else:
        print("Ha sido un empate!")

def jugar_maquina_vs_maquina(juego, algoritmo_max, algoritmo_min, pieza_max):
    estado, turno_actual = juego.estado_inicial()
    numero_turno = 1
    
    pieza_min = juego.obtener_rival(pieza_max)
    
    print(f"--- Juego: Máquina MAX ({pieza_max}) [{algoritmo_max}] vs Máquina MIN ({pieza_min}) [{algoritmo_min}] ---")

    while not juego.es_terminal(estado):
        imprimir_tablero(estado)
        print(f"\n Turno de la máquina {turno_actual}")
        
        movimientos_validos = juego.obtener_movimientos_legales(estado, turno_actual)
        
        if movimientos_validos == [None]:
            print("La máquina tiene que pasar su turno.")
            time.sleep(1)
            movimiento_elegido = None
        else:
            algoritmo_actual = algoritmo_max if turno_actual == pieza_max else algoritmo_min
            movimiento_elegido, tiempo_calc, nodos_calc = obtener_movimiento_maquina(juego, estado, turno_actual, algoritmo_actual, turnos_aleatorios_inicio=0, epsilon=0.0)
            print(f"La máquina {turno_actual} elige la fila {movimiento_elegido[0]+1}, columna {movimiento_elegido[1]+1} (calculado en {tiempo_calc:.4f}s | {nodos_calc} nodos)")

        estado = juego.hacer_movimiento(estado, movimiento_elegido, turno_actual)
        turno_actual = juego.obtener_rival(turno_actual)

    imprimir_tablero(estado)
    utilidad = juego.obtener_utilidad(estado, pieza_max)
    
    if utilidad == 1:
        print(f"Ha ganado la máquina MAX ({pieza_max})!")
    elif utilidad == -1:
        print(f"Ha ganado la máquina MIN ({pieza_min})!")
    else:
        print("Ha sido un empate!")

def serializar_tablero(estado):
    """Convierte el tablero bidimensional a una cadena plana."""
    return ''.join([''.join(fila) for fila in estado])

def generar_dataset_por_tiempo(juego, algoritmo_max, algoritmo_min, minutos, archivo_salida, patron_max, turnos_aleatorios_inicio=3, epsilon=0.10):
    tiempo_limite_segundos = minutos * 60
    inicio_global = time.time()
    
    directorio_script = os.path.dirname(os.path.abspath(__file__))
    ruta_completa = os.path.join(directorio_script, archivo_salida) if not os.path.isabs(archivo_salida) else archivo_salida
    os.makedirs(os.path.dirname(ruta_completa), exist_ok=True)
    
    pieza_inicial_juego = juego.estado_inicial()[1]
    pieza_rival_juego = juego.obtener_rival(pieza_inicial_juego)
    
    print(f"\nIniciando generación de datos durante {minutos} minutos...")
    print(f"Juego: {juego.__class__.__name__}")
    print(f"Máquina MAX: {algoritmo_max}")
    print(f"Máquina MIN: {algoritmo_min}")
    print(f"Turnos iniciales aleatorios (apertura): {turnos_aleatorios_inicio} | Probabilidad epsilon-greedy: {epsilon}")
    print(f"Archivo de destino: {ruta_completa}")
    print("Puedes pulsar Ctrl+C en cualquier momento para parar sin perder lo generado.")
    
    partidas_jugadas = 0
    
    try:
        with open(ruta_completa, mode='w', newline='', encoding='utf-8') as archivo:
            escritor = csv.writer(archivo)
            escritor.writerow(['Id_Partida', 'Numero_Turno', 'Turno_Actual', 'Tablero', 'Movimiento', 'Algoritmo', 'Tiempo_Segundos', 'Ganador_Final'])
            
            while time.time() - inicio_global < tiempo_limite_segundos:
                partidas_jugadas += 1
                
                if patron_max == '1':
                    pieza_max = pieza_inicial_juego
                elif patron_max == '2':
                    pieza_max = pieza_rival_juego
                else:
                    pieza_max = pieza_inicial_juego if partidas_jugadas % 2 != 0 else pieza_rival_juego
                    
                estado, turno_actual = juego.estado_inicial()
                pieza_min = juego.obtener_rival(pieza_max)
                
                historial_partida = []
                numero_turno = 1
                
                while not juego.es_terminal(estado):
                    movimientos_validos = juego.obtener_movimientos_legales(estado, turno_actual)
                    algoritmo_actual = algoritmo_max if turno_actual == pieza_max else algoritmo_min
                    
                    if movimientos_validos == [None]:
                        movimiento_elegido = None
                        tiempo = 0.0
                    else:
                        movimiento_elegido, tiempo, _ = obtener_movimiento_maquina(
                            juego, estado, turno_actual, algoritmo_actual, 
                            silencioso=True, numero_turno=numero_turno, 
                            turnos_aleatorios_inicio=turnos_aleatorios_inicio, epsilon=epsilon
                        )
                    
                    tablero_str = serializar_tablero(estado)
                    mov_str = f"({movimiento_elegido[0]},{movimiento_elegido[1]})" if movimiento_elegido else "None"
                    
                    historial_partida.append([
                        partidas_jugadas, numero_turno, turno_actual, tablero_str, mov_str, algoritmo_actual, round(tiempo, 4)
                    ])
                    
                    estado = juego.hacer_movimiento(estado, movimiento_elegido, turno_actual)
                    turno_actual = juego.obtener_rival(turno_actual)
                    numero_turno += 1
                    
                # Evaluar partida final respecto a MAX
                utilidad = juego.obtener_utilidad(estado, pieza_max)
                ganador_final = 0 # Empate
                if utilidad == 1:
                    ganador_final = 1 # Gana MAX
                elif utilidad == -1:
                    ganador_final = -1 # Gana MIN
                    
                # Guardar al csv
                for fila in historial_partida:
                    fila.append(ganador_final)
                    escritor.writerow(fila)
                    
                archivo.flush()
                
                if partidas_jugadas % 5 == 0:
                    tiempo_transcurrido = time.time() - inicio_global
                    tiempo_restante = max(0, tiempo_limite_segundos - tiempo_transcurrido)
                    print(f"Partidas completadas: {partidas_jugadas} | Tiempo restante: {tiempo_restante/60:.1f} mins")

    except KeyboardInterrupt:
        print("\nGeneración detenida manualmente")

    print(f"\nProceso finalizado. Se generaron {partidas_jugadas} partidas en '{archivo_salida}'.")

def crear_juego():
    print("\n=== SELECCIÓN DE JUEGO ===")
    print("1. Tres en Raya")
    print("2. Othello (Reversi) - Mini 4x4 ")
    print("3. Othello (Reversi) - Clásico 8x8")
    
    while True:
        opcion = input("Elige un juego (1-3): ")
        if opcion == '1':
            return TresEnRaya()
        elif opcion == '2':
            return Othello(tamano=4)
        elif opcion == '3':
            return Othello(tamano=8)
        print("Opción inválida.")

def menu_principal():
    while True:
        juego = crear_juego()
        pieza_inicial = juego.estado_inicial()[1]
        pieza_rival = juego.obtener_rival(pieza_inicial)
        
        print("\n=== SELECCIÓN DE MODO ===")
        print("1. Humano vs Humano")
        print("2. Humano vs Máquina")
        print("3. Máquina vs Máquina")
        print("4. Generar conjunto de datos (Límite de Tiempo)")
        print("5. Volver a elegir juego")
        print("6. Salir")
        
        while True:
            opcion = input("Elige un modo de juego (1-6): ")

            if opcion == '1':
                jugar_humano_vs_humano(juego)
                break
            elif opcion == '2':
                algoritmo = seleccionar_algoritmo("la Máquina (MAX)", juego)
                jugar_humano_vs_maquina(juego, algoritmo)
                break
            elif opcion == '3':
                algoritmo_max = seleccionar_algoritmo("Máquina MAX", juego)
                algoritmo_min = seleccionar_algoritmo("Máquina MIN", juego)
                print("\n¿Qué piezas controlará la Máquina MAX?")
                print(f"1. {pieza_inicial} (Empieza)")
                print(f"2. {pieza_rival} (Va segunda)")
                while True:
                    op = input("Elige (1-2): ")
                    if op == '1':
                        pieza_max = pieza_inicial
                        break
                    elif op == '2':
                        pieza_max = pieza_rival
                        break
                    print("Opción inválida.")
                jugar_maquina_vs_maquina(juego, algoritmo_max, algoritmo_min, pieza_max)
                break
            elif opcion == '4':
                print("\n--- Configuración de Generación de Datos ---")
                algoritmo_max = seleccionar_algoritmo("Máquina MAX", juego)
                algoritmo_min = seleccionar_algoritmo("Máquina MIN", juego)
                
                print("\nEn las partidas generadas, ¿qué rol tomará la máquina MAX?")
                print(f"1. Siempre jugará como {pieza_inicial}")
                print(f"2. Siempre jugará como {pieza_rival}")
                print("3. Alternando en cada partida")
                while True:
                    patron_max = input("Elige (1-3): ")
                    if patron_max in ['1', '2', '3']:
                        break
                    print("Opción inválida.")
                
                # Parámetros de aleatoriedad y exploración
                entrada_turnos = input("\nNúmero de turnos iniciales 100% aleatorios para aperturas [Enter = 3]: ").strip()
                if entrada_turnos == "":
                    turnos_aleatorios_inicio = 3
                else:
                    try:
                        turnos_aleatorios_inicio = int(entrada_turnos)
                    except ValueError:
                        print("Valor no numérico. Se usará el valor por defecto: 3.")
                        turnos_aleatorios_inicio = 3

                entrada_eps = input("Porcentaje/probabilidad epsilon de movimientos aleatorios (0.0 a 1.0) [Enter = 0.10]: ").strip()
                if entrada_eps == "":
                    epsilon = 0.10
                else:
                    try:
                        epsilon = float(entrada_eps)
                    except ValueError:
                        print("Valor no numérico. Se usará el valor por defecto: 0.10.")
                        epsilon = 0.10
                
                try:
                    entrada_mins = input("\n¿Cuántos minutos quieres estar generando datos? [Enter = 1.0]: ").strip()
                    mins = float(entrada_mins) if entrada_mins != "" else 1.0
                except ValueError:
                    print("Por favor, introduce un número válido. Usando 1 minuto por defecto.")
                    mins = 1.0

                # Carpeta y archivo de destino
                entrada_carpeta = input("\nCarpeta de destino para este experimento [Enter = pruebas/experimento_1]: ").strip()
                if entrada_carpeta == "":
                    carpeta_salida = os.path.join("pruebas", "experimento_1")
                else:
                    carpeta_salida = entrada_carpeta if (entrada_carpeta.startswith("pruebas/") or entrada_carpeta.startswith("pruebas\\")) else os.path.join("pruebas", entrada_carpeta)
                
                archivo = input("Nombre del archivo de salida [Enter = dataset.csv]: ").strip()
                if archivo == "":
                    archivo = "dataset.csv"
                if not archivo.endswith(".csv"):
                    archivo += ".csv"

                ruta_dataset = os.path.join(carpeta_salida, archivo)

                generar_dataset_por_tiempo(
                    juego, algoritmo_max, algoritmo_min, mins, ruta_dataset, patron_max,
                    turnos_aleatorios_inicio=turnos_aleatorios_inicio, epsilon=epsilon
                )
                break
            elif opcion == '5':
                break
            elif opcion == '6':
                print("Hasta pronto!")
                return
            else:
                print("Opción no válida. Inténtalo de nuevo.")

if __name__ == "__main__":
    menu_principal()