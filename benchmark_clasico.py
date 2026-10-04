import os
import sys
import time
import random
import csv
import numpy as np
import matplotlib.pyplot as plt

from juegos import Othello, TresEnRaya
from main import obtener_movimiento_maquina, obtener_agente_mcts

class AgenteBenchmark:
    """Representa un agente configurado para el torneo de benchmark."""
    def __init__(self, nombre, id_algoritmo):
        self.nombre = nombre
        self.id_algoritmo = id_algoritmo
        
        # Métricas acumuladas
        self.partidas_jugadas = 0
        self.victorias = 0
        self.empates = 0
        self.derrotas = 0
        self.fichas_totales = 0
        self.tiempo_total_calculo = 0.0
        self.nodos_totales_evaluados = 0
        self.turnos_totales_jugados = 0

    @property
    def porcentaje_victorias(self):
        if self.partidas_jugadas == 0:
            return 0.0
        return (self.victorias / self.partidas_jugadas) * 100.0

    @property
    def porcentaje_puntos_efectivos(self):
        """Puntos estilo ajedrez: Victoria = 1 pt, Empate = 0.5 pt"""
        if self.partidas_jugadas == 0:
            return 0.0
        return ((self.victorias + 0.5 * self.empates) / self.partidas_jugadas) * 100.0

    @property
    def tiempo_medio_turno(self):
        if self.turnos_totales_jugados == 0:
            return 0.0
        return self.tiempo_total_calculo / self.turnos_totales_jugados

    @property
    def nodos_medio_turno(self):
        if self.turnos_totales_jugados == 0:
            return 0.0
        return self.nodos_totales_evaluados / self.turnos_totales_jugados

    @property
    def media_fichas_finales(self):
        if self.partidas_jugadas == 0:
            return 0.0
        return self.fichas_totales / self.partidas_jugadas


def jugar_partida_benchmark(juego, agente_n, agente_b, turnos_aleatorios_inicio=0, epsilon=0.0):
    """
    Ejecuta una partida completa entre dos agentes en Othello.
    Registra tiempos y nodos evaluados para cada movimiento.
    Retorna: (ganador, fichas_n, fichas_b, tiempo_total_partida)
      ganador: 'N', 'B' o 'E' (empate)
    """
    estado, turno_actual = juego.estado_inicial()
    numero_turno = 1
    inicio_partida = time.perf_counter()
    
    while not juego.es_terminal(estado):
        movimientos_validos = juego.obtener_movimientos_legales(estado, turno_actual)
        
        if movimientos_validos == [None]:
            movimiento_elegido = None
        else:
            agente_actual = agente_n if turno_actual == 'N' else agente_b
            
            movimiento_elegido, t_mov, nodos_mov = obtener_movimiento_maquina(
                juego,
                estado,
                turno_actual,
                agente_actual.id_algoritmo,
                silencioso=True,
                numero_turno=numero_turno,
                turnos_aleatorios_inicio=turnos_aleatorios_inicio,
                epsilon=epsilon
            )
            
            # Registrar métricas del turno
            agente_actual.tiempo_total_calculo += t_mov
            agente_actual.nodos_totales_evaluados += nodos_mov
            agente_actual.turnos_totales_jugados += 1
            
        estado = juego.hacer_movimiento(estado, movimiento_elegido, turno_actual)
        turno_actual = juego.obtener_rival(turno_actual)
        numero_turno += 1
        
    tiempo_total_partida = time.perf_counter() - inicio_partida
    
    puntos_n = sum(fila.count('N') for fila in estado)
    puntos_b = sum(fila.count('B') for fila in estado)
    
    if puntos_n > puntos_b:
        ganador = 'N'
    elif puntos_b > puntos_n:
        ganador = 'B'
    else:
        ganador = 'E'
        
    return ganador, puntos_n, puntos_b, tiempo_total_partida


def ejecutar_torneo(
    agentes,
    num_partidas_por_cruce=10,
    turnos_aleatorios_inicio=0,
    epsilon=0.0,
    carpeta_salida="pruebas/benchmark_clasico"
):
    """
    Ejecuta un torneo todos contra todos cruzando cada pareja de agentes N partidas
    (50% jugando como Negras y 50% como Blancas).
    """
    directorio_script = os.path.dirname(os.path.abspath(__file__))
    ruta_carpeta = os.path.join(directorio_script, carpeta_salida) if not os.path.isabs(carpeta_salida) else carpeta_salida
    os.makedirs(ruta_carpeta, exist_ok=True)
    
    juego = Othello(tamano=8)
    num_agentes = len(agentes)
    
    matriz_victorias_a_vs_b = np.full((num_agentes, num_agentes), np.nan)
    total_cruces = (num_agentes * (num_agentes - 1)) // 2
    cruce_actual = 0
    tiempo_inicio_torneo = time.perf_counter()

    print("\n" + "=" * 75)
    print("=== INICIANDO BENCHMARK DE ALGORITMOS CLÁSICOS EN OTHELLO 8x8 ===")
    print("=" * 75)
    print(f"Número de agentes participantes: {num_agentes}")
    print(f"Partidas por cruce: {num_partidas_por_cruce} (alternando colores)")
    print(f"Turnos aleatorios de apertura: {turnos_aleatorios_inicio} | Epsilon-greedy: {epsilon}")
    print(f"Carpeta de exportación: {ruta_carpeta}")
    print("-" * 75)
    
    for i in range(num_agentes):
        for j in range(i + 1, num_agentes):
            cruce_actual += 1
            agente_1 = agentes[i]
            agente_2 = agentes[j]
            
            print(f"[{cruce_actual:02d}/{total_cruces:02d}] Enfrentamiento: {agente_1.nombre} vs {agente_2.nombre} ({num_partidas_por_cruce} partidas)...", end="", flush=True)
            
            victorias_1 = 0
            victorias_2 = 0
            empates = 0
            
            mitad = num_partidas_por_cruce // 2
            resto = num_partidas_por_cruce - mitad
            
            t0_cruce = time.perf_counter()
            
            # Bloque 1: Agente 1 (Negras) vs Agente 2 (Blancas)
            for _ in range(mitad):
                ganador, pn, pb, _ = jugar_partida_benchmark(juego, agente_1, agente_2, turnos_aleatorios_inicio, epsilon)
                agente_1.partidas_jugadas += 1
                agente_2.partidas_jugadas += 1
                agente_1.fichas_totales += pn
                agente_2.fichas_totales += pb
                
                if ganador == 'N':
                    agente_1.victorias += 1
                    agente_2.derrotas += 1
                    victorias_1 += 1
                elif ganador == 'B':
                    agente_2.victorias += 1
                    agente_1.derrotas += 1
                    victorias_2 += 1
                else:
                    agente_1.empates += 1
                    agente_2.empates += 1
                    empates += 1

            # Bloque 2: Agente 2 (Negras) vs Agente 1 (Blancas)
            for _ in range(resto):
                ganador, pn, pb, _ = jugar_partida_benchmark(juego, agente_2, agente_1, turnos_aleatorios_inicio, epsilon)
                agente_1.partidas_jugadas += 1
                agente_2.partidas_jugadas += 1
                agente_1.fichas_totales += pb
                agente_2.fichas_totales += pn
                
                if ganador == 'N':
                    agente_2.victorias += 1
                    agente_1.derrotas += 1
                    victorias_2 += 1
                elif ganador == 'B':
                    agente_1.victorias += 1
                    agente_2.derrotas += 1
                    victorias_1 += 1
                else:
                    agente_1.empates += 1
                    agente_2.empates += 1
                    empates += 1
                    
            t_cruce = time.perf_counter() - t0_cruce
            
            pct_1 = (victorias_1 / num_partidas_por_cruce) * 100.0
            pct_2 = (victorias_2 / num_partidas_por_cruce) * 100.0
            
            matriz_victorias_a_vs_b[i][j] = pct_1
            matriz_victorias_a_vs_b[j][i] = pct_2
            
            print(f" -> Resultado: {victorias_1}-{empates}-{victorias_2} ({t_cruce:.2f}s)")
            
    tiempo_total_torneo = time.perf_counter() - tiempo_inicio_torneo
    print("-" * 75)
    print(f"Torneo finalizado en {tiempo_total_torneo:.2f} segundos.")
    
    # -------------------------------------------------------------
    # 1. TABLA EN CONSOLA Y LATEX
    # -------------------------------------------------------------
    agentes_ordenados = sorted(agentes, key=lambda a: a.porcentaje_puntos_efectivos, reverse=True)
    
    print("\n" + "=" * 105)
    print(f"{'TABLA DE RESULTADOS GENERALES DEL BENCHMARK':^105}")
    print("=" * 105)
    print(f"{'Agente / Algoritmo':<22} | {'Partidas':<8} | {'V':<4} | {'E':<4} | {'D':<4} | {'Win Rate (%)':<12} | {'Score (%)':<10} | {'Nodos/Turno':<14} | {'T/Turno (s)':<12} | {'Fichas':<6}")
    print("-" * 105)
    for a in agentes_ordenados:
        print(f"{a.nombre:<22} | {a.partidas_jugadas:<8} | {a.victorias:<4} | {a.empates:<4} | {a.derrotas:<4} | {a.porcentaje_victorias:<12.1f} | {a.porcentaje_puntos_efectivos:<10.1f} | {a.nodos_medio_turno:<14.1f} | {a.tiempo_medio_turno:<12.5f} | {a.media_fichas_finales:<6.1f}")
    print("=" * 105)
    
    # Generar tabla LaTeX booktabs
    codigo_latex = generar_tabla_latex(agentes_ordenados)
    print("\n--- CÓDIGO LATEX PARA MEMORIA (booktabs) ---")
    print(codigo_latex)
    
    ruta_latex = os.path.join(ruta_carpeta, "tabla_benchmark.tex")
    with open(ruta_latex, "w", encoding="utf-8") as f:
        f.write(codigo_latex)
    print(f"\n[+] Tabla LaTeX guardada en: {ruta_latex}")
    
    # -------------------------------------------------------------
    # 2. GUARDAR RESULTADOS EN CSV
    # -------------------------------------------------------------
    ruta_csv = os.path.join(ruta_carpeta, "benchmark_resultados.csv")
    with open(ruta_csv, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Agente", "Id_Algoritmo", "Partidas", "Victorias", "Empates",
            "Derrotas", "WinRate_Pct", "ScoreEfectivo_Pct", "NodosMedioTurno",
            "TiempoMedioTurno_Seg", "FichasMediasFinales"
        ])
        for a in agentes_ordenados:
            writer.writerow([
                a.nombre, a.id_algoritmo, a.partidas_jugadas, a.victorias, a.empates,
                a.derrotas, round(a.porcentaje_victorias, 2), round(a.porcentaje_puntos_efectivos, 2),
                round(a.nodos_medio_turno, 2), round(a.tiempo_medio_turno, 6), round(a.media_fichas_finales, 2)
            ])
    print(f"[+] Resultados exportados a CSV: {ruta_csv}")
    
    # -------------------------------------------------------------
    # 3. HEATMAP DE PORCENTAJE DE VICTORIAS
    # -------------------------------------------------------------
    ruta_heatmap = os.path.join(ruta_carpeta, "heatmap_victorias.png")
    generar_heatmap_victorias(agentes, matriz_victorias_a_vs_b, ruta_heatmap)
    print(f"[+] Heatmap de victorias guardado en: {ruta_heatmap}")
    
    # -------------------------------------------------------------
    # 4. GRÁFICA DE COMPLEJIDAD: NODOS Y TIEMPOS VS PROFUNDIDAD
    # -------------------------------------------------------------
    ruta_complejidad = os.path.join(ruta_carpeta, "complejidad_nodos_tiempo.png")
    generar_grafica_complejidad(agentes_ordenados, ruta_complejidad)
    print(f"[+] Gráfica de complejidad (Nodos vs Tiempo) guardada en: {ruta_complejidad}")
    print("=" * 75 + "\n")
    
    return agentes_ordenados, matriz_victorias_a_vs_b


def generar_tabla_latex(agentes):
    """Construye el string de código LaTeX con formato booktabs para la memoria del TFG."""
    lineas = [
        r"\begin{table}[htbp]",
        r"  \centering",
        r"  \caption{Comparativa de Rendimiento de Algoritmos y Heurísticas en Othello 8$\times$8}",
        r"  \label{tab:benchmark_clasico}",
        r"  \begin{tabular}{lccccccrr}",
        r"    \toprule",
        r"    \textbf{Agente / Algoritmo} & \textbf{Partidas} & \textbf{V} & \textbf{E} & \textbf{D} & \textbf{Win Rate (\%)} & \textbf{Score (\%)} & \textbf{Nodos/Turno} & \textbf{T/Turno (s)} \\",
        r"    \midrule"
    ]
    for a in agentes:
        nombre_tex = a.nombre.replace("&", r"\&").replace("%", r"\%")
        lineas.append(
            f"    {nombre_tex:<28} & {a.partidas_jugadas:>4} & {a.victorias:>3} & {a.empates:>3} & {a.derrotas:>3} & {a.porcentaje_victorias:>6.1f}\\% & {a.porcentaje_puntos_efectivos:>6.1f}\\% & {a.nodos_medio_turno:>10.1f} & {a.tiempo_medio_turno:>9.5f} \\\\"
        )
    lineas.extend([
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}"
    ])
    return "\n".join(lineas)


def generar_heatmap_victorias(agentes, matriz_victorias, ruta_salida):
    """Genera y guarda un heatmap visualmente nítido con la matriz cruzada de victorias."""
    n = len(agentes)
    nombres = [a.nombre for a in agentes]
    
    plt.figure(figsize=(10, 8))
    
    matriz_plot = np.copy(matriz_victorias)
    for i in range(n):
        matriz_plot[i, i] = 50.0
        
    im = plt.imshow(matriz_plot, cmap="Blues", vmin=0, vmax=100, aspect="auto")
    
    cbar = plt.colorbar(im)
    cbar.set_label("Porcentaje de Victorias (% Fila vs Columna)", fontsize=11, fontweight="bold")
    
    plt.xticks(range(n), nombres, rotation=35, ha="right", fontsize=10, fontweight="bold")
    plt.yticks(range(n), nombres, fontsize=10, fontweight="bold")
    plt.title("Matriz Cruzada de Rendimiento (% Victorias Fila frente a Columna)", fontsize=13, fontweight="bold", pad=15)
    
    for i in range(n):
        for j in range(n):
            if i == j:
                texto = "-"
                color_texto = "gray"
            else:
                val = matriz_victorias[i, j]
                texto = f"{val:.1f}%"
                color_texto = "white" if val > 65 else "black"
            plt.text(j, i, texto, ha="center", va="center", color=color_texto, fontsize=10, fontweight="bold")
            
    plt.tight_layout()
    plt.savefig(ruta_salida, dpi=300)
    plt.close()


def generar_grafica_complejidad(agentes, ruta_salida):
    """
    Genera un gráfico comparativo con dos ejes Y:
      - Eje Y Izquierdo (Azul): Nodos evaluados por turno (Escala Logarítmica).
      - Eje Y Derecho (Rojo/Naranja): Tiempo medio por turno en segundos (Escala Logarítmica).
    """
    # Ordenar por tiempo medio ascendente para visualizar la curva de complejidad
    agentes_ordenados = sorted(agentes, key=lambda a: a.tiempo_medio_turno)
    nombres = [a.nombre for a in agentes_ordenados]
    nodos = [max(a.nodos_medio_turno, 1.0) for a in agentes_ordenados]
    tiempos = [max(a.tiempo_medio_turno, 1e-6) for a in agentes_ordenados]
    
    x = np.arange(len(nombres))
    ancho = 0.38
    
    fig, ax1 = plt.subplots(figsize=(12, 6.5))
    
    # Eje 1: Nodos evaluados
    color_nodos = "#1f77b4"
    ax1.set_xlabel("Agente / Configuración de Búsqueda", fontsize=11, fontweight="bold", labelpad=10)
    ax1.set_ylabel("Nodos Evaluados por Turno (Escala Logarítmica)", color=color_nodos, fontsize=11, fontweight="bold")
    barras_nodos = ax1.bar(x - ancho/2, nodos, ancho, label="Nodos / Turno", color=color_nodos, edgecolor="black", alpha=0.85)
    ax1.set_yscale("log")
    ax1.tick_params(axis='y', labelcolor=color_nodos)
    ax1.set_xticks(x)
    ax1.set_xticklabels(nombres, rotation=28, ha="right", fontsize=10, fontweight="bold")
    ax1.grid(axis='y', linestyle='--', alpha=0.5, which='both')
    
    # Eje 2: Tiempo de cálculo
    ax2 = ax1.twinx()
    color_tiempo = "#d62728"
    ax2.set_ylabel("Tiempo Medio por Turno (Segundos - Escala Logarítmica)", color=color_tiempo, fontsize=11, fontweight="bold")
    barras_tiempo = ax2.bar(x + ancho/2, tiempos, ancho, label="Tiempo / Turno (s)", color=color_tiempo, edgecolor="black", alpha=0.85)
    ax2.set_yscale("log")
    ax2.tick_params(axis='y', labelcolor=color_tiempo)
    
    # Añadir valores sobre las barras
    for barra, n_val in zip(barras_nodos, nodos):
        h = barra.get_height()
        lbl = f"{int(n_val):,}" if n_val >= 10 else f"{n_val:.1f}"
        ax1.text(barra.get_x() + barra.get_width() / 2., h * 1.15, lbl, ha='center', va='bottom', fontsize=8, color="#0b407a", fontweight="bold", rotation=0)
        
    for barra, t_val in zip(barras_tiempo, tiempos):
        h = barra.get_height()
        lbl = f"{t_val*1000:.1f}ms" if t_val < 0.01 else f"{t_val:.3f}s"
        ax2.text(barra.get_x() + barra.get_width() / 2., h * 1.15, lbl, ha='center', va='bottom', fontsize=8, color="#8a0f0f", fontweight="bold", rotation=0)
        
    plt.title("Complejidad Computacional en Othello 8x8: Nodos Explorados vs Tiempo de Inferencia", fontsize=13, fontweight="bold", pad=15)
    
    # Leyenda combinada
    lineas1, etiquetas1 = ax1.get_legend_handles_labels()
    lineas2, etiquetas2 = ax2.get_legend_handles_labels()
    ax1.legend(lineas1 + lineas2, etiquetas1 + etiquetas2, loc="upper left", fontsize=10)
    
    plt.tight_layout()
    plt.savefig(ruta_salida, dpi=300)
    plt.close()


def verificar_teorema_zermelo(num_partidas=10, carpeta_salida="pruebas/benchmark_clasico"):
    """
    Verificación empírica del Teorema de Zermelo (1913) en Tres en Raya:
    Dos agentes que siguen la estrategia Minimax universal óptima deben
    empatar el 100% de las partidas (juego resuelto / tablas teóricas).
    """
    print("\n" + "=" * 75)
    print("=== VERIFICACIÓN EMPÍRICA DEL TEOREMA DE ZERMELO (TRES EN RAYA) ===")
    print("=" * 75)
    print("Teorema de Zermelo (1913): En todo juego finito bipersonal de información")
    print("perfecta y suma cero sin azar, existe una estrategia óptima que garantiza")
    print("la victoria a uno de los jugadores o el empate a ambos.")
    print(f"\nEjecutando {num_partidas} partidas de Minimax Universal vs Minimax Universal...")
    print("-" * 75)
    
    juego = TresEnRaya()
    victorias_x = 0
    victorias_o = 0
    empates = 0
    tiempos_turnos = []
    nodos_turnos = []
    
    inicio_verificacion = time.perf_counter()
    
    for partida_idx in range(1, num_partidas + 1):
        estado, turno_actual = juego.estado_inicial()
        numero_turno = 1
        
        while not juego.es_terminal(estado):
            movimientos_validos = juego.obtener_movimientos_legales(estado, turno_actual)
            
            if movimientos_validos == [None]:
                movimiento_elegido = None
            else:
                movimiento_elegido, t_mov, nodos_mov = obtener_movimiento_maquina(
                    juego, estado, turno_actual, "universal", silencioso=True,
                    turnos_aleatorios_inicio=0, epsilon=0.0
                )
                tiempos_turnos.append(t_mov)
                nodos_turnos.append(nodos_mov)
                
            estado = juego.hacer_movimiento(estado, movimiento_elegido, turno_actual)
            turno_actual = juego.obtener_rival(turno_actual)
            numero_turno += 1
            
        utilidad_x = juego.obtener_utilidad(estado, 'X')
        if utilidad_x == 1:
            victorias_x += 1
            resultado_str = "Victoria X"
        elif utilidad_x == -1:
            victorias_o += 1
            resultado_str = "Victoria O"
        else:
            empates += 1
            resultado_str = "Empate (Tablas)"
            
        print(f"  Partida {partida_idx:02d}/{num_partidas:02d}: Resultado = {resultado_str} ({numero_turno-1} turnos)")
        
    tiempo_total = time.perf_counter() - inicio_verificacion
    pct_empates = (empates / num_partidas) * 100.0
    media_nodos = np.mean(nodos_turnos) if nodos_turnos else 0.0
    media_tiempo = np.mean(tiempos_turnos) if tiempos_turnos else 0.0
    
    print("-" * 75)
    print(f"Total de partidas: {num_partidas}")
    print(f"Victorias de X: {victorias_x} ({victorias_x/num_partidas*100:.1f}%)")
    print(f"Victorias de O: {victorias_o} ({victorias_o/num_partidas*100:.1f}%)")
    print(f"Empates (Tablas): {empates} ({pct_empates:.1f}%)")
    print(f"Nodos explorados medios por turno: {media_nodos:.1f}")
    print(f"Tiempo medio por turno: {media_tiempo:.4f} segundos")
    print(f"Tiempo total: {tiempo_total:.2f} segundos")
    
    if pct_empates == 100.0:
        print("\n[*] RESULTADO: TEOREMA DE ZERMELO VERIFICADO AL 100%.")
        print("    Minimax exhaustivo garantiza el empate forzado en Tres en Raya.")
    else:
        print("\n[!] Discrepancia detectada respecto al resultado teórico.")
    print("=" * 75 + "\n")
    
    # Guardar reporte de Zermelo
    directorio_script = os.path.dirname(os.path.abspath(__file__))
    ruta_carpeta = os.path.join(directorio_script, carpeta_salida) if not os.path.isabs(carpeta_salida) else carpeta_salida
    os.makedirs(ruta_carpeta, exist_ok=True)
    ruta_reporte = os.path.join(ruta_carpeta, "verificacion_teorema_zermelo.txt")
    with open(ruta_reporte, "w", encoding="utf-8") as f:
        f.write("=== VERIFICACION EMPIRICA DEL TEOREMA DE ZERMELO ===\n")
        f.write(f"Juego: Tres en Raya (Minimax Universal vs Minimax Universal)\n")
        f.write(f"Partidas: {num_partidas}\n")
        f.write(f"Empates: {empates}/{num_partidas} ({pct_empates:.1f}%)\n")
        f.write(f"Victorias X: {victorias_x}\n")
        f.write(f"Victorias O: {victorias_o}\n")
        f.write(f"Nodos medios por turno: {media_nodos:.1f}\n")
        f.write(f"Tiempo medio por turno: {media_tiempo:.6f} s\n")
        f.write(f"Resultado: {'VERIFICADO (100% Empates)' if pct_empates == 100.0 else 'FALLO'}\n")
    print(f"[+] Informe de verificación guardado en: {ruta_reporte}")


def menu_benchmark():
    """Menú principal interactivo de benchmark."""
    print("=" * 65)
    print(f"{'BENCHMARK DE ALGORITMOS Y HEURÍSTICAS':^65}")
    print("=" * 65)
    print("1. Ejecutar torneo estándar predefinido (Othello 8x8):")
    print("   - Pesos (d=2, d=4)")
    print("   - Movilidad (d=2, d=4, d=6)")
    print("   - Híbrida (d=4)")
    print("   - Aleatorio (Random)")
    print("   (10 partidas por cruce alternando colores)")
    print("2. Configurar torneo personalizado (Othello 8x8)")
    print("3. Verificación empírica del Teorema de Zermelo (Tres en Raya)")
    print("4. Salir")
    print("-" * 65)
    
    while True:
        opcion = input("Elige una opción (1-4): ").strip()
        if opcion == '1':
            agentes_estandar = [
                AgenteBenchmark("Pesos (d=2)", "alfa_beta_pesos|2"),
                AgenteBenchmark("Pesos (d=4)", "alfa_beta_pesos|4"),
                AgenteBenchmark("Movilidad (d=2)", "alfa_beta_movilidad|2"),
                AgenteBenchmark("Movilidad (d=4)", "alfa_beta_movilidad|4"),
                AgenteBenchmark("Movilidad (d=6)", "alfa_beta_movilidad|6"),
                AgenteBenchmark("Híbrida (d=4)", "alfa_beta_hibrida|4"),
                AgenteBenchmark("Random", "random")
            ]
            ejecutar_torneo(
                agentes_estandar,
                num_partidas_por_cruce=10,
                turnos_aleatorios_inicio=0,
                epsilon=0.0,
                carpeta_salida="pruebas/benchmark_clasico"
            )
            break
        elif opcion == '2':
            print("\n--- Configuración de Torneo Personalizado ---")
            catalogo_posible = [
                ("Pesos (d=2)", "alfa_beta_pesos|2"),
                ("Pesos (d=4)", "alfa_beta_pesos|4"),
                ("Pesos (d=6)", "alfa_beta_pesos|6"),
                ("Movilidad (d=2)", "alfa_beta_movilidad|2"),
                ("Movilidad (d=4)", "alfa_beta_movilidad|4"),
                ("Movilidad (d=6)", "alfa_beta_movilidad|6"),
                ("Híbrida (d=2)", "alfa_beta_hibrida|2"),
                ("Híbrida (d=4)", "alfa_beta_hibrida|4"),
                ("Híbrida (d=6)", "alfa_beta_hibrida|6"),
                ("MCTS Neuronal (s=30)", "mcts_neuronal|30"),
                ("MCTS Neuronal (s=50)", "mcts_neuronal|50"),
                ("MCTS Neuronal (s=100)", "mcts_neuronal|100"),
                ("Random", "random")
            ]
            print("\nAgentes disponibles:")
            for idx, (nombre_ag, _) in enumerate(catalogo_posible, 1):
                print(f"  {idx:02d}. {nombre_ag}")
                
            entrada_indices = input("\nElige los números de los agentes separados por coma o espacio (ej. '1, 2, 4, 8, 13') [Enter = estándar]: ").strip()
            if entrada_indices == "":
                seleccionados = [0, 1, 3, 4, 5, 7, 12]
            else:
                indices_parseados = []
                for p in entrada_indices.replace(",", " ").split():
                    try:
                        val = int(p) - 1
                        if 0 <= val < len(catalogo_posible):
                            indices_parseados.append(val)
                    except ValueError:
                        pass
                seleccionados = list(dict.fromkeys(indices_parseados))
                if len(seleccionados) < 2:
                    print("Se deben seleccionar al menos 2 agentes. Usando agentes estándar.")
                    seleccionados = [0, 1, 3, 4, 5, 7, 12]
                    
            agentes_personalizados = [
                AgenteBenchmark(catalogo_posible[i][0], catalogo_posible[i][1])
                for i in seleccionados
            ]
            
            entrada_partidas = input("Partidas por cruce [Enter = 10]: ").strip()
            try:
                num_partidas = int(entrada_partidas) if entrada_partidas != "" else 10
            except ValueError:
                num_partidas = 10
                
            entrada_turnos = input("Turnos iniciales aleatorios (aperturas) [Enter = 0]: ").strip()
            try:
                turnos_aleatorios = int(entrada_turnos) if entrada_turnos != "" else 0
            except ValueError:
                turnos_aleatorios = 0
                
            entrada_eps = input("Probabilidad epsilon-greedy [Enter = 0.0]: ").strip()
            try:
                eps = float(entrada_eps) if entrada_eps != "" else 0.0
            except ValueError:
                eps = 0.0
                
            entrada_carpeta = input("Carpeta de destino [Enter = pruebas/benchmark_personalizado]: ").strip()
            carpeta_dest = "pruebas/benchmark_personalizado" if entrada_carpeta == "" else entrada_carpeta
            
            ejecutar_torneo(
                agentes_personalizados,
                num_partidas_por_cruce=num_partidas,
                turnos_aleatorios_inicio=turnos_aleatorios,
                epsilon=eps,
                carpeta_salida=carpeta_dest
            )
            break
        elif opcion == '3':
            entrada_partidas = input("\nNúmero de partidas de Tres en Raya para verificar Zermelo [Enter = 10]: ").strip()
            try:
                n_part = int(entrada_partidas) if entrada_partidas != "" else 10
            except ValueError:
                n_part = 10
            verificar_teorema_zermelo(num_partidas=n_part)
            break
        elif opcion == '4':
            print("Saliendo del benchmark.")
            break
        print("Opción no válida. Inténtalo de nuevo.")

if __name__ == "__main__":
    menu_benchmark()
