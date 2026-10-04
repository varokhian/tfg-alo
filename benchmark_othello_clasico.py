import os
import sys
import time
import csv
import numpy as np
import matplotlib.pyplot as plt

# Añadir la raíz del proyecto al sys.path
directorio_script = os.path.dirname(os.path.abspath(__file__))
directorio_ptfg = os.path.abspath(os.path.join(directorio_script, "..", "..", ".."))
if directorio_ptfg not in sys.path:
    sys.path.insert(0, directorio_ptfg)

from juegos import Othello
from main import obtener_movimiento_maquina

class AgenteOthello:
    """Clase para registrar métricas de cada agente en el torneo de Othello."""
    def __init__(self, nombre, id_algoritmo, heuristica=None, profundidad=None):
        self.nombre = nombre
        self.id_algoritmo = id_algoritmo
        self.heuristica = heuristica
        self.profundidad = profundidad
        
        # Métricas agregadas
        self.partidas_jugadas = 0
        self.victorias = 0
        self.empates = 0
        self.derrotas = 0
        
        self.diferencial_total_fichas = 0
        self.tiempo_total_calculo = 0.0
        self.nodos_totales_evaluados = 0
        self.turnos_totales_jugados = 0

    @property
    def winrate(self):
        if self.partidas_jugadas == 0:
            return 0.0
        return (self.victorias / self.partidas_jugadas) * 100.0

    @property
    def score_efectivo(self):
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
    def diferencial_fichas_medio(self):
        if self.partidas_jugadas == 0:
            return 0.0
        return self.diferencial_total_fichas / self.partidas_jugadas


def jugar_partida_othello(juego, agente_n, agente_b):
    """
    Ejecuta una partida determinista de Othello 8x8 entre dos agentes.
    Retorna: (ganador, puntos_n, puntos_b, tiempo_total_partida, turnos_partida, nodos_n, nodos_b, t_n, t_b)
    """
    estado, turno_actual = juego.estado_inicial()
    numero_turno = 1
    inicio_partida = time.perf_counter()
    
    nodos_n = 0
    nodos_b = 0
    t_calc_n = 0.0
    t_calc_b = 0.0
    
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
                turnos_aleatorios_inicio=0,
                epsilon=0.0
            )
            
            if turno_actual == 'N':
                nodos_n += nodos_mov
                t_calc_n += t_mov
                agente_n.tiempo_total_calculo += t_mov
                agente_n.nodos_totales_evaluados += nodos_mov
                agente_n.turnos_totales_jugados += 1
            else:
                nodos_b += nodos_mov
                t_calc_b += t_mov
                agente_b.tiempo_total_calculo += t_mov
                agente_b.nodos_totales_evaluados += nodos_mov
                agente_b.turnos_totales_jugados += 1
                
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
        
    return ganador, puntos_n, puntos_b, tiempo_total_partida, numero_turno - 1, nodos_n, nodos_b, t_calc_n, t_calc_b


def generar_tabla_latex(agentes):
    """Genera la tabla LaTeX compacta en formato booktabs."""
    lineas = [
        r"\begin{table}[htbp]",
        r"  \centering",
        r"  \caption{Evaluación Experimental de Heurísticas y Algoritmos Clásicos en Othello 8$\times$8}",
        r"  \label{tab:resumen_othello_clasico}",
        r"  \begin{tabular}{lcccc}",
        r"    \toprule",
        r"    \textbf{Agente / Algoritmo} & \textbf{Winrate (\%)} & \textbf{Nodos / Turno} & \textbf{Tiempo / Turno (s)} & \textbf{$\Delta$ Fichas} \\",
        r"    \midrule"
    ]
    for a in agentes:
        nombre_tex = a.nombre.replace("_", r"\_")
        signo_dif = "+" if a.diferencial_fichas_medio > 0 else ""
        dif_str = f"{signo_dif}{a.diferencial_fichas_medio:.1f}"
        lineas.append(
            f"    {nombre_tex:<26} & {a.winrate:>6.1f}\\% & {a.nodos_medio_turno:>12.1f} & {a.tiempo_medio_turno:>14.5f} & {dif_str:>9} \\\\"
        )
    lineas.extend([
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}"
    ])
    return "\n".join(lineas)


def generar_heatmap_victorias(agentes, matriz_victorias, ruta_salida):
    """
    Genera el heatmap de porcentaje de victorias de la matriz cruzada N x N a 300 DPI.
    Asegura la diagonalización ordenando de mejor a peor agente si no viniesen ya ordenados.
    """
    # Si los agentes no están ordenados por rendimiento, reordenar agentes y matriz
    if len(agentes) > 0 and hasattr(agentes[0], 'score_efectivo'):
        agentes_ordenados = sorted(agentes, key=lambda a: (a.score_efectivo, a.diferencial_fichas_medio), reverse=True)
        if agentes != agentes_ordenados:
            indices_ordenados = [agentes.index(a) for a in agentes_ordenados]
            matriz_victorias = matriz_victorias[np.ix_(indices_ordenados, indices_ordenados)]
            agentes = agentes_ordenados
            
    n = len(agentes)
    nombres = [a.nombre.replace("_", " ") for a in agentes]
    
    plt.figure(figsize=(10.5, 8.5))
    
    matriz_plot = np.copy(matriz_victorias)
    for i in range(n):
        matriz_plot[i, i] = 50.0
        
    im = plt.imshow(matriz_plot, cmap="Blues", vmin=0, vmax=100, aspect="auto")
    
    cbar = plt.colorbar(im)
    cbar.set_label("Porcentaje de Victorias (% Fila vs Columna)", fontsize=11, fontweight="bold")
    
    plt.xticks(range(n), nombres, rotation=35, ha="right", fontsize=9.5, fontweight="bold")
    plt.yticks(range(n), nombres, fontsize=9.5, fontweight="bold")
    plt.title("Matriz Cruzada de Victorias en Othello 8x8 (% Fila frente a Columna)", fontsize=13, fontweight="bold", pad=15)
    
    for i in range(n):
        for j in range(n):
            if i == j:
                texto = "-"
                color_texto = "gray"
            else:
                val = matriz_victorias[i, j]
                texto = f"{val:.1f}%"
                color_texto = "white" if val > 65 else "black"
            plt.text(j, i, texto, ha="center", va="center", color=color_texto, fontsize=9.5, fontweight="bold")
            
    plt.tight_layout()
    plt.savefig(ruta_salida, dpi=300)
    plt.close()


def generar_grafico_complejidad(agentes, ruta_salida):
    """
    Genera dos subgráficas comparativas:
      - Subgráfica 1: Profundidad vs Tiempo/Turno (s) en escala logarítmica.
      - Subgráfica 2: Profundidad vs Nodos/Turno en escala logarítmica.
    """
    # Agrupar por heurística
    heuristicas_config = {
        "Pesos": {"color": "#1f77b4", "marker": "o", "label": "Heurística Pesos"},
        "Movilidad": {"color": "#ff7f0e", "marker": "s", "label": "Heurística Movilidad"},
        "Hibrida": {"color": "#2ca02c", "marker": "^", "label": "Heurística Híbrida"}
    }
    
    agente_random = next((a for a in agentes if a.heuristica is None), None)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    for heur_key, cfg in heuristicas_config.items():
        agentes_heur = [a for a in agentes if a.heuristica == heur_key and a.profundidad is not None]
        agentes_heur.sort(key=lambda a: a.profundidad)
        
        if agentes_heur:
            profs = [a.profundidad for a in agentes_heur]
            tiempos = [max(a.tiempo_medio_turno, 1e-6) for a in agentes_heur]
            nodos = [max(a.nodos_medio_turno, 1.0) for a in agentes_heur]
            
            # Subplot 1: Tiempo vs Profundidad
            ax1.plot(profs, tiempos, marker=cfg["marker"], color=cfg["color"], label=cfg["label"], linewidth=2.2, markersize=8)
            for p, t in zip(profs, tiempos):
                lbl = f"{t*1000:.1f}ms" if t < 0.01 else f"{t:.3f}s"
                ax1.annotate(lbl, (p, t), textcoords="offset points", xytext=(0, 8), ha='center', fontsize=8.5, fontweight="bold")
                
            # Subplot 2: Nodos vs Profundidad
            ax2.plot(profs, nodos, marker=cfg["marker"], color=cfg["color"], label=cfg["label"], linewidth=2.2, markersize=8)
            for p, nd in zip(profs, nodos):
                lbl = f"{int(nd):,}"
                ax2.annotate(lbl, (p, nd), textcoords="offset points", xytext=(0, 8), ha='center', fontsize=8.5, fontweight="bold")
                
    # Línea base de Random si existe
    if agente_random:
        t_rand = max(agente_random.tiempo_medio_turno, 1e-6)
        n_rand = max(agente_random.nodos_medio_turno, 1.0)
        ax1.axhline(t_rand, color="#7f7f7f", linestyle="--", alpha=0.8, label=f"Random ({t_rand*1000:.2f}ms)")
        ax2.axhline(n_rand, color="#7f7f7f", linestyle="--", alpha=0.8, label=f"Random ({n_rand:.0f} nodo)")
        
    # Configuración Subplot 1 (Tiempo)
    ax1.set_yscale("log")
    ax1.set_xlabel("Profundidad de Búsqueda ($d$)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Tiempo Medio por Turno (Segundos - Escala Log)", fontsize=11, fontweight="bold")
    ax1.set_title("(a) Coste Temporal vs Profundidad", fontsize=12, fontweight="bold")
    ax1.set_xticks([2, 4, 6])
    ax1.grid(True, linestyle="--", alpha=0.6, which="both")
    ax1.legend(fontsize=9.5)
    
    # Configuración Subplot 2 (Nodos)
    ax2.set_yscale("log")
    ax2.set_xlabel("Profundidad de Búsqueda ($d$)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Nodos Evaluados por Turno (Escala Log)", fontsize=11, fontweight="bold")
    ax2.set_title("(b) Espacio de Estados Explorados vs Profundidad", fontsize=12, fontweight="bold")
    ax2.set_xticks([2, 4, 6])
    ax2.grid(True, linestyle="--", alpha=0.6, which="both")
    ax2.legend(fontsize=9.5)
    
    plt.suptitle("Análisis de Complejidad Computacional en Othello 8x8 (Poda Alfa-Beta)", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    plt.savefig(ruta_salida, dpi=300)
    plt.close()


def ejecutar_torneo_othello(partidas_por_cruce=4):
    """
    Ejecuta el torneo experimental definitivo de Othello 8x8 con los 8 agentes clásicos.
    """
    # Directorios de salida
    subcarpeta_visual = os.path.join(directorio_script, "figuras")
    os.makedirs(subcarpeta_visual, exist_ok=True)
    
    # 1. Definición de los 8 Agentes
    agentes = [
        AgenteOthello("Random", "random"),
        AgenteOthello("AlfaBeta_Pesos_d2", "alfa_beta_pesos|2", heuristica="Pesos", profundidad=2),
        AgenteOthello("AlfaBeta_Pesos_d4", "alfa_beta_pesos|4", heuristica="Pesos", profundidad=4),
        AgenteOthello("AlfaBeta_Movilidad_d2", "alfa_beta_movilidad|2", heuristica="Movilidad", profundidad=2),
        AgenteOthello("AlfaBeta_Movilidad_d4", "alfa_beta_movilidad|4", heuristica="Movilidad", profundidad=4),
        AgenteOthello("AlfaBeta_Movilidad_d6", "alfa_beta_movilidad|6", heuristica="Movilidad", profundidad=6),
        AgenteOthello("AlfaBeta_Hibrida_d2", "alfa_beta_hibrida|2", heuristica="Hibrida", profundidad=2),
        AgenteOthello("AlfaBeta_Hibrida_d4", "alfa_beta_hibrida|4", heuristica="Hibrida", profundidad=4),
    ]
    
    num_agentes = len(agentes)
    total_cruces = (num_agentes * (num_agentes - 1)) // 2
    total_partidas_torneo = total_cruces * partidas_por_cruce
    
    print("=" * 85)
    print("=== CAPÍTULO 5: TORNEO EXPERIMENTAL DEFINITIVO EN OTHELLO 8x8 ===")
    print("=" * 85)
    print(f"Número de Agentes: {num_agentes}")
    print(f"Partidas por Cruce: {partidas_por_cruce} (Alternancia estricta: 50% Negras / 50% Blancas)")
    print(f"Total de Cruces: {total_cruces} | Total de Partidas: {total_partidas_torneo}")
    print(f"Modo: Determinista (turnos_aleatorios_inicio=0, epsilon=0.0)")
    print(f"Directorio de salida: {directorio_script}")
    print(f"Subcarpeta visual para memoria: {subcarpeta_visual}")
    print("-" * 85)
    
    juego = Othello(tamano=8)
    matriz_victorias = np.full((num_agentes, num_agentes), np.nan)
    registro_partidas_individuales = []
    
    cruce_actual = 0
    id_partida_global = 1
    tiempo_inicio_torneo = time.perf_counter()
    
    for i in range(num_agentes):
        for j in range(i + 1, num_agentes):
            cruce_actual += 1
            agente_1 = agentes[i]
            agente_2 = agentes[j]
            
            mitad = partidas_por_cruce // 2
            resto = partidas_por_cruce - mitad
            
            print(f"[{cruce_actual:02d}/{total_cruces:02d}] Cruce: {agente_1.nombre} vs {agente_2.nombre} ({partidas_por_cruce} partidas)...", end="", flush=True)
            t0_cruce = time.perf_counter()
            
            victorias_1 = 0
            victorias_2 = 0
            empates = 0
            
            # Bloque 1: Agente 1 (Negras) vs Agente 2 (Blancas)
            for _ in range(mitad):
                ganador, pn, pb, t_part, turnos, nod_n, nod_b, t_n, t_b = jugar_partida_othello(juego, agente_1, agente_2)
                agente_1.partidas_jugadas += 1
                agente_2.partidas_jugadas += 1
                agente_1.diferencial_total_fichas += (pn - pb)
                agente_2.diferencial_total_fichas += (pb - pn)
                
                if ganador == 'N':
                    agente_1.victorias += 1
                    agente_2.derrotas += 1
                    victorias_1 += 1
                    res_txt = "Victoria Negras (Agente 1)"
                elif ganador == 'B':
                    agente_2.victorias += 1
                    agente_1.derrotas += 1
                    victorias_2 += 1
                    res_txt = "Victoria Blancas (Agente 2)"
                else:
                    agente_1.empates += 1
                    agente_2.empates += 1
                    empates += 1
                    res_txt = "Empate"
                    
                registro_partidas_individuales.append([
                    id_partida_global, cruce_actual, agente_1.nombre, agente_2.nombre,
                    "N", "B", pn, pb, (pn - pb), ganador, res_txt,
                    round(t_part, 4), turnos, nod_n, nod_b, round(t_n, 4), round(t_b, 4)
                ])
                id_partida_global += 1

            # Bloque 2: Agente 2 (Negras) vs Agente 1 (Blancas)
            for _ in range(resto):
                ganador, pn, pb, t_part, turnos, nod_n, nod_b, t_n, t_b = jugar_partida_othello(juego, agente_2, agente_1)
                agente_1.partidas_jugadas += 1
                agente_2.partidas_jugadas += 1
                agente_2.diferencial_total_fichas += (pn - pb)
                agente_1.diferencial_total_fichas += (pb - pn)
                
                if ganador == 'N':
                    agente_2.victorias += 1
                    agente_1.derrotas += 1
                    victorias_2 += 1
                    res_txt = "Victoria Negras (Agente 2)"
                elif ganador == 'B':
                    agente_1.victorias += 1
                    agente_2.derrotas += 1
                    victorias_1 += 1
                    res_txt = "Victoria Blancas (Agente 1)"
                else:
                    agente_1.empates += 1
                    agente_2.empates += 1
                    empates += 1
                    res_txt = "Empate"
                    
                registro_partidas_individuales.append([
                    id_partida_global, cruce_actual, agente_2.nombre, agente_1.nombre,
                    "N", "B", pn, pb, (pn - pb), ganador, res_txt,
                    round(t_part, 4), turnos, nod_n, nod_b, round(t_n, 4), round(t_b, 4)
                ])
                id_partida_global += 1
                
            t_cruce = time.perf_counter() - t0_cruce
            pct_1 = (victorias_1 / partidas_por_cruce) * 100.0
            pct_2 = (victorias_2 / partidas_por_cruce) * 100.0
            matriz_victorias[i][j] = pct_1
            matriz_victorias[j][i] = pct_2
            
            print(f" -> {victorias_1}-{empates}-{victorias_2} ({t_cruce:.2f}s)")
            
    tiempo_total_torneo = time.perf_counter() - tiempo_inicio_torneo
    print("-" * 85)
    print(f"Torneo completado en {tiempo_total_torneo:.2f} segundos.")
    
    # -------------------------------------------------------------
    # 1. TABLA RESUMEN EN CONSOLA Y LATEX
    # -------------------------------------------------------------
    agentes_ordenados = sorted(agentes, key=lambda a: (a.score_efectivo, a.diferencial_fichas_medio), reverse=True)
    
    print("\n" + "=" * 95)
    print(f"{'TABLA DE RESULTADOS GENERALES (OTHELLO 8x8)':^95}")
    print("=" * 95)
    print(f"{'Agente / Algoritmo':<24} | {'Partidas':<8} | {'V':<4} | {'E':<3} | {'D':<4} | {'Winrate (%)':<11} | {'Nodos/Turno':<13} | {'T/Turno (s)':<11} | {'Δ Fichas':<8}")
    print("-" * 95)
    for a in agentes_ordenados:
        signo = "+" if a.diferencial_fichas_medio > 0 else ""
        dif_str = f"{signo}{a.diferencial_fichas_medio:.1f}"
        print(f"{a.nombre:<24} | {a.partidas_jugadas:<8} | {a.victorias:<4} | {a.empates:<3} | {a.derrotas:<4} | {a.winrate:<11.1f} | {a.nodos_medio_turno:<13.1f} | {a.tiempo_medio_turno:<11.5f} | {dif_str:<8}")
    print("=" * 95)
    
    # Generar tabla LaTeX
    codigo_latex = generar_tabla_latex(agentes_ordenados)
    print("\n" + "=" * 85)
    print("=== CÓDIGO LATEX LISTO PARA COMPILAR (tabla_othello_resumen.tex) ===")
    print("=" * 85)
    print(codigo_latex)
    print("=" * 85)
    
    # Guardar archivo .tex
    ruta_tex = os.path.join(directorio_script, "tabla_othello_resumen.tex")
    with open(ruta_tex, "w", encoding="utf-8") as f:
        f.write(codigo_latex)
    print(f"\n[+] Tabla LaTeX guardada en: {ruta_tex}")
    
    # -------------------------------------------------------------
    # 2. GUARDAR CSV DETALLADO
    # -------------------------------------------------------------
    ruta_csv = os.path.join(directorio_script, "benchmark_othello_resultados.csv")
    with open(ruta_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "ID_Partida", "ID_Cruce", "Agente_Negras", "Agente_Blancas",
            "Color_Negras", "Color_Blancas", "Fichas_Negras", "Fichas_Blancas",
            "Diferencial_Negras", "Ganador_Color", "Resultado_Texto",
            "Tiempo_Total_Segundos", "Turnos_Totales", "Nodos_Negras",
            "Nodos_Blancas", "Tiempo_Calc_Negras", "Tiempo_Calc_Blancas"
        ])
        for row in registro_partidas_individuales:
            writer.writerow(row)
    print(f"[+] CSV detallado de partidas guardado en: {ruta_csv}")
    
    # -------------------------------------------------------------
    # 3. GENERAR FIGURAS VISUALES EN LA SUBCARPETA 'figuras'
    # -------------------------------------------------------------
    # Reordenar matriz y agentes de mejor a peor para que el heatmap salga diagonalizado
    indices_ordenados = [agentes.index(a) for a in agentes_ordenados]
    matriz_victorias_ordenada = matriz_victorias[np.ix_(indices_ordenados, indices_ordenados)]
    
    ruta_heatmap = os.path.join(subcarpeta_visual, "heatmap_othello_clasico.png")
    generar_heatmap_victorias(agentes_ordenados, matriz_victorias_ordenada, ruta_heatmap)
    print(f"[+] Heatmap de victorias diagonalizado (300 DPI) guardado en: {ruta_heatmap}")
    
    # Guardar copia directa en la carpeta raíz del script (Capítulo_5_adjuntar)
    ruta_heatmap_directorio = os.path.join(directorio_script, "heatmap_othello_clasico.png")
    if os.path.abspath(ruta_heatmap) != os.path.abspath(ruta_heatmap_directorio):
        import shutil
        shutil.copy2(ruta_heatmap, ruta_heatmap_directorio)
        print(f"[+] Heatmap de victorias (300 DPI) copiado en: {ruta_heatmap_directorio}")
    
    ruta_complejidad = os.path.join(subcarpeta_visual, "complejidad_othello.png")
    generar_grafico_complejidad(agentes, ruta_complejidad)
    print(f"[+] Gráfico de complejidad (300 DPI) guardado en: {ruta_complejidad}")
    print("=" * 85 + "\n")
    
    return agentes_ordenados, matriz_victorias_ordenada


def menu_principal():
    """Menú interactivo para ejecutar el torneo experimental."""
    print("=" * 70)
    print(f"{'BENCHMARK EXPERIMENTAL DE OTHELLO 8x8 (CAPÍTULO 5)':^70}")
    print("=" * 70)
    print("1. Torneo Rápido (4 partidas por cruce - 2 Negras / 2 Blancas)")
    print("2. Torneo Completo (10 partidas por cruce - 5 Negras / 5 Blancas)")
    print("3. Torneo Personalizado (Elegir número de partidas por cruce)")
    print("4. Salir")
    print("-" * 70)
    
    while True:
        opcion = input("Selecciona una opción (1-4): ").strip()
        if opcion == '1':
            ejecutar_torneo_othello(partidas_por_cruce=4)
            break
        elif opcion == '2':
            ejecutar_torneo_othello(partidas_por_cruce=10)
            break
        elif opcion == '3':
            entrada = input("Introduce el número de partidas por cruce (par sugerido) [Enter = 4]: ").strip()
            try:
                num_p = int(entrada) if entrada != "" else 4
            except ValueError:
                num_p = 4
            ejecutar_torneo_othello(partidas_por_cruce=num_p)
            break
        elif opcion == '4':
            print("Saliendo del benchmark.")
            break
        print("Opción inválida. Inténtalo de nuevo.")

if __name__ == "__main__":
    menu_principal()
