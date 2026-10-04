import os
import sys
import copy
import time
import random
import csv
import shutil
import numpy as np
import torch
import matplotlib.pyplot as plt

# Añadir la raíz del proyecto al sys.path
directorio_actual = os.path.dirname(os.path.abspath(__file__))
directorio_ptfg = os.path.abspath(os.path.join(directorio_actual, "..", ".."))
if directorio_ptfg not in sys.path:
    sys.path.insert(0, directorio_ptfg)

from juegos import Othello
from minimax import minimax_universal, minimax_alfa_beta
from mcts import MCTSClasico, MCTSNeuronal

# Importar RedDual4x4 y MCTSPUCT de la Fase 3
sys.path.insert(0, os.path.join(directorio_actual, "..", "Fase_3_Tabula_Rasa_4x4"))
try:
    from tabula_rasa_4x4 import RedDual4x4, MCTSPUCT
except ImportError:
    RedDual4x4, MCTSPUCT = None, None

class RegistroAgente:
    """Clase para registrar las estadísticas acumuladas de un agente."""
    def __init__(self, nombre, descripcion=""):
        self.nombre = nombre
        self.descripcion = descripcion
        self.partidas_jugadas = 0
        self.victorias = 0
        self.empates = 0
        self.derrotas = 0
        self.diferencial_fichas_total = 0
        self.tiempo_total_calculo = 0.0
        self.nodos_totales = 0
        self.turnos_jugados = 0

    @property
    def winrate(self):
        if self.partidas_jugadas == 0: return 0.0
        return (self.victorias / self.partidas_jugadas) * 100.0

    @property
    def score_efectivo(self):
        if self.partidas_jugadas == 0: return 0.0
        return ((self.victorias + 0.5 * self.empates) / self.partidas_jugadas) * 100.0

    @property
    def tiempo_medio_turno(self):
        if self.turnos_jugados == 0: return 0.0
        return self.tiempo_total_calculo / self.turnos_jugados

    @property
    def nodos_medio_turno(self):
        if self.turnos_jugados == 0: return 0.0
        return self.nodos_totales / self.turnos_jugados

    @property
    def dif_fichas_medio(self):
        if self.partidas_jugadas == 0: return 0.0
        return self.diferencial_fichas_total / self.partidas_jugadas


def obtener_decision_agente(tipo_agente, juego, estado, turno, contexto):
    """
    Despacha el cálculo del movimiento para cualquier agente de 8x8 o 4x4.
    Retorna: (movimiento, tiempo_segundos, nodos_o_sims)
    """
    movs = juego.obtener_movimientos_legales(estado, turno)
    if movs == [None]:
        return None, 0.0, 0
        
    t0 = time.perf_counter()
    nodos = 1
    mov_elegido = None
    
    if tipo_agente == "Random":
        mov_elegido = random.choice(movs)
        nodos = 1
    elif tipo_agente == "Minimax_Universal":
        _, mov_elegido, nodos = minimax_universal(juego, estado, turno, turno)
    elif tipo_agente == "AlfaBeta_Pesos_d4":
        _, mov_elegido, nodos = minimax_alfa_beta(juego, estado, turno, turno, profundidad=4, heuristica="pesos")
    elif tipo_agente == "AlfaBeta_Hibrida_d4":
        _, mov_elegido, nodos = minimax_alfa_beta(juego, estado, turno, turno, profundidad=4, heuristica="hibrida")
    elif tipo_agente == "MCTS_Clasico":
        mcts = contexto["mcts_clasico"]
        mov_elegido, _ = mcts.mejor_movimiento(estado, turno, num_simulaciones=50)
        nodos = 50
    elif tipo_agente == "MCTS_Neuronal":
        mcts = contexto["mcts_neuronal"]
        mov_elegido, _ = mcts.mejor_movimiento(estado, turno, num_simulaciones=50)
        nodos = 50
    elif tipo_agente == "MCTS_Neuronal_100":
        mcts = contexto["mcts_neuronal"]
        mov_elegido, _ = mcts.mejor_movimiento(estado, turno, num_simulaciones=100)
        nodos = 100
    elif tipo_agente == "Agente_Tabula_Rasa_PUCT":
        mcts_puct = contexto["mcts_puct_4x4"]
        _, mov_elegido = mcts_puct.buscar(estado, turno, num_simulaciones=50, con_ruido_dirichlet=False)
        nodos = 50
    else:
        raise ValueError(f"Agente no reconocido: {tipo_agente}")
        
    t_calc = time.perf_counter() - t0
    return mov_elegido, t_calc, nodos


def jugar_partida_benchmark(juego, nombre_n, nombre_b, contexto):
    """Ejecuta una partida completa entre dos agentes registrados."""
    estado, turno = juego.estado_inicial()
    
    nod_n = 0
    nod_b = 0
    t_n = 0.0
    t_b = 0.0
    turnos_n = 0
    turnos_b = 0
    
    while not juego.es_terminal(estado):
        agente_actual = nombre_n if turno == 'N' else nombre_b
        mov, t_mov, nod_mov = obtener_decision_agente(agente_actual, juego, estado, turno, contexto)
        
        if turno == 'N':
            nod_n += nod_mov
            t_n += t_mov
            turnos_n += 1
        else:
            nod_b += nod_mov
            t_b += t_mov
            turnos_b += 1
            
        estado = juego.hacer_movimiento(estado, mov, turno)
        turno = juego.obtener_rival(turno)
        
    pn = sum(fila.count('N') for fila in estado)
    pb = sum(fila.count('B') for fila in estado)
    
    if pn > pb: ganador = 'N'
    elif pb > pn: ganador = 'B'
    else: ganador = 'E'
    
    return ganador, pn, pb, nod_n, nod_b, t_n, t_b, turnos_n, turnos_b


def ejecutar_bloque_8x8(contexto, partidas_por_cruce=20):
    """
    Ejecuta el Bloque A: Torneo de Othello 8x8 entre 5 paradigmas:
    1. AlfaBeta_Pesos_d4
    2. AlfaBeta_Hibrida_d4
    3. MCTS_Clasico (50)
    4. MCTS_Neuronal (50)
    5. MCTS_Neuronal_100 (100)
    """
    print("\n" + "=" * 85)
    print("=== BLOQUE A: EVALUACIÓN EXPERIMENTAL GLOBAL EN OTHELLO 8x8 ===")
    print("=" * 85)
    
    juego = Othello(tamano=8)
    nombres_agentes = [
        "AlfaBeta_Pesos_d4",
        "AlfaBeta_Hibrida_d4",
        "MCTS_Clasico",
        "MCTS_Neuronal",
        "MCTS_Neuronal_100"
    ]
    agentes_dict = {nom: RegistroAgente(nom) for nom in nombres_agentes}
    
    num_ag = len(nombres_agentes)
    matriz_victorias = np.full((num_ag, num_ag), np.nan)
    registros_partidas = []
    
    total_cruces = (num_ag * (num_ag - 1)) // 2
    cruce_idx = 0
    id_partida = 1
    
    for i in range(num_ag):
        for j in range(i + 1, num_ag):
            cruce_idx += 1
            ag1 = nombres_agentes[i]
            ag2 = nombres_agentes[j]
            
            mitad = partidas_por_cruce // 2
            v1 = 0
            v2 = 0
            emp = 0
            
            print(f"[{cruce_idx:02d}/{total_cruces:02d}] Cruce: {ag1} vs {ag2} ({partidas_por_cruce} partidas)...", end="", flush=True)
            t0_cruce = time.perf_counter()
            
            # Bloque 1: ag1 (Negras) vs ag2 (Blancas)
            for _ in range(mitad):
                gan, pn, pb, nn, nb, tn, tb, trn_n, trn_b = jugar_partida_benchmark(juego, ag1, ag2, contexto)
                agentes_dict[ag1].partidas_jugadas += 1
                agentes_dict[ag2].partidas_jugadas += 1
                agentes_dict[ag1].diferencial_fichas_total += (pn - pb)
                agentes_dict[ag2].diferencial_fichas_total += (pb - pn)
                agentes_dict[ag1].nodos_totales += nn
                agentes_dict[ag2].nodos_totales += nb
                agentes_dict[ag1].tiempo_total_calculo += tn
                agentes_dict[ag2].tiempo_total_calculo += tb
                agentes_dict[ag1].turnos_jugados += trn_n
                agentes_dict[ag2].turnos_jugados += trn_b
                
                if gan == 'N':
                    agentes_dict[ag1].victorias += 1
                    agentes_dict[ag2].derrotas += 1
                    v1 += 1
                elif gan == 'B':
                    agentes_dict[ag2].victorias += 1
                    agentes_dict[ag1].derrotas += 1
                    v2 += 1
                else:
                    agentes_dict[ag1].empates += 1
                    agentes_dict[ag2].empates += 1
                    emp += 1
                    
                registros_partidas.append([id_partida, ag1, ag2, pn, pb, (pn-pb), gan, tn+tb])
                id_partida += 1

            # Bloque 2: ag2 (Negras) vs ag1 (Blancas)
            for _ in range(partidas_por_cruce - mitad):
                gan, pn, pb, nn, nb, tn, tb, trn_n, trn_b = jugar_partida_benchmark(juego, ag2, ag1, contexto)
                agentes_dict[ag1].partidas_jugadas += 1
                agentes_dict[ag2].partidas_jugadas += 1
                agentes_dict[ag2].diferencial_fichas_total += (pn - pb)
                agentes_dict[ag1].diferencial_fichas_total += (pb - pn)
                agentes_dict[ag2].nodos_totales += nn
                agentes_dict[ag1].nodos_totales += nb
                agentes_dict[ag2].tiempo_total_calculo += tn
                agentes_dict[ag1].tiempo_total_calculo += tb
                agentes_dict[ag2].turnos_jugados += trn_n
                agentes_dict[ag1].turnos_jugados += trn_b
                
                if gan == 'N':
                    agentes_dict[ag2].victorias += 1
                    agentes_dict[ag1].derrotas += 1
                    v2 += 1
                elif gan == 'B':
                    agentes_dict[ag1].victorias += 1
                    agentes_dict[ag2].derrotas += 1
                    v1 += 1
                else:
                    agentes_dict[ag1].empates += 1
                    agentes_dict[ag2].empates += 1
                    emp += 1
                    
                registros_partidas.append([id_partida, ag2, ag1, pn, pb, (pn-pb), gan, tn+tb])
                id_partida += 1
                
            t_cruce = time.perf_counter() - t0_cruce
            pct1 = (v1 / partidas_por_cruce) * 100.0
            pct2 = (v2 / partidas_por_cruce) * 100.0
            matriz_victorias[i, j] = pct1
            matriz_victorias[j, i] = pct2
            print(f" -> {v1}-{emp}-{v2} ({t_cruce:.2f}s)")
            
    agentes_ordenados = sorted(agentes_dict.values(), key=lambda a: (a.score_efectivo, a.dif_fichas_medio), reverse=True)
    return agentes_ordenados, matriz_victorias, registros_partidas, nombres_agentes


def ejecutar_bloque_4x4(contexto, partidas_por_cruce=20):
    """
    Ejecuta el Bloque B: Torneo de Othello 4x4 validando Tabula Rasa frente al Minimax óptimo:
    1. Minimax_Universal (Zermelo / Óptimo)
    2. AlfaBeta_Pesos_d4
    3. Agente_Tabula_Rasa_PUCT
    4. Random
    """
    print("\n" + "=" * 85)
    print("=== BLOQUE B: EVALUACIÓN TABULA RASA (ALPHAZERO) EN OTHELLO 4x4 ===")
    print("=" * 85)
    
    juego = Othello(tamano=4)
    nombres_agentes = [
        "Minimax_Universal",
        "AlfaBeta_Pesos_d4",
        "Agente_Tabula_Rasa_PUCT",
        "Random"
    ]
    agentes_dict = {nom: RegistroAgente(nom) for nom in nombres_agentes}
    
    num_ag = len(nombres_agentes)
    matriz_victorias = np.full((num_ag, num_ag), np.nan)
    registros_partidas = []
    
    total_cruces = (num_ag * (num_ag - 1)) // 2
    cruce_idx = 0
    id_partida = 1
    
    for i in range(num_ag):
        for j in range(i + 1, num_ag):
            cruce_idx += 1
            ag1 = nombres_agentes[i]
            ag2 = nombres_agentes[j]
            
            mitad = partidas_por_cruce // 2
            v1 = 0
            v2 = 0
            emp = 0
            
            print(f"[{cruce_idx:02d}/{total_cruces:02d}] Cruce: {ag1} vs {ag2} ({partidas_por_cruce} partidas)...", end="", flush=True)
            t0_cruce = time.perf_counter()
            
            for _ in range(mitad):
                gan, pn, pb, nn, nb, tn, tb, trn_n, trn_b = jugar_partida_benchmark(juego, ag1, ag2, contexto)
                agentes_dict[ag1].partidas_jugadas += 1
                agentes_dict[ag2].partidas_jugadas += 1
                agentes_dict[ag1].diferencial_fichas_total += (pn - pb)
                agentes_dict[ag2].diferencial_fichas_total += (pb - pn)
                agentes_dict[ag1].nodos_totales += nn
                agentes_dict[ag2].nodos_totales += nb
                agentes_dict[ag1].tiempo_total_calculo += tn
                agentes_dict[ag2].tiempo_total_calculo += tb
                agentes_dict[ag1].turnos_jugados += trn_n
                agentes_dict[ag2].turnos_jugados += trn_b
                
                if gan == 'N':
                    agentes_dict[ag1].victorias += 1
                    agentes_dict[ag2].derrotas += 1
                    v1 += 1
                elif gan == 'B':
                    agentes_dict[ag2].victorias += 1
                    agentes_dict[ag1].derrotas += 1
                    v2 += 1
                else:
                    agentes_dict[ag1].empates += 1
                    agentes_dict[ag2].empates += 1
                    emp += 1
                    
                registros_partidas.append([id_partida, ag1, ag2, pn, pb, (pn-pb), gan, tn+tb])
                id_partida += 1

            for _ in range(partidas_por_cruce - mitad):
                gan, pn, pb, nn, nb, tn, tb, trn_n, trn_b = jugar_partida_benchmark(juego, ag2, ag1, contexto)
                agentes_dict[ag1].partidas_jugadas += 1
                agentes_dict[ag2].partidas_jugadas += 1
                agentes_dict[ag2].diferencial_fichas_total += (pn - pb)
                agentes_dict[ag1].diferencial_fichas_total += (pb - pn)
                agentes_dict[ag2].nodos_totales += nn
                agentes_dict[ag1].nodos_totales += nb
                agentes_dict[ag2].tiempo_total_calculo += tn
                agentes_dict[ag1].tiempo_total_calculo += tb
                agentes_dict[ag2].turnos_jugados += trn_n
                agentes_dict[ag1].turnos_jugados += trn_b
                
                if gan == 'N':
                    agentes_dict[ag2].victorias += 1
                    agentes_dict[ag1].derrotas += 1
                    v2 += 1
                elif gan == 'B':
                    agentes_dict[ag1].victorias += 1
                    agentes_dict[ag2].derrotas += 1
                    v1 += 1
                else:
                    agentes_dict[ag1].empates += 1
                    agentes_dict[ag2].empates += 1
                    emp += 1
                    
                registros_partidas.append([id_partida, ag2, ag1, pn, pb, (pn-pb), gan, tn+tb])
                id_partida += 1
                
            t_cruce = time.perf_counter() - t0_cruce
            pct1 = (v1 / partidas_por_cruce) * 100.0
            pct2 = (v2 / partidas_por_cruce) * 100.0
            matriz_victorias[i, j] = pct1
            matriz_victorias[j, i] = pct2
            print(f" -> {v1}-{emp}-{v2} ({t_cruce:.2f}s)")
            
    agentes_ordenados = sorted(agentes_dict.values(), key=lambda a: (a.score_efectivo, a.dif_fichas_medio), reverse=True)
    return agentes_ordenados, matriz_victorias, registros_partidas


def exportar_artefactos_fase4(res_8x8, mat_8x8, csv_8x8, nom_8x8, res_4x4, mat_4x4, csv_4x4):
    """Genera y exporta todas las figuras PNG (300 DPI) y tablas LaTeX (booktabs)."""
    dir_csv = os.path.join(directorio_actual, "resultados_csv")
    dir_graf = os.path.join(directorio_actual, "graficas")
    dir_tab = os.path.join(directorio_actual, "tablas")
    
    dir_cap6 = os.path.abspath(os.path.join(directorio_actual, ".."))
    dir_memoria_figuras = os.path.join(dir_cap6, "artefactos_memoria", "figuras")
    dir_memoria_tablas = os.path.join(dir_cap6, "artefactos_memoria", "tablas")
    
    for d in [dir_csv, dir_graf, dir_tab, dir_memoria_figuras, dir_memoria_tablas]:
        os.makedirs(d, exist_ok=True)
        
    # 1. Guardar CSVs
    p_csv_8 = os.path.join(dir_csv, "benchmark_global_8x8_resultados.csv")
    with open(p_csv_8, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["ID_Partida", "Negras", "Blancas", "Fichas_N", "Fichas_B", "Dif_N", "Ganador", "Tiempo_Seg"])
        for row in csv_8x8: w.writerow(row)
        
    p_csv_4 = os.path.join(dir_csv, "benchmark_global_4x4_resultados.csv")
    with open(p_csv_4, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["ID_Partida", "Negras", "Blancas", "Fichas_N", "Fichas_B", "Dif_N", "Ganador", "Tiempo_Seg"])
        for row in csv_4x4: w.writerow(row)
        
    # 2. Heatmap Global 8x8 (300 DPI)
    plt.figure(figsize=(10, 8))
    mat_plot = np.copy(mat_8x8)
    for i in range(len(nom_8x8)): mat_plot[i, i] = 50.0
    im = plt.imshow(mat_plot, cmap="Blues", vmin=0, vmax=100, aspect="auto")
    cbar = plt.colorbar(im)
    cbar.set_label("Porcentaje de Victorias (% Fila vs Columna)", fontsize=11, fontweight="bold")
    etiquetas = [n.replace("_", " ") for n in nom_8x8]
    plt.xticks(range(len(nom_8x8)), etiquetas, rotation=30, ha="right", fontsize=9.5, fontweight="bold")
    plt.yticks(range(len(nom_8x8)), etiquetas, fontsize=9.5, fontweight="bold")
    plt.title("Matriz de Enfrentamientos Globales en Othello 8x8 (% Victorias)", fontsize=12.5, fontweight="bold", pad=15)
    for i in range(len(nom_8x8)):
        for j in range(len(nom_8x8)):
            if i == j: txt, col = "-", "gray"
            else:
                val = mat_8x8[i, j]
                txt = f"{val:.1f}%"
                col = "white" if val > 65 else "black"
            plt.text(j, i, txt, ha="center", va="center", color=col, fontsize=9.5, fontweight="bold")
    plt.tight_layout()
    ruta_heatmap = os.path.join(dir_graf, "heatmap_global_8x8.png")
    plt.savefig(ruta_heatmap, dpi=300)
    plt.close()
    
    # 3. Gráfica de Latencia de Decisión (300 DPI)
    plt.figure(figsize=(10, 6))
    agentes_lat = [a for a in res_8x8]
    nombres_lat = [a.nombre.replace("_", "\n") for a in agentes_lat]
    tiempos_lat = [a.tiempo_medio_turno for a in agentes_lat]
    colores_lat = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    
    barras = plt.bar(nombres_lat, tiempos_lat, color=colores_lat[:len(nombres_lat)], edgecolor="black", alpha=0.88, width=0.55)
    plt.ylabel("Tiempo Medio de Decisión por Turno (Segundos)", fontsize=11, fontweight="bold")
    plt.title("Comparativa de Latencia de Inferencia por Jugada (Othello 8x8)", fontsize=13, fontweight="bold", pad=12)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    
    for b, t in zip(barras, tiempos_lat):
        lbl = f"{t*1000:.1f}ms" if t < 0.1 else f"{t:.3f}s"
        plt.text(b.get_x() + b.get_width()/2., b.get_height() + 0.02 * max(tiempos_lat), lbl, ha='center', va='bottom', fontsize=9.5, fontweight="bold")
        
    plt.tight_layout()
    ruta_latencia = os.path.join(dir_graf, "comparativa_latencia_decision.png")
    plt.savefig(ruta_latencia, dpi=300)
    plt.close()
    
    # 4. Tabla LaTeX 8x8 (booktabs)
    lineas_8 = [
        r"\begin{table}[htbp]",
        r"  \centering",
        r"  \caption{Comparativa Global de Rendimiento en Othello 8$\times$8 (Capítulo 6)}",
        r"  \label{tab:comparativa_global_8x8}",
        r"  \begin{tabular}{lcccc}",
        r"    \toprule",
        r"    \textbf{Paradigma / Algoritmo} & \textbf{Winrate (\%)} & \textbf{Nodos/Turno} & \textbf{Tiempo/Turno (s)} & \textbf{$\Delta$ Fichas} \\",
        r"    \midrule"
    ]
    for a in res_8x8:
        nom_t = a.nombre.replace("_", r"\_")
        sg = "+" if a.dif_fichas_medio > 0 else ""
        lineas_8.append(f"    {nom_t:<28} & {a.winrate:>6.1f}\\% & {a.nodos_medio_turno:>12.1f} & {a.tiempo_medio_turno:>14.4f} & {sg}{a.dif_fichas_medio:>6.1f} \\\\")
    lineas_8.extend([r"    \bottomrule", r"  \end{tabular}", r"\end{table}"])
    codigo_8 = "\n".join(lineas_8)
    
    ruta_tab_8 = os.path.join(dir_tab, "tabla_comparativa_global_8x8.tex")
    with open(ruta_tab_8, "w", encoding="utf-8") as f: f.write(codigo_8)
    
    # 5. Tabla LaTeX 4x4 (booktabs)
    lineas_4 = [
        r"\begin{table}[htbp]",
        r"  \centering",
        r"  \caption{Evaluación de Tabula Rasa frente al Óptimo Teórico en Othello 4$\times$4}",
        r"  \label{tab:comparativa_4x4_tabula_rasa}",
        r"  \begin{tabular}{lcccc}",
        r"    \toprule",
        r"    \textbf{Agente / Algoritmo} & \textbf{Winrate (\%)} & \textbf{Nodos/Turno} & \textbf{Tiempo/Turno (s)} & \textbf{$\Delta$ Fichas} \\",
        r"    \midrule"
    ]
    for a in res_4x4:
        nom_t = a.nombre.replace("_", r"\_")
        sg = "+" if a.dif_fichas_medio > 0 else ""
        lineas_4.append(f"    {nom_t:<26} & {a.winrate:>6.1f}\\% & {a.nodos_medio_turno:>12.1f} & {a.tiempo_medio_turno:>14.4f} & {sg}{a.dif_fichas_medio:>6.1f} \\\\")
    lineas_4.extend([r"    \bottomrule", r"  \end{tabular}", r"\end{table}"])
    codigo_4 = "\n".join(lineas_4)
    
    ruta_tab_4 = os.path.join(dir_tab, "tabla_comparativa_4x4_tabula_rasa.tex")
    with open(ruta_tab_4, "w", encoding="utf-8") as f: f.write(codigo_4)
    
    # 6. Copia centralizada a artefactos_memoria
    shutil.copy2(ruta_heatmap, os.path.join(dir_memoria_figuras, "heatmap_global_8x8.png"))
    shutil.copy2(ruta_latencia, os.path.join(dir_memoria_figuras, "comparativa_latencia_decision.png"))
    shutil.copy2(ruta_tab_8, os.path.join(dir_memoria_tablas, "tabla_comparativa_global_8x8.tex"))
    shutil.copy2(ruta_tab_4, os.path.join(dir_memoria_tablas, "tabla_comparativa_4x4_tabula_rasa.tex"))
    
    dir_con_tilde = os.path.join(directorio_ptfg, "Capítulo_6")
    if dir_con_tilde != dir_cap6:
        try: shutil.copytree(dir_cap6, dir_con_tilde, dirs_exist_ok=True)
        except Exception: pass
        
    print("\n[+] Todos los artefactos de la Fase 4 guardados y copiados con éxito a artefactos_memoria/")
    print(f"  - Heatmap: {ruta_heatmap}")
    print(f"  - Latencia: {ruta_latencia}")
    print(f"  - Tabla 8x8: {ruta_tab_8}")
    print(f"  - Tabla 4x4: {ruta_tab_4}")
    
    print("\n--- TABLA LATEX 8x8 ---")
    print(codigo_8)
    print("\n--- TABLA LATEX 4x4 ---")
    print(codigo_4)


def ejecutar_benchmark_completo(partidas_8x8=20, partidas_4x4=20):
    """Inicializa modelos y lanza los bloques A y B del benchmark experimental."""
    print("=" * 85)
    print("=== FASE 4: BENCHMARK EXPERIMENTAL GLOBAL DEL CAPÍTULO 6 ===")
    print("=" * 85)
    
    dir_cap6 = os.path.abspath(os.path.join(directorio_actual, ".."))
    
    # 1. Cargar agentes para 8x8
    juego_8 = Othello(tamano=8)
    ruta_modelo_8 = os.path.join(dir_cap6, "artefactos_memoria", "modelos", "mejor_red_valor_8x8.pth")
    if not os.path.exists(ruta_modelo_8):
        # Buscar en Fase 1
        ruta_modelo_8 = os.path.join(dir_cap6, "Fase_1_Entrenamiento_Supervisado", "modelos", "mejor_red_valor_8x8.pth")
        
    print(f"Cargando Red de Valor CNN 8x8 desde: {ruta_modelo_8}")
    mcts_neuronal_8 = MCTSNeuronal(juego=juego_8, modelo_o_ruta=ruta_modelo_8)
    mcts_clasico_8 = MCTSClasico(juego=juego_8)
    
    # 2. Cargar modelo Tabula Rasa para 4x4
    juego_4 = Othello(tamano=4)
    ruta_modelo_4 = os.path.join(dir_cap6, "artefactos_memoria", "modelos", "tabula_rasa_4x4.pth")
    if not os.path.exists(ruta_modelo_4):
        ruta_modelo_4 = os.path.join(dir_cap6, "Fase_3_Tabula_Rasa_4x4", "modelos", "tabula_rasa_4x4.pth")
        
    print(f"Cargando Red Dual 4x4 desde: {ruta_modelo_4}")
    disp = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modelo_4 = RedDual4x4().to(disp)
    if os.path.exists(ruta_modelo_4):
        st = torch.load(ruta_modelo_4, map_location=disp, weights_only=True)
        modelo_4.load_state_dict(st)
    modelo_4.eval()
    mcts_puct_4 = MCTSPUCT(juego=juego_4, modelo=modelo_4, dispositivo=disp)
    
    contexto = {
        "mcts_clasico": mcts_clasico_8,
        "mcts_neuronal": mcts_neuronal_8,
        "mcts_puct_4x4": mcts_puct_4
    }
    
    # Ejecutar Bloque A (8x8)
    res_8x8, mat_8x8, csv_8x8, nom_8x8 = ejecutar_bloque_8x8(contexto, partidas_por_cruce=partidas_8x8)
    
    # Ejecutar Bloque B (4x4)
    res_4x4, mat_4x4, csv_4x4 = ejecutar_bloque_4x4(contexto, partidas_por_cruce=partidas_4x4)
    
    # Exportar todos los artefactos para LaTeX y memoria
    exportar_artefactos_fase4(res_8x8, mat_8x8, csv_8x8, nom_8x8, res_4x4, mat_4x4, csv_4x4)

if __name__ == "__main__":
    ejecutar_benchmark_completo(partidas_8x8=20, partidas_4x4=20)
