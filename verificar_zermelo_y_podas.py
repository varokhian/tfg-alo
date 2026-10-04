import os
import sys
import time
import csv
import numpy as np

# Añadir la raíz del proyecto al sys.path
directorio_script = os.path.dirname(os.path.abspath(__file__))
directorio_ptfg = os.path.abspath(os.path.join(directorio_script, "..", "..", "..", ".."))
if directorio_ptfg not in sys.path:
    sys.path.insert(0, directorio_ptfg)

from juegos import TresEnRaya
from minimax import minimax_universal, minimax_alfa_beta

def ejecutar_serie_partidas(juego, tipo_algoritmo, num_partidas=10):
    """
    Ejecuta num_partidas autónomas en Tres en Raya midiendo:
      - Tasa de empates (tablas).
      - Nodos evaluados en el primer turno (T1).
      - Nodos totales evaluados en toda la partida.
      - Tiempo total de cada partida.
    """
    partidas_info = []
    empates = 0
    victorias_x = 0
    victorias_o = 0
    
    for idx in range(1, num_partidas + 1):
        t_inicio = time.perf_counter()
        estado, turno_actual = juego.estado_inicial()
        jugador_inicial = turno_actual # 'X'
        numero_turno = 1
        nodos_t1 = 0
        nodos_totales_partida = 0
        
        while not juego.es_terminal(estado):
            if tipo_algoritmo == "universal":
                _, mov, nodos_mov = minimax_universal(juego, estado, turno_actual, turno_actual)
            elif tipo_algoritmo == "alfa_beta":
                _, mov, nodos_mov = minimax_alfa_beta(juego, estado, turno_actual, turno_actual, profundidad=9)
            else:
                raise ValueError(f"Algoritmo no soportado: {tipo_algoritmo}")
                
            if numero_turno == 1:
                nodos_t1 = nodos_mov
            nodos_totales_partida += nodos_mov
            
            estado = juego.hacer_movimiento(estado, mov, turno_actual)
            turno_actual = juego.obtener_rival(turno_actual)
            numero_turno += 1
            
        t_partida = time.perf_counter() - t_inicio
        utilidad = juego.obtener_utilidad(estado, jugador_inicial)
        
        if utilidad == 0:
            empates += 1
            res_str = "Empate"
        elif utilidad == 1:
            victorias_x += 1
            res_str = "Victoria X"
        else:
            victorias_o += 1
            res_str = "Victoria O"
            
        partidas_info.append({
            "partida": idx,
            "resultado": res_str,
            "nodos_t1": nodos_t1,
            "nodos_totales": nodos_totales_partida,
            "tiempo": t_partida
        })
        
    return {
        "tipo": tipo_algoritmo,
        "partidas": partidas_info,
        "total_partidas": num_partidas,
        "empates": empates,
        "tasa_tablas": (empates / num_partidas) * 100.0,
        "nodos_t1_medio": float(np.mean([p["nodos_t1"] for p in partidas_info])),
        "nodos_totales_medio": float(np.mean([p["nodos_totales"] for p in partidas_info])),
        "tiempo_medio": float(np.mean([p["tiempo"] for p in partidas_info])),
        "tiempo_std": float(np.std([p["tiempo"] for p in partidas_info], ddof=1)) if num_partidas > 1 else 0.0
    }

def generar_tabla_latex_resumen(res_universal, res_alfa_beta):
    """
    Construye la tabla LaTeX vertical y compacta para el resumen de Zermelo y poda alfa-beta.
    """
    # Cálculos de reducción porcentual
    red_t1 = ((res_universal["nodos_t1_medio"] - res_alfa_beta["nodos_t1_medio"]) / res_universal["nodos_t1_medio"]) * 100.0
    red_tot = ((res_universal["nodos_totales_medio"] - res_alfa_beta["nodos_totales_medio"]) / res_universal["nodos_totales_medio"]) * 100.0
    red_tiempo = ((res_universal["tiempo_medio"] - res_alfa_beta["tiempo_medio"]) / res_universal["tiempo_medio"]) * 100.0
    
    nodos_u_t1_str = f"{int(res_universal['nodos_t1_medio']):,d}"
    nodos_u_tot_str = f"{int(res_universal['nodos_totales_medio']):,d}"
    tiempo_u_str = f"{res_universal['tiempo_medio']:.3f} $\\pm$ {res_universal['tiempo_std']:.3f}"
    
    nodos_ab_t1_str = f"{int(res_alfa_beta['nodos_t1_medio']):,d}"
    nodos_ab_tot_str = f"{int(res_alfa_beta['nodos_totales_medio']):,d}"
    tiempo_ab_str = f"{res_alfa_beta['tiempo_medio']:.3f} $\\pm$ {res_alfa_beta['tiempo_std']:.3f}"
    
    lineas = [
        r"\begin{table}[htbp]",
        r"  \centering",
        r"  \caption{Verificación de Zermelo y Eficiencia de Poda en Tres en Raya ($N=10$)}",
        r"  \label{tab:resumen_zermelo}",
        r"  \begin{tabular}{lcccc}",
        r"    \toprule",
        r"    \textbf{Algoritmo} & \textbf{Tablas} & \textbf{Nodos T1} & \textbf{Nodos Total} & \textbf{Tiempo (s)} \\",
        r"    \midrule",
        f"    Minimax Universal & {res_universal['tasa_tablas']:.0f}\\% & {nodos_u_t1_str} & {nodos_u_tot_str} & {tiempo_u_str} \\\\",
        f"    Poda Alfa-Beta    & {res_alfa_beta['tasa_tablas']:.0f}\\% & {nodos_ab_t1_str} & {nodos_ab_tot_str} & {tiempo_ab_str} \\\\",
        r"    \midrule",
        f"    \\textbf{{Reducción (\\%)}} & --- & \\textbf{{{red_t1:.1f}\\%}} & \\textbf{{{red_tot:.1f}\\%}} & \\textbf{{{red_tiempo:.1f}\\%}} \\\\",
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}"
    ]
    return "\n".join(lineas), red_t1, red_tot, red_tiempo

def ejecutar_comparativa_completa(num_partidas=10):
    print("=" * 85)
    print("=== CAPÍTULO 5: COMPARATIVA MINIMAX UNIVERSAL VS PODA ALFA-BETA (ZERMELO) ===")
    print("=" * 85)
    print(f"Juego: Tres en Raya (Partidas autónomas por algoritmo: {num_partidas})")
    print("Objetivo: Medir tasa de empates teóricos y cuantificar la reducción de poda Alfa-Beta.")
    print("-" * 85)
    
    juego = TresEnRaya()
    
    # 1. Serie Minimax Universal
    print(f"\n[1/2] Ejecutando {num_partidas} partidas de Minimax Universal vs Minimax Universal...")
    t0_uni = time.perf_counter()
    res_universal = ejecutar_serie_partidas(juego, "universal", num_partidas)
    t_tot_uni = time.perf_counter() - t0_uni
    print(f"      -> Completado en {t_tot_uni:.2f}s | Tablas: {res_universal['empates']}/{num_partidas} ({res_universal['tasa_tablas']:.0f}%)")
    print(f"      -> Nodos Turno 1: {int(res_universal['nodos_t1_medio']):,d} | Nodos Total Partida: {int(res_universal['nodos_totales_medio']):,d} | Tiempo Medio: {res_universal['tiempo_medio']:.4f}s")
    
    # 2. Serie Poda Alfa-Beta
    print(f"\n[2/2] Ejecutando {num_partidas} partidas de Poda Alfa-Beta vs Poda Alfa-Beta (Prof=9)...")
    t0_ab = time.perf_counter()
    res_alfa_beta = ejecutar_serie_partidas(juego, "alfa_beta", num_partidas)
    t_tot_ab = time.perf_counter() - t0_ab
    print(f"      -> Completado en {t_tot_ab:.2f}s | Tablas: {res_alfa_beta['empates']}/{num_partidas} ({res_alfa_beta['tasa_tablas']:.0f}%)")
    print(f"      -> Nodos Turno 1: {int(res_alfa_beta['nodos_t1_medio']):,d} | Nodos Total Partida: {int(res_alfa_beta['nodos_totales_medio']):,d} | Tiempo Medio: {res_alfa_beta['tiempo_medio']:.4f}s")
    
    # 3. Generar tabla LaTeX compacta
    codigo_latex, red_t1, red_tot, red_tiempo = generar_tabla_latex_resumen(res_universal, res_alfa_beta)
    
    # 4. Mostrar tabla formateada en texto plano
    print("\n" + "=" * 85)
    print(f"{'TABLA RESUMEN COMPARATIVA (TEXTO PLANO)':^85}")
    print("=" * 85)
    print(f"{'Algoritmo':<20} | {'Tablas (%)':<12} | {'Nodos T1':<14} | {'Nodos Total':<14} | {'Tiempo (s)':<18}")
    print("-" * 85)
    u_tiempo_txt = f"{res_universal['tiempo_medio']:.3f} ± {res_universal['tiempo_std']:.3f}"
    ab_tiempo_txt = f"{res_alfa_beta['tiempo_medio']:.3f} ± {res_alfa_beta['tiempo_std']:.3f}"
    print(f"{'Minimax Universal':<20} | {res_universal['tasa_tablas']:<12.0f}% | {int(res_universal['nodos_t1_medio']):<14,d} | {int(res_universal['nodos_totales_medio']):<14,d} | {u_tiempo_txt:<18}")
    print(f"{'Poda Alfa-Beta':<20} | {res_alfa_beta['tasa_tablas']:<12.0f}% | {int(res_alfa_beta['nodos_t1_medio']):<14,d} | {int(res_alfa_beta['nodos_totales_medio']):<14,d} | {ab_tiempo_txt:<18}")
    print("-" * 85)
    print(f"{'Reducción (%)':<20} | {'---':<12} | {red_t1:<13.1f}% | {red_tot:<13.1f}% | {red_tiempo:<17.1f}%")
    print("=" * 85)
    
    # 5. Mostrar código LaTeX
    print("\n" + "=" * 85)
    print("=== CÓDIGO LATEX LISTO PARA COMPILAR (tabla_zermelo_resumen.tex) ===")
    print("=" * 85)
    print(codigo_latex)
    print("=" * 85)
    
    # 6. Guardar archivos en pruebas/Pruebas_Capitulo_5/pruebas_3_en_raya/comparativa/
    ruta_tex = os.path.join(directorio_script, "tabla_zermelo_resumen.tex")
    with open(ruta_tex, "w", encoding="utf-8") as f:
        f.write(codigo_latex)
    print(f"\n[+] Tabla LaTeX guardada en: {ruta_tex}")
    
    ruta_csv = os.path.join(directorio_script, "comparativa_zermelo_podas.csv")
    with open(ruta_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Algoritmo", "Tasa_Tablas_Pct", "Nodos_T1_Medio", "Nodos_Totales_Medio", "Tiempo_Medio_Seg", "Tiempo_Std_Seg"])
        writer.writerow(["Minimax Universal", res_universal["tasa_tablas"], res_universal["nodos_t1_medio"], res_universal["nodos_totales_medio"], res_universal["tiempo_medio"], res_universal["tiempo_std"]])
        writer.writerow(["Poda Alfa-Beta", res_alfa_beta["tasa_tablas"], res_alfa_beta["nodos_t1_medio"], res_alfa_beta["nodos_totales_medio"], res_alfa_beta["tiempo_medio"], res_alfa_beta["tiempo_std"]])
        writer.writerow(["Reduccion_Porcentual", "---", round(red_t1, 2), round(red_tot, 2), round(red_tiempo, 2), "---"])
    print(f"[+] Archivo CSV guardado en: {ruta_csv}")
    print("=" * 85 + "\n")

if __name__ == "__main__":
    ejecutar_comparativa_completa(num_partidas=10)
