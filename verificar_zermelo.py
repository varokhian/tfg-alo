import os
import sys
import time
import csv

# Añadir la raíz del proyecto al sys.path para importar módulos principales
directorio_script = os.path.dirname(os.path.abspath(__file__))
directorio_ptfg = os.path.abspath(os.path.join(directorio_script, "..", "..", ".."))
if directorio_ptfg not in sys.path:
    sys.path.insert(0, directorio_ptfg)

from juegos import TresEnRaya
from minimax import minimax_universal

def generar_tabla_latex_zermelo(resultados):
    """
    Genera el código LaTeX con el paquete booktabs para las 10 partidas.
    Columnas: [ID Partida, Resultado Final, Nodos evaluados en el Turno 1, Tiempo total de la partida (s)].
    """
    lineas = [
        r"\begin{table}[htbp]",
        r"  \centering",
        r"  \caption{Verificación Empírica del Teorema de Zermelo en Tres en Raya (\textit{Minimax Universal vs Minimax Universal})}",
        r"  \label{tab:verificacion_zermelo}",
        r"  \begin{tabular}{cccc}",
        r"    \toprule",
        r"    \textbf{ID Partida} & \textbf{Resultado Final} & \textbf{Nodos evaluados en Turno 1} & \textbf{Tiempo total de partida (s)} \\",
        r"    \midrule"
    ]
    for r in resultados:
        lineas.append(
            f"    {r['id_partida']:^10} & {r['resultado']:^18} & {r['nodos_turno_1']:>24,d} & {r['tiempo_total']:>25.4f} \\\\"
        )
    lineas.extend([
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}"
    ])
    return "\n".join(lineas)

def ejecutar_verificacion_zermelo(num_partidas=10):
    """
    Ejecuta partidas autónomas de Minimax Universal vs Minimax Universal en Tres en Raya,
    contando los nodos evaluados en cada turno y verificando el 100% de empates (Teorema de Zermelo).
    """
    print("=" * 85)
    print("=== CAPÍTULO 5: VERIFICACIÓN EMPÍRICA DEL TEOREMA DE ZERMELO (TRES EN RAYA) ===")
    print("=" * 85)
    print("Teorema de Zermelo (1913): En todo juego finito bipersonal de información")
    print("perfecta y suma cero sin azar, existe una estrategia óptima que garantiza")
    print("la victoria a uno de los jugadores o el empate forzado a ambos.")
    print(f"\nEjecutando {num_partidas} partidas autónomas de Minimax Universal vs Minimax Universal...")
    print("-" * 85)
    
    juego = TresEnRaya()
    resultados_partidas = []
    victorias_max = 0
    victorias_min = 0
    empates = 0
    
    tiempo_inicio_global = time.perf_counter()
    
    for id_partida in range(1, num_partidas + 1):
        t_inicio_partida = time.perf_counter()
        estado, turno_actual = juego.estado_inicial()
        jugador_inicial = turno_actual # 'X'
        numero_turno = 1
        nodos_turno_1 = 0
        
        while not juego.es_terminal(estado):
            # Inferencia Minimax Universal determinista pura contando los nodos evaluados
            _, movimiento_elegido, nodos_evaluados = minimax_universal(
                juego, estado, turno_actual, turno_actual
            )
            
            if numero_turno == 1:
                nodos_turno_1 = nodos_evaluados
                
            estado = juego.hacer_movimiento(estado, movimiento_elegido, turno_actual)
            turno_actual = juego.obtener_rival(turno_actual)
            numero_turno += 1
            
        t_total_partida = time.perf_counter() - t_inicio_partida
        utilidad = juego.obtener_utilidad(estado, jugador_inicial)
        
        if utilidad == 1:
            resultado_texto = "Victoria MAX (X)"
            victorias_max += 1
        elif utilidad == -1:
            resultado_texto = "Victoria MIN (O)"
            victorias_min += 1
        else:
            resultado_texto = "Empate (Tablas)"
            empates += 1
            
        resultados_partidas.append({
            "id_partida": id_partida,
            "resultado": resultado_texto,
            "nodos_turno_1": nodos_turno_1,
            "tiempo_total": t_total_partida
        })
        
        print(f"Partida {id_partida:02d}/{num_partidas:02d} | Resultado: {resultado_texto:<16} | Nodos Turno 1: {nodos_turno_1:>7,d} | Tiempo: {t_total_partida:.4f}s")
        
    tiempo_total_global = time.perf_counter() - tiempo_inicio_global
    
    print("-" * 85)
    print("\n--- RESUMEN GLOBAL ---")
    print(f"Total Partidas:    {num_partidas}")
    print(f"Victorias MAX (X): {victorias_max} ({victorias_max/num_partidas*100:.1f}%)")
    print(f"Victorias MIN (O): {victorias_min} ({victorias_min/num_partidas*100:.1f}%)")
    print(f"Empates (Tablas):  {empates} ({empates/num_partidas*100:.1f}%)")
    print(f"Tiempo Total:      {tiempo_total_global:.2f} segundos")
    
    if empates == num_partidas and victorias_max == 0 and victorias_min == 0:
        print("\n[OK] VERIFICACION EXITOSA: El 100% de las partidas acabaron en empate (0 victorias MAX, 0 victorias MIN, 10 empates).")
        print("     Queda empiricamente demostrado el Teorema de Zermelo para el juego de Tres en Raya.")
    else:
        print("\n[!] ALERTA: Se detectaron resultados distintos al empate forzado.")
        
    # Generar tabla LaTeX con booktabs
    codigo_latex = generar_tabla_latex_zermelo(resultados_partidas)
    print("\n" + "=" * 85)
    print("=== CÓDIGO LATEX (BOOKTABS) PARA LA MEMORIA ===")
    print("=" * 85)
    print(codigo_latex)
    
    # Guardar en la carpeta designada: pruebas/Pruebas_Capitulo_5/pruebas_3_en_raya/
    ruta_archivo_tex = os.path.join(directorio_script, "tabla_zermelo_tres_en_raya.tex")
    with open(ruta_archivo_tex, "w", encoding="utf-8") as f:
        f.write(codigo_latex)
    print(f"\n[+] Tabla LaTeX guardada en: {ruta_archivo_tex}")
    
    ruta_archivo_csv = os.path.join(directorio_script, "resultados_zermelo.csv")
    with open(ruta_archivo_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["ID_Partida", "Resultado_Final", "Nodos_Evaluados_Turno_1", "Tiempo_Total_Segundos"])
        for r in resultados_partidas:
            writer.writerow([r["id_partida"], r["resultado"], r["nodos_turno_1"], round(r["tiempo_total"], 4)])
    print(f"[+] Archivo CSV guardado en: {ruta_archivo_csv}")
    print("=" * 85)

if __name__ == "__main__":
    ejecutar_verificacion_zermelo(num_partidas=10)
