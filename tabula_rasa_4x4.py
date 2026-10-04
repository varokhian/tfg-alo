import os
import sys
import copy
import math
import time
import random
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Añadir la raíz del proyecto al sys.path
directorio_actual = os.path.dirname(os.path.abspath(__file__))
directorio_ptfg = os.path.abspath(os.path.join(directorio_actual, "..", ".."))
if directorio_ptfg not in sys.path:
    sys.path.insert(0, directorio_ptfg)

from juegos import Othello

# ==============================================================================
# 1. ARQUITECTURA RED DUAL ACTOR-CRÍTICO (4x4)
# ==============================================================================
class RedDual4x4(nn.Module):
    """
    Red Neuronal Dual (Actor-Crítico) para Othello 4x4.
    Entrada: (batch_size, 2, 4, 4)
      - Canal 0: Fichas del jugador activo.
      - Canal 1: Fichas del rival.
    Salidas:
      - Política (Actor): Log-probabilidades sobre las 16 casillas (batch_size, 16).
      - Valor (Crítico): Escalar acotado en [-1.0, 1.0] mediante Tanh (batch_size, 1).
    """
    def __init__(self):
        super(RedDual4x4, self).__init__()
        
        # Tronco convolucional común
        self.conv_trunk = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        # Cabeza de Política (Actor)
        self.policy_head = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=2, kernel_size=1),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * 4 * 4, 16),
            nn.LogSoftmax(dim=1)
        )
        
        # Cabeza de Valor (Crítico)
        self.value_head = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(1 * 4 * 4, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Tanh()
        )
        
    def forward(self, x):
        trunk = self.conv_trunk(x)
        log_pi = self.policy_head(trunk)
        v = self.value_head(trunk)
        return log_pi, v


# ==============================================================================
# 2. MCTS CON SELECCIÓN PUCT Y RUIDO DE DIRICHLET
# ==============================================================================
class NodoPUCT:
    """Nodo del árbol MCTS que implementa el algoritmo de selección PUCT (AlphaZero)."""
    def __init__(self, estado, jugador_actual, movimiento_origen=None, padre=None, prob_prior=1.0):
        self.estado = estado
        self.jugador_actual = jugador_actual
        self.movimiento_origen = movimiento_origen
        self.padre = padre
        self.prob_prior = prob_prior # P(s, a)
        
        self.hijos = {} # {movimiento: NodoPUCT}
        self.visitas = 0           # N(s, a)
        self.valor_acumulado = 0.0 # W(s, a)
        self.valor_promedio = 0.0  # Q(s, a) = W / N

    def puct(self, c_puct=1.5, total_visitas_padre=1):
        """Calcula el valor PUCT = Q(s, a) + U(s, a)"""
        u = c_puct * self.prob_prior * (math.sqrt(total_visitas_padre) / (1.0 + self.visitas))
        return self.valor_promedio + u


class MCTSPUCT:
    """Motor de búsqueda MCTS guiado por la Red Dual (Actor-Crítico)."""
    def __init__(self, juego, modelo, dispositivo=None, c_puct=1.5):
        self.juego = juego
        self.modelo = modelo
        self.c_puct = c_puct
        self.dispositivo = dispositivo if dispositivo else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.modelo.to(self.dispositivo)

    def estado_a_tensor(self, estado, jugador):
        """Convierte matriz 4x4 a tensor canónico (1, 2, 4, 4)."""
        rival = self.juego.obtener_rival(jugador)
        c0 = np.zeros((4, 4), dtype=np.float32)
        c1 = np.zeros((4, 4), dtype=np.float32)
        for r in range(4):
            for c in range(4):
                if estado[r][c] == jugador:
                    c0[r, c] = 1.0
                elif estado[r][c] == rival:
                    c1[r, c] = 1.0
        tensor = np.stack([c0, c1], axis=0)
        return torch.from_numpy(tensor).unsqueeze(0).to(self.dispositivo)

    def evaluar_estado_red(self, estado, jugador):
        """Obtiene priors de política y valor predicho desde la red dual."""
        tensor = self.estado_a_tensor(estado, jugador)
        self.modelo.eval()
        with torch.no_grad():
            log_pi, v = self.modelo(tensor)
            pi = torch.exp(log_pi).squeeze(0).cpu().numpy() # vector de 16 probabilidades
            val = v.item()
        return pi, val

    def buscar(self, estado_raiz, jugador_raiz, num_simulaciones=50, con_ruido_dirichlet=False, alpha_dir=0.3, eps_dir=0.25):
        """
        Ejecuta num_simulaciones búsquedas PUCT desde estado_raiz.
        Retorna: vector de política pi_mcts (16,) y el mejor movimiento.
        """
        raiz = NodoPUCT(estado_raiz, jugador_raiz)
        movs_legales = self.juego.obtener_movimientos_legales(estado_raiz, jugador_raiz)
        
        if movs_legales == [None]:
            # Pase obligatorio
            return np.zeros(16, dtype=np.float32), None
            
        # Expansión inicial de la raíz con la red dual
        priors_raw, _ = self.evaluar_estado_red(estado_raiz, jugador_raiz)
        
        # Filtrar y normalizar priors sobre movimientos legales
        priors_legales = {}
        suma_priors = 0.0
        for mov in movs_legales:
            idx = mov[0] * 4 + mov[1]
            p = priors_raw[idx]
            priors_legales[mov] = p
            suma_priors += p
            
        if suma_priors > 0:
            for mov in priors_legales:
                priors_legales[mov] /= suma_priors
        else:
            uniforme = 1.0 / len(movs_legales)
            for mov in priors_legales:
                priors_legales[mov] = uniforme
                
        # Ruido de Dirichlet en la raíz para auto-juego estocástico
        if con_ruido_dirichlet:
            ruido = np.random.dirichlet([alpha_dir] * len(movs_legales))
            for i, mov in enumerate(movs_legales):
                priors_legales[mov] = (1.0 - eps_dir) * priors_legales[mov] + eps_dir * ruido[i]
                
        for mov, p in priors_legales.items():
            nuevo_est = self.juego.hacer_movimiento(estado_raiz, mov, jugador_raiz)
            sig_jug = self.juego.obtener_rival(jugador_raiz)
            raiz.hijos[mov] = NodoPUCT(nuevo_est, sig_jug, movimiento_origen=mov, padre=raiz, prob_prior=p)
            
        # Simulaciones MCTS-PUCT
        for _ in range(num_simulaciones):
            nodo = raiz
            
            # 1. Selección descendente por PUCT
            while len(nodo.hijos) > 0 and not self.juego.es_terminal(nodo.estado):
                total_visitas = max(1, nodo.visitas)
                nodo = max(nodo.hijos.values(), key=lambda h: h.puct(self.c_puct, total_visitas))
                
            # 2. Evaluación
            if self.juego.es_terminal(nodo.estado):
                # Utilidad exacta terminal (+1, -1, 0)
                v = float(self.juego.obtener_utilidad(nodo.estado, nodo.jugador_actual))
            else:
                # Expansión del nodo hoja con inferencia de la red
                pi_hoja, v = self.evaluar_estado_red(nodo.estado, nodo.jugador_actual)
                movs_hoja = self.juego.obtener_movimientos_legales(nodo.estado, nodo.jugador_actual)
                
                if movs_hoja != [None]:
                    suma_p = sum(pi_hoja[m[0]*4 + m[1]] for m in movs_hoja)
                    for m in movs_hoja:
                        idx_m = m[0] * 4 + m[1]
                        prob_m = (pi_hoja[idx_m] / suma_p) if suma_p > 0 else (1.0 / len(movs_hoja))
                        est_hijo = self.juego.hacer_movimiento(nodo.estado, m, nodo.jugador_actual)
                        jug_hijo = self.juego.obtener_rival(nodo.jugador_actual)
                        nodo.hijos[m] = NodoPUCT(est_hijo, jug_hijo, movimiento_origen=m, padre=nodo, prob_prior=prob_m)
                        
            # 3. Retropropagación
            jug_eval = nodo.jugador_actual
            actual = nodo
            while actual is not None:
                actual.visitas += 1
                if actual.jugador_actual == jug_eval:
                    actual.valor_acumulado += v
                else:
                    actual.valor_acumulado -= v
                actual.valor_promedio = actual.valor_acumulado / actual.visitas
                actual = actual.padre
                
        # Construir vector de política resultante (visitas normalizadas)
        pi_vector = np.zeros(16, dtype=np.float32)
        total_v = sum(h.visitas for h in raiz.hijos.values())
        
        mejor_mov = None
        max_visitas = -1
        
        for mov, h in raiz.hijos.items():
            idx = mov[0] * 4 + mov[1]
            if total_v > 0:
                pi_vector[idx] = h.visitas / total_v
            if h.visitas > max_visitas:
                max_visitas = h.visitas
                mejor_mov = mov
                
        return pi_vector, mejor_mov


# ==============================================================================
# 3. AUTO-JUEGO Y ENTRENAMIENTO POR REFUERZO (TABULA RASA)
# ==============================================================================
def jugar_partida_selfplay(juego, mcts, num_sims=50):
    """
    Ejecuta una partida completa de auto-juego generando ejemplos (estado, pi, z).
    """
    estado, turno = juego.estado_inicial()
    ejemplos_partida = [] # [(estado, turno, pi_vector)]
    turnos_jugados = 0
    
    while not juego.es_terminal(estado):
        movs = juego.obtener_movimientos_legales(estado, turno)
        if movs == [None]:
            mov_elegido = None
        else:
            # Añadir ruido de Dirichlet en los primeros turnos para máxima exploración
            con_dirichlet = (turnos_jugados < 4)
            pi_vector, mejor_mov = mcts.buscar(
                estado, turno, num_simulaciones=num_sims,
                con_ruido_dirichlet=con_dirichlet
            )
            
            ejemplos_partida.append((copy.deepcopy(estado), turno, pi_vector))
            
            # En aperturas muestrear según la distribución; luego explotar argmax
            if turnos_jugados < 2:
                # Muestreo estocástico proporcional
                movs_lista = list(movs)
                probs = np.array([pi_vector[m[0]*4 + m[1]] for m in movs_lista], dtype=np.float64)
                suma_pr = np.sum(probs)
                if suma_pr > 0:
                    probs /= suma_pr
                    idx_elegido = np.random.choice(len(movs_lista), p=probs)
                    mov_elegido = movs_lista[idx_elegido]
                else:
                    mov_elegido = random.choice(movs_lista)
            else:
                mov_elegido = mejor_mov
                
        estado = juego.hacer_movimiento(estado, mov_elegido, turno)
        turno = juego.obtener_rival(turno)
        turnos_jugados += 1
        
    # Recompensa terminal
    utilidad_n = juego.obtener_utilidad(estado, 'N')
    datos_entrenamiento = []
    
    for est, jugador_step, pi_step in ejemplos_partida:
        if utilidad_n == 0:
            z = 0.0
        elif utilidad_n == 1:
            z = 1.0 if jugador_step == 'N' else -1.0
        else:
            z = -1.0 if jugador_step == 'N' else 1.0
            
        datos_entrenamiento.append((est, jugador_step, pi_step, z))
        
    return datos_entrenamiento


def evaluar_contra_random(juego, mcts, num_partidas=10):
    """Evalúa la fuerza táctica actual frente a un agente aleatorio."""
    victorias = 0
    empates = 0
    derrotas = 0
    
    mitad = num_partidas // 2
    
    # Bloque 1: MCTS (Negras) vs Random (Blancas)
    for _ in range(mitad):
        estado, turno = juego.estado_inicial()
        while not juego.es_terminal(estado):
            movs = juego.obtener_movimientos_legales(estado, turno)
            if movs == [None]:
                mov = None
            elif turno == 'N':
                _, mov = mcts.buscar(estado, turno, num_simulaciones=30, con_ruido_dirichlet=False)
            else:
                mov = random.choice(movs)
            estado = juego.hacer_movimiento(estado, mov, turno)
            turno = juego.obtener_rival(turno)
        u = juego.obtener_utilidad(estado, 'N')
        if u == 1: victorias += 1
        elif u == 0: empates += 1
        else: derrotas += 1

    # Bloque 2: Random (Negras) vs MCTS (Blancas)
    for _ in range(num_partidas - mitad):
        estado, turno = juego.estado_inicial()
        while not juego.es_terminal(estado):
            movs = juego.obtener_movimientos_legales(estado, turno)
            if movs == [None]:
                mov = None
            elif turno == 'B':
                _, mov = mcts.buscar(estado, turno, num_simulaciones=30, con_ruido_dirichlet=False)
            else:
                mov = random.choice(movs)
            estado = juego.hacer_movimiento(estado, mov, turno)
            turno = juego.obtener_rival(turno)
        u = juego.obtener_utilidad(estado, 'B')
        if u == 1: victorias += 1
        elif u == 0: empates += 1
        else: derrotas += 1
        
    winrate = ((victorias + 0.5 * empates) / num_partidas) * 100.0
    return winrate


def ejecutar_tabula_rasa(
    num_iteraciones=15,
    partidas_por_iteracion=20,
    num_simulaciones_mcts=50,
    lr=0.001,
    weight_decay=1e-4,
    batch_size=32,
    epocas_por_iter=4
):
    """
    Bucle principal de auto-juego y aprendizaje por refuerzo Tabula Rasa (AlphaZero en Othello 4x4).
    """
    juego = Othello(tamano=4)
    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modelo = RedDual4x4().to(dispositivo)
    optimizador = torch.optim.Adam(modelo.parameters(), lr=lr, weight_decay=weight_decay)
    mcts = MCTSPUCT(juego=juego, modelo=modelo, dispositivo=dispositivo, c_puct=1.5)
    
    # Rutas de guardado
    dir_modelos = os.path.join(directorio_actual, "modelos")
    dir_graficas = os.path.join(directorio_actual, "graficas")
    dir_tablas = os.path.join(directorio_actual, "tablas")
    
    dir_cap6 = os.path.abspath(os.path.join(directorio_actual, ".."))
    dir_memoria_modelos = os.path.join(dir_cap6, "artefactos_memoria", "modelos")
    dir_memoria_figuras = os.path.join(dir_cap6, "artefactos_memoria", "figuras")
    dir_memoria_tablas = os.path.join(dir_cap6, "artefactos_memoria", "tablas")
    
    for d in [dir_modelos, dir_graficas, dir_tablas, dir_memoria_modelos, dir_memoria_figuras, dir_memoria_tablas]:
        os.makedirs(d, exist_ok=True)
        
    ruta_modelo = os.path.join(dir_modelos, "tabula_rasa_4x4.pth")
    ruta_grafica = os.path.join(dir_graficas, "curva_perdida_tabula_rasa_4x4.png")
    ruta_tabla = os.path.join(dir_tablas, "tabla_progreso_selfplay.tex")
    
    print("=" * 85)
    print("=== FASE 3: APRENDIZAJE POR REFUERZO TABULA RASA (ALPHAZERO EN OTHELLO 4x4) ===")
    print("=" * 85)
    print(f"Dispositivo: {dispositivo}")
    print(f"Iteraciones: {num_iteraciones} | Partidas auto-juego por iteración: {partidas_por_iteracion}")
    print(f"Simulaciones MCTS-PUCT por turno: {num_simulaciones_mcts}")
    print("-" * 85)
    
    historial_loss_total = []
    historial_loss_valor = []
    historial_loss_politica = []
    historial_winrate_vs_random = []
    historial_tiempos = []
    
    buffer_entrenamiento = []
    max_buffer = 1500
    
    print(f"{'Iter':<6} | {'Loss Total':<12} | {'Loss Valor':<12} | {'Loss Pol':<12} | {'Win vs Rand':<13} | {'Tiempo (s)':<10}")
    print("-" * 75)
    
    t_inicio_total = time.perf_counter()
    
    for iter_idx in range(1, num_iteraciones + 1):
        t_inicio_iter = time.perf_counter()
        
        # 1. Generación de datos mediante Auto-Juego
        modelo.eval()
        nuevos_ejemplos = []
        for _ in range(partidas_por_iteracion):
            partida_datos = jugar_partida_selfplay(juego, mcts, num_sims=num_simulaciones_mcts)
            nuevos_ejemplos.extend(partida_datos)
            
        buffer_entrenamiento.extend(nuevos_ejemplos)
        if len(buffer_entrenamiento) > max_buffer:
            buffer_entrenamiento = buffer_entrenamiento[-max_buffer:]
            
        # 2. Entrenamiento de la Red Dual
        modelo.train()
        suma_loss_tot = 0.0
        suma_loss_val = 0.0
        suma_loss_pol = 0.0
        num_pasos = 0
        
        # Preparar lotes aleatorios
        random.shuffle(buffer_entrenamiento)
        for _ in range(epocas_por_iter):
            for b_idx in range(0, len(buffer_entrenamiento), batch_size):
                lote = buffer_entrenamiento[b_idx:b_idx+batch_size]
                if len(lote) < 4:
                    continue
                    
                # Convertir a tensores
                x_lista = []
                pi_lista = []
                z_lista = []
                for est, jug, pi, z in lote:
                    x_tensor = mcts.estado_a_tensor(est, jug).squeeze(0) # (2, 4, 4)
                    x_lista.append(x_tensor)
                    pi_lista.append(torch.tensor(pi, dtype=torch.float32))
                    z_lista.append(torch.tensor([z], dtype=torch.float32))
                    
                x_batch = torch.stack(x_lista).to(dispositivo)
                pi_batch = torch.stack(pi_lista).to(dispositivo)
                z_batch = torch.stack(z_lista).to(dispositivo)
                
                optimizador.zero_grad()
                pred_log_pi, pred_v = modelo(x_batch)
                
                # Pérdida de valor (MSE) y pérdida de política (CrossEntropy = -sum(pi * log_p))
                loss_v = F.mse_loss(pred_v, z_batch)
                loss_p = -torch.mean(torch.sum(pi_batch * pred_log_pi, dim=1))
                loss_total = loss_v + loss_p
                
                loss_total.backward()
                optimizador.step()
                
                suma_loss_tot += loss_total.item()
                suma_loss_val += loss_v.item()
                suma_loss_pol += loss_p.item()
                num_pasos += 1
                
        media_loss_tot = suma_loss_tot / max(1, num_pasos)
        media_loss_val = suma_loss_val / max(1, num_pasos)
        media_loss_pol = suma_loss_pol / max(1, num_pasos)
        
        historial_loss_total.append(media_loss_tot)
        historial_loss_valor.append(media_loss_val)
        historial_loss_politica.append(media_loss_pol)
        
        # 3. Evaluación de fuerza frente a Random (10 partidas)
        winrate_rand = evaluar_contra_random(juego, mcts, num_partidas=10)
        historial_winrate_vs_random.append(winrate_rand)
        
        t_iter = time.perf_counter() - t_inicio_iter
        historial_tiempos.append(t_iter)
        
        print(f"[{iter_idx:02d}/{num_iteraciones:02d}] | {media_loss_tot:<12.4f} | {media_loss_val:<12.4f} | {media_loss_pol:<12.4f} | {winrate_rand:>10.1f}% | {t_iter:<10.2f}s")
        
    t_total = time.perf_counter() - t_inicio_total
    print("-" * 75)
    print(f"Entrenamiento Tabula Rasa finalizado en {t_total:.2f}s.")
    
    # 4. Guardar Pesos Finales
    torch.save(modelo.state_dict(), ruta_modelo)
    print(f"[+] Modelo final guardado en: {ruta_modelo}")
    
    # 5. Generar Gráfica de Pérdida (300 DPI)
    plt.figure(figsize=(10, 6))
    iters_range = range(1, num_iteraciones + 1)
    plt.plot(iters_range, historial_loss_total, label="Pérdida Total", color="#1f77b4", marker="o", linewidth=2)
    plt.plot(iters_range, historial_loss_valor, label="Pérdida de Valor (MSE)", color="#2ca02c", marker="s", linewidth=1.8)
    plt.plot(iters_range, historial_loss_politica, label="Pérdida de Política (CrossEntropy)", color="#d62728", marker="^", linewidth=1.8)
    plt.xlabel("Iteración de Aprendizaje por Refuerzo", fontsize=12, fontweight="bold")
    plt.ylabel("Pérdida Media", fontsize=12, fontweight="bold")
    plt.title("Evolución de Pérdidas en Auto-Juego Tabula Rasa (Othello 4x4)", fontsize=13, fontweight="bold", pad=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(ruta_grafica, dpi=300)
    plt.close()
    print(f"[+] Gráfica de pérdidas guardada en: {ruta_grafica}")
    
    # 6. Generar Tabla LaTeX (booktabs)
    lineas_tex = [
        r"\begin{table}[htbp]",
        r"  \centering",
        r"  \caption{Progreso del Aprendizaje Tabula Rasa en Othello 4$\times$4}",
        r"  \label{tab:progreso_selfplay_4x4}",
        r"  \begin{tabular}{cccccc}",
        r"    \toprule",
        r"    \textbf{Iteración} & \textbf{Pérdida Total} & \textbf{Pérdida Valor} & \textbf{Pérdida Política} & \textbf{Winrate vs Random (\%)} & \textbf{Tiempo (s)} \\",
        r"    \midrule"
    ]
    for it, lt, lv, lp, wr, tm in zip(iters_range, historial_loss_total, historial_loss_valor, historial_loss_politica, historial_winrate_vs_random, historial_tiempos):
        lineas_tex.append(f"    {it:^9} & {lt:^13.4f} & {lv:^13.4f} & {lp:^16.4f} & {wr:>16.1f}\\% & {tm:^10.2f} \\\\")
    lineas_tex.extend([
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}"
    ])
    codigo_latex = "\n".join(lineas_tex)
    
    with open(ruta_tabla, "w", encoding="utf-8") as f:
        f.write(codigo_latex)
    print(f"[+] Tabla LaTeX guardada en: {ruta_tabla}")
    
    # 7. Copia a artefactos_memoria
    shutil.copy2(ruta_modelo, os.path.join(dir_memoria_modelos, "tabula_rasa_4x4.pth"))
    shutil.copy2(ruta_grafica, os.path.join(dir_memoria_figuras, "curva_perdida_tabula_rasa_4x4.png"))
    shutil.copy2(ruta_tabla, os.path.join(dir_memoria_tablas, "tabla_progreso_selfplay.tex"))
    
    dir_con_tilde = os.path.join(directorio_ptfg, "Capítulo_6")
    if dir_con_tilde != dir_cap6:
        try:
            shutil.copytree(dir_cap6, dir_con_tilde, dirs_exist_ok=True)
        except Exception:
            pass
            
    print("\n[+] Artefactos de la Fase 3 copiados con éxito a artefactos_memoria/")
    print("=" * 85 + "\n")
    
    return modelo

if __name__ == "__main__":
    ejecutar_tabula_rasa()
