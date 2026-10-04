import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from dataset_pytorch import obtener_dataloaders

class RedDeValorCNN(nn.Module):
    """
    Red de Valor Convolucional (CNN) para evaluar tableros de Othello 8x8.
    Entrada: (batch_size, 2, 8, 8)
      - Canal 0: Fichas del jugador activo (Turno_Actual).
      - Canal 1: Fichas del rival.
    Salida: (batch_size, 1) acotado en [-1.0, 1.0] con Tanh, representando
            la probabilidad/valor de victoria para el jugador activo.
    """
    def __init__(self):
        super(RedDeValorCNN, self).__init__()
        
        # Bloque convolucional: 3 capas Conv2d (32, 64, 64) con BatchNorm y ReLU
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        # Bloque denso: Flatten -> Linear(64*8*8, 128) -> ReLU -> Dropout(0.3) -> Linear(128, 1) -> Tanh
        self.dense_block = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Tanh()
        )
        
    def forward(self, x):
        x = self.conv_block(x)
        x = self.dense_block(x)
        return x

def entrenar_modelo(
    ruta_csv="pruebas/prueba_max_pesos/prueba_max_pesos.csv",
    num_epocas=25,
    batch_size=64,
    train_ratio=0.8,
    lr=0.001,
    weight_decay=1e-4,
    modelo_salida="mejor_red_valor_8x8.pth",
    grafica_salida="curva_perdida.png"
):
    # Selección de dispositivo (GPU si está disponible, CPU en caso contrario)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== INICIANDO ENTRENAMIENTO DE RED DE VALOR CNN (8x8) ===")
    print(f"Dispositivo de cómputo: {device}")
    print(f"Cargando dataset desde: {ruta_csv}...")
    
    # Determinar carpeta de salida a partir de la ubicación del CSV
    carpeta_salida = os.path.dirname(ruta_csv)
    if carpeta_salida == "":
        carpeta_salida = "."
    os.makedirs(carpeta_salida, exist_ok=True)
    
    ruta_modelo = os.path.join(carpeta_salida, os.path.basename(modelo_salida)) if os.path.dirname(modelo_salida) == "" else modelo_salida
    ruta_grafica = os.path.join(carpeta_salida, os.path.basename(grafica_salida)) if os.path.dirname(grafica_salida) == "" else grafica_salida
    
    train_loader, val_loader = obtener_dataloaders(
        ruta_csv,
        batch_size=batch_size,
        train_ratio=train_ratio,
        shuffle=True,
        seed=42
    )
    
    total_train = len(train_loader.dataset)
    total_val = len(val_loader.dataset)
    print(f"Muestras de entrenamiento: {total_train} | Muestras de validación: {total_val}")
    print(f"Batches por época (Train): {len(train_loader)} | (Val): {len(val_loader)}")
    print(f"Carpeta de salida de artefactos: {carpeta_salida}")
    
    # Inicialización del modelo, función de pérdida y optimizador
    model = RedDeValorCNN().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    historial_train_loss = []
    historial_val_loss = []
    mejor_val_loss = float("inf")
    
    print("\n--- Bucle de Entrenamiento (25 Épocas) ---")
    for epoca in range(1, num_epocas + 1):
        # 1. Fase de Entrenamiento
        model.train()
        suma_train_loss = 0.0
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs).squeeze(-1)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            suma_train_loss += loss.item() * inputs.size(0)
            
        train_loss = suma_train_loss / total_train
        historial_train_loss.append(train_loss)
        
        # 2. Fase de Validación
        model.eval()
        suma_val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                
                outputs = model(inputs).squeeze(-1)
                loss = criterion(outputs, targets)
                suma_val_loss += loss.item() * inputs.size(0)
                
        val_loss = suma_val_loss / total_val
        historial_val_loss.append(val_loss)
        
        # 3. Guardar el mejor modelo según menor pérdida de validación
        guardado = ""
        if val_loss < mejor_val_loss:
            mejor_val_loss = val_loss
            torch.save(model.state_dict(), ruta_modelo)
            guardado = f" -> [*] Guardado mejor modelo ({mejor_val_loss:.5f})"
            
        print(f"Época [{epoca:02d}/{num_epocas:02d}] - Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f}{guardado}")
        
    print(f"\nEntrenamiento finalizado.")
    print(f"Mejor pérdida de validación: {mejor_val_loss:.5f}")
    print(f"Pesos del mejor modelo guardados en: {ruta_modelo}")
    
    # 4. Generación y exportación de la gráfica de pérdida
    plt.figure(figsize=(10, 6))
    epocas = range(1, num_epocas + 1)
    plt.plot(epocas, historial_train_loss, label="Pérdida Entrenamiento (Train MSE)", color="#1f77b4", marker="o", linewidth=2)
    plt.plot(epocas, historial_val_loss, label="Pérdida Validación (Val MSE)", color="#ff7f0e", marker="s", linewidth=2)
    plt.xlabel("Época", fontsize=12)
    plt.ylabel("Pérdida Media (MSE)", fontsize=12)
    plt.title("Evolución de la Función de Pérdida (Red de Valor CNN 8x8)", fontsize=14, fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(ruta_grafica, dpi=300)
    plt.close()
    print(f"Gráfica de evolución de pérdida guardada en: {ruta_grafica}")
    
    return model, historial_train_loss, historial_val_loss

if __name__ == "__main__":
    import glob
    csvs = glob.glob("pruebas/**/*.csv", recursive=True) + glob.glob("*.csv")
    csv_defecto = "pruebas/prueba_max_pesos/prueba_max_pesos.csv"
    if csvs and (not os.path.exists(csv_defecto)):
        csv_defecto = csvs[0]
        
    print("\n=== ENTRENAMIENTO SUPERVISADO DE RED DE VALOR CNN ===")
    if csvs:
        print("Archivos CSV de entrenamiento encontrados:")
        for i, c in enumerate(csvs, 1):
            print(f"  {i}. {c}")
            
    entrada_csv = input(f"\nIntroduce la ruta del archivo CSV [Enter = {csv_defecto}]: ").strip()
    ruta_csv_final = csv_defecto if entrada_csv == "" else entrada_csv
    
    entrenar_modelo(ruta_csv=ruta_csv_final)
