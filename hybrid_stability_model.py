import numpy as np
import torch
import torch.nn as nn
from scipy.stats import levy_stable

# =====================================================================
# 1. СТОХАСТИЧЕСКИЕ ДВИЖКИ ДАННЫХ (Параметры калибровки Elia Group)
# =====================================================================
def generate_solar_irradiance(timesteps, alpha=0.942, sigma_s=0.031):
    """Формулы (1)-(2): Облачные транзиты на базе Марковских процессов"""
    G_clear = 800 * np.maximum(0, np.sin(np.linspace(0, np.pi * (timesteps / 1440), timesteps)))
    C = np.ones(timesteps)
    Z = np.random.uniform(0.2, 1.0, timesteps)
    
    for t in range(1, timesteps):
        C[t] = alpha * C[t-1] + (1 - alpha) * Z[t]
        
    epsilon_s = np.random.normal(0, sigma_s, timesteps) * 100
    G = G_clear * C + epsilon_s
    return np.maximum(0, G)

def generate_wind_power(timesteps, phi1=0.85, theta1=0.1, prated=1200):
    """Формулы (3)-(4): Альфа-устойчивый шум Леви и кривая мощности TSO"""
    v = np.zeros(timesteps)
    v[0] = 8.0
    epsilon_w = np.random.normal(0, 1.2, timesteps)
    eta = levy_stable.rvs(alpha=1.5, beta=0.0, loc=0, scale=0.5, size=timesteps)
    
    for t in range(1, timesteps):
        v[t] = phi1 * v[t-1] + theta1 * epsilon_w[t-1] + epsilon_w[t] + eta[t]
        
    v = np.clip(v, 0, 30)
    P_w = np.zeros(timesteps)
    v_in, v_rated, v_out = 3.0, 12.0, 25.0
    
    idx_curve = (v >= v_in) & (v < v_rated)
    P_w[idx_curve] = prated * ((v[idx_curve] - v_in) / (v_rated - v_in))**3
    P_w[(v >= v_rated) & (v < v_out)] = prated
    return P_w

# =====================================================================
# 2. ВЕРОЯТНОСТНАЯ НЕЙРОСЕТЬ (PNN) С MONTE CARLO DROPOUT
# =====================================================================
class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(nn.functional.softplus(x))

class ProbabilisticNeuralNetwork(nn.Module):
    """Раздел 2.D: PNN со слоями 128->64 и MC Dropout"""
    def __init__(self, input_dim=1):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            Mish(),
            nn.Dropout(p=0.2),
            nn.Linear(128, 64),
            Mish(),
            nn.Dropout(p=0.2),
            nn.Linear(64, 3)
        )
        
    def forward(self, x):
        return self.network(x)

# =====================================================================
# 3. КОНТУР АДАПТАЦИИ ПО ЛЯПУНОВУ
# =====================================================================
class LyapunovAdaptiveEnsemble:
    """Раздел 2.F: Ограниченное Ляпуновым перераспределение весов в реальном времени"""
    def __init__(self, lambda_rate=0.05, gamma=0.01):
        self.wd = 0.5
        self.wp = 0.5
        self.lambda_rate = lambda_rate
        self.gamma = gamma
        self.delta_noise = 0.002

    def update_weights(self, y_true, y_gbr, y_pnn):
        l_d = (y_true - y_gbr) ** 2
        l_p = (y_true - y_pnn) ** 2
        e_t = l_d - l_p
        
        max_err = max(abs(e_t), 1e-4)
        adaptive_lambda = min(self.lambda_rate, 1.99 / max_err)
        
        numerator_d = self.wd * np.exp(-adaptive_lambda * l_d)
        numerator_p = self.wp * np.exp(-adaptive_lambda * l_p)
        denom = numerator_d + numerator_p
        
        self.wd = numerator_d / denom
        self.wp = 1.0 - self.wd
        
        self.wd = np.clip(self.wd, self.delta_noise, 1.0 - self.delta_noise)
        self.wp = 1.0 - self.wd
        return self.wd, self.wp

# =====================================================================
# 4. ДЕМОНСТРАЦИОННЫЙ ЗАПУСК СИМУЛЯЦИИ
# =====================================================================
if __name__ == "__main__":
    print("--- Инициализация симуляции цифрового двойника ЭЭС ---")
    np.random.seed(42)
    timesteps = 10
    
    solar_fleet = generate_solar_irradiance(timesteps)
    wind_fleet = generate_wind_power(timesteps)
    y_true_grid = (solar_fleet + wind_fleet) / 2000.0
    
    y_gbr_pred = y_true_grid + np.random.normal(0, 0.04, timesteps)
    pnn_model = ProbabilisticNeuralNetwork(input_dim=1)
    adaptive_loop = LyapunovAdaptiveEnsemble(lambda_rate=0.1, gamma=0.01)
    
    print("\nЗапуск контура адаптации SCADA...")
    for t in range(5):
        x_tensor = torch.tensor([[y_true_grid[t]]], dtype=torch.float32)
        outputs = pnn_model(x_tensor).detach().numpy()[0]
        y_pnn_median = outputs[1] # Медианный квантиль q50
        
        w_gbr, w_pnn = adaptive_loop.update_weights(y_true_grid[t], y_gbr_pred[t], y_pnn_median)
        y_ensemble = w_gbr * y_gbr_pred[t] + w_pnn * y_pnn_median
        
        print(f"Шаг {t+1:02d} | Истинное: {y_true_grid[t]:.4f} | Вес GBR (w_d): {w_gbr:.3f} | Вес PNN (w_p): {w_pnn:.3f} | Ансамбль: {y_ensemble:.4f}")
