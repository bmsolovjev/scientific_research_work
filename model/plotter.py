import matplotlib.pyplot as plt
import numpy as np
import csv

def read_vectors_from_log(filename='output_log.csv', target_theta=30.0):
    """
    Чтение волновых векторов для заданного угла из CSV-файла.
    """
    with open(filename, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            theta = float(row['theta_deg'])
            if np.isclose(theta, target_theta, atol=0.01):
                k_I = np.array([float(row['k_Ix']), float(row['k_Iy']), float(row['k_Iz'])])
                k_R = np.array([float(row['k_Rx']), float(row['k_Ry']), float(row['k_Rz'])])
                k_T = np.array([float(row['k_Tx']), float(row['k_Ty']), float(row['k_Tz'])])
                return {
                    'theta': theta,
                    'k_I': k_I,
                    'k_R': k_R,
                    'k_T': k_T
                }
    return None


import matplotlib.pyplot as plt
import numpy as np
import csv

def read_vectors_from_log(filename='output_log.csv', target_theta=30.0):
    """
    Чтение волновых векторов для заданного угла из CSV-файла.
    """
    with open(filename, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            theta = float(row['theta_deg'])
            if np.isclose(theta, target_theta, atol=0.01):
                k_I = np.array([float(row['k_Ix']), float(row['k_Iy']), float(row['k_Iz'])])
                k_R = np.array([float(row['k_Rx']), float(row['k_Ry']), float(row['k_Rz'])])
                k_T = np.array([float(row['k_Tx']), float(row['k_Ty']), float(row['k_Tz'])])
                return {
                    'theta': theta,
                    'k_I': k_I,
                    'k_R': k_R,
                    'k_T': k_T
                }
    return None


def plot_wave_vectors_from_log(filename='output_log.csv', target_theta=30.0):
    """
    Построение диаграммы волновых векторов.
    Ось X направлена ВНИЗ, ось Z ВПРАВО.
    Нормаль к границе — вертикальная линия (вдоль оси X).
    """
    data = read_vectors_from_log(filename, target_theta)
    
    if data is None:
        print(f"Ошибка: угол {target_theta}° не найден в файле {filename}")
        return
    
    k_I = data['k_I']
    k_R = data['k_R']
    k_T = data['k_T']
    theta = data['theta']
    
    # Масштабирование
    scale = 1e-7
    k_I_scaled = k_I * scale
    k_R_scaled = k_R * scale
    k_T_scaled = k_T * scale
    
    # Инвертируем X-компоненту для отрисовки
    # В matplotlib: ось Y вверх (+), ось X вправо (+)
    # У нас: ось X вниз, ось Z вправо
    # Поэтому: X_plot = Z_real, Y_plot = -X_real
    k_I_plot_x = k_I_scaled[2]   # Z → X графика (вправо)
    k_I_plot_y = -k_I_scaled[0]  # -X → Y графика (X>0 вниз → Y<0)
    
    k_R_plot_x = k_R_scaled[2]   # Z → X графика
    k_R_plot_y = -k_R_scaled[0]  # -X → Y графика (X<0 вверх → Y>0)
    
    k_T_plot_x = k_T_scaled[2]   # Z → X графика
    k_T_plot_y = -k_T_scaled[0]  # -X → Y графика (X>0 вниз → Y<0)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Граница раздела сред
    ax.axhline(y=0, color='black', linewidth=3, label='Граница раздела (x = 0)')
    
    # Нормаль к границе (вертикальная линия)
    ax.axvline(x=0, color='gray', linewidth=1.5, linestyle='--', alpha=0.5)
    
    # Подписи сред
    ax.text(1.5, 1.5, 'Среда 1 (воздух)\nn₁ = 1.0, x < 0', fontsize=12,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
            ha='center', fontweight='bold')
    ax.text(1.5, -1.5, 'Среда 2 (стекло)\nn₂ = 1.5, x > 0', fontsize=12,
            bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8),
            ha='center', fontweight='bold')
    
    # Падающий луч: из среды 1 (сверху) к точке падения
    ax.arrow(-k_I_plot_x, -k_I_plot_y, k_I_plot_x, k_I_plot_y,
             head_width=0.08, head_length=0.12,
             fc='limegreen', ec='darkgreen', linewidth=2.5,
             label=f'Падающий луч (kᵢ)\n|kᵢ| = {np.linalg.norm(k_I)*scale:.2f}',
             length_includes_head=True, zorder=5)
    
    # Отражённый луч: в среду 1 (вверх)
    ax.arrow(0, 0, k_R_plot_x, k_R_plot_y,
             head_width=0.08, head_length=0.12,
             fc='red', ec='darkred', linewidth=2.5,
             label=f'Отражённый луч (kᵣ)\n|kᵣ| = {np.linalg.norm(k_R)*scale:.2f}',
             length_includes_head=True, zorder=5)
    
    # Преломлённый луч: в среду 2 (вниз)
    ax.arrow(0, 0, k_T_plot_x, k_T_plot_y,
             head_width=0.08, head_length=0.12,
             fc='dodgerblue', ec='darkblue', linewidth=2.5,
             label=f'Преломлённый луч (kₜ)\n|kₜ| = {np.linalg.norm(k_T)*scale:.2f}',
             length_includes_head=True, zorder=5)
    
    # Точка падения
    ax.plot(0, 0, 'o', color='black', markersize=10, zorder=10)
    
    # ============================================================
    # Углы (ИСПРАВЛЕНО!)
    # Нормаль — это вертикальная линия (вдоль оси X, x=0)
    # Угол отсчитывается от нормали к лучу
    # ============================================================
    
    # Направление нормали в среде 1: вверх (0, 1) — против оси X
    # Направление нормали в среде 2: вниз (0, -1) — по оси X
    
    # Угол падения: от нормали (вверх) к падающему лучу
    # Падающий луч идёт вправо-вниз: вектор (k_I_plot_x, k_I_plot_y)
    # k_I_plot_y < 0 (вниз), k_I_plot_x > 0 (вправо)
    # Угол от вертикали вверх: arctan(k_I_plot_x / |k_I_plot_y|)
    theta_rad = np.arctan2(k_I_plot_x, -k_I_plot_y)  # угол от нормали вверх
    
    # Угол преломления: от нормали (вниз) к преломлённому лучу
    # k_T_plot_y < 0 (вниз), k_T_plot_x > 0 (вправо)
    theta_T_rad = np.arctan2(k_T_plot_x, -k_T_plot_y)  # угол от нормали вниз
    
    arc_r = 0.6
    
    # Угол падения θ (дуга от нормали вверх к падающему лучу)
    # Нормаль вверх: угол 90° в полярных координатах (направление (0,1))
    # Падающий луч: угол 90° - θ от оси X
    theta_start = np.pi/2  # нормаль вверх
    theta_end = np.pi/2 - theta_rad  # падающий луч
    theta_arc = np.linspace(theta_end, theta_start, 30)
    ax.plot(arc_r * np.cos(theta_arc), arc_r * np.sin(theta_arc),
            'red', linewidth=2, linestyle='--')
    ax.text(0.15, 0.55, f'θ = {theta:.1f}°', fontsize=11,
            color='darkred', fontweight='bold')
    
    # Угол отражения θ (дуга от нормали вверх к отражённому лучу)
    # Отражённый луч: угол 90° + θ от оси X
    theta_refl_start = np.pi/2  # нормаль вверх
    theta_refl_end = np.pi/2 + theta_rad  # отражённый луч
    theta_refl_arc = np.linspace(theta_refl_start, theta_refl_end, 30)
    ax.plot(arc_r * np.cos(theta_refl_arc), arc_r * np.sin(theta_refl_arc),
            'green', linewidth=2, linestyle='--')
    ax.text(0.15, 0.45, f'θ = {theta:.1f}°', fontsize=11,
            color='darkgreen', fontweight='bold')
    
    # Угол преломления θ_T (дуга от нормали вниз к преломлённому лучу)
    # Нормаль вниз: угол -90° (или 270°) 
    # Преломлённый луч: угол -90° + θ_T
    theta_T_start = -np.pi/2  # нормаль вниз
    theta_T_end = -np.pi/2 + theta_T_rad  # преломлённый луч
    theta_T_arc = np.linspace(theta_T_start, theta_T_end, 30)
    ax.plot(arc_r * np.cos(theta_T_arc), arc_r * np.sin(theta_T_arc),
            'blue', linewidth=2, linestyle='--')
    ax.text(0.2, -0.55, f'θₜ = {np.degrees(theta_T_rad):.1f}°', fontsize=11,
            color='darkblue', fontweight='bold')
    
    # Настройка осей
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2.5, 2.5)
    ax.set_aspect('equal')
    
    ax.set_xlabel('Ось Z (вдоль границы) →', fontsize=14, fontweight='bold')
    ax.set_ylabel('Ось X (↓ в среду 2)', fontsize=14, fontweight='bold')
    
    ax.set_title(f'Диаграмма волновых векторов в плоскости XZ\n'
                 f'(угол падения {theta:.1f}°, воздух → стекло)',
                 fontsize=15, fontweight='bold', pad=20)
    
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
    ax.grid(True, linestyle=':', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('wave_vectors_diagram.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_amplitude_vs_angle(filename='output_log.csv'):
    """
    График зависимости амплитудных коэффициентов Френеля от угла падения.
    """
    angles = []
    r_s_vals, t_s_vals = [], []
    r_p_vals, t_p_vals = [], []
    
    with open(filename, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            angles.append(float(row['theta_deg']))
            r_s_vals.append(float(row['r_s']))
            t_s_vals.append(float(row['t_s']))
            r_p_vals.append(float(row['r_p']))
            t_p_vals.append(float(row['t_p']))
    
    angles = np.array(angles)
    r_s_vals = np.array(r_s_vals)
    t_s_vals = np.array(t_s_vals)
    r_p_vals = np.array(r_p_vals)
    t_p_vals = np.array(t_p_vals)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # s-поляризация
    ax1.plot(angles, r_s_vals, 'o-', color='red', label='r_s (отражение)', markersize=6)
    ax1.plot(angles, t_s_vals, 's-', color='blue', label='t_s (пропускание)', markersize=6)
    ax1.set_xlabel('Угол падения θ, градусы', fontsize=12)
    ax1.set_ylabel('Амплитудный коэффициент', fontsize=12)
    ax1.set_title('s-поляризация (TE)', fontsize=13)
    ax1.legend(fontsize=11)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.set_ylim(-0.5, 1.2)
    
    # p-поляризация
    ax2.plot(angles, r_p_vals, 'o-', color='red', label='r_p (отражение)', markersize=6)
    ax2.plot(angles, t_p_vals, 's-', color='blue', label='t_p (пропускание)', markersize=6)
    ax2.set_xlabel('Угол падения θ, градусы', fontsize=12)
    ax2.set_ylabel('Амплитудный коэффициент', fontsize=12)
    ax2.set_title('p-поляризация (TM)', fontsize=13)
    ax2.legend(fontsize=11)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.set_ylim(-0.5, 1.2)
    
    fig.suptitle('Коэффициенты Френеля для границы воздух-стекло (n₁=1.0, n₂=1.5)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('amplitude_vs_angle.png', dpi=150)
    plt.show()


if __name__ == "__main__":
    print("Построение диаграммы волновых векторов (θ = 30°)...")
    plot_wave_vectors_from_log('output_log.csv', target_theta=30.0)
    
    print("Построение графиков коэффициентов Френеля...")
    try:
        plot_amplitude_vs_angle('output_log.csv')
    except FileNotFoundError:
        print("Файл output_log.csv не найден. Сначала запустите main.py")
    except KeyError as e:
        print(f"Ошибка чтения данных: {e}")

def plot_amplitude_vs_angle(filename='output_log.csv'):
    """
    График зависимости амплитудных коэффициентов Френеля от угла падения.
    """
    angles = []
    r_s_vals, t_s_vals = [], []
    r_p_vals, t_p_vals = [], []
    
    with open(filename, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            angles.append(float(row['theta_deg']))
            r_s_vals.append(float(row['r_s']))
            t_s_vals.append(float(row['t_s']))
            r_p_vals.append(float(row['r_p']))
            t_p_vals.append(float(row['t_p']))
    
    angles = np.array(angles)
    r_s_vals = np.array(r_s_vals)
    t_s_vals = np.array(t_s_vals)
    r_p_vals = np.array(r_p_vals)
    t_p_vals = np.array(t_p_vals)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # s-поляризация
    ax1.plot(angles, r_s_vals, 'o-', color='red', label='r_s (отражение)', markersize=6)
    ax1.plot(angles, t_s_vals, 's-', color='blue', label='t_s (пропускание)', markersize=6)
    ax1.set_xlabel('Угол падения θ, градусы', fontsize=12)
    ax1.set_ylabel('Амплитудный коэффициент', fontsize=12)
    ax1.set_title('s-поляризация (TE)', fontsize=13)
    ax1.legend(fontsize=11)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.set_ylim(-0.5, 1.2)
    
    # p-поляризация
    ax2.plot(angles, r_p_vals, 'o-', color='red', label='r_p (отражение)', markersize=6)
    ax2.plot(angles, t_p_vals, 's-', color='blue', label='t_p (пропускание)', markersize=6)
    ax2.set_xlabel('Угол падения θ, градусы', fontsize=12)
    ax2.set_ylabel('Амплитудный коэффициент', fontsize=12)
    ax2.set_title('p-поляризация (TM)', fontsize=13)
    ax2.legend(fontsize=11)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.set_ylim(-0.5, 1.2)
    
    fig.suptitle('Коэффициенты Френеля для границы воздух-стекло (n₁=1.0, n₂=1.5)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('amplitude_vs_angle.png', dpi=150)
    plt.show()


if __name__ == "__main__":
    print("Построение диаграммы волновых векторов (θ = 30°)...")
    plot_wave_vectors_from_log('output_log.csv', target_theta=30.0)
    
    print("Построение графиков коэффициентов Френеля...")
    try:
        plot_amplitude_vs_angle('output_log.csv')
    except FileNotFoundError:
        print("Файл output_log.csv не найден. Сначала запустите main.py")
    except KeyError as e:
        print(f"Ошибка чтения данных: {e}")
