import matplotlib.pyplot as plt
import numpy as np
import csv

# ============================================================================
# КОНФИГУРАЦИЯ (меняй здесь всё, что нужно)
# ============================================================================

CONFIG = {
    'theta_deg': 30.0,           # Угол падения для всех графиков
    'n1': 1.0,                   # Показатель преломления среды 1 (воздух)
    'n2': 1.5,                   # Показатель преломления среды 2 (стекло)
    'wavelength_min': 200,       # Мин. длина волны для спектральных графиков, нм
    'wavelength_max': 2000,      # Макс. длина волны для спектральных графиков, нм
    'dpi': 300,                  # Разрешение сохраняемых графиков
    'csv_filename': 'output_log.csv'  # Имя CSV файла с данными
}

# ============================================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ============================================================================

def read_vectors_from_log(filename=None, target_theta=None):
    """
    Чтение волновых векторов для заданного угла из CSV-файла.
    """
    if filename is None:
        filename = CONFIG['csv_filename']
    if target_theta is None:
        target_theta = CONFIG['theta_deg']
    
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


# ============================================================================
# ГРАФИК 1: ДИАГРАММА ВОЛНОВЫХ ВЕКТОРОВ
# ============================================================================

def plot_wave_vectors_from_log(filename=None, target_theta=None):
    """
    Построение диаграммы волновых векторов.
    """
    if filename is None:
        filename = CONFIG['csv_filename']
    if target_theta is None:
        target_theta = CONFIG['theta_deg']
    
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
    ax.text(1.5, 1.5, f'Среда 1 (воздух)\nn₁ = {CONFIG["n1"]}, x < 0', fontsize=12,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
            ha='center', fontweight='bold')
    ax.text(1.5, -1.5, f'Среда 2 (стекло)\nn₂ = {CONFIG["n2"]}, x > 0', fontsize=12,
            bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8),
            ha='center', fontweight='bold')
    
    # Падающий луч
    ax.arrow(-k_I_plot_x, -k_I_plot_y, k_I_plot_x, k_I_plot_y,
             head_width=0.08, head_length=0.12,
             fc='limegreen', ec='darkgreen', linewidth=2.5,
             label=f'Падающий луч (kᵢ)\n|kᵢ| = {np.linalg.norm(k_I)*scale:.2f}',
             length_includes_head=True, zorder=5)
    
    # Отражённый луч
    ax.arrow(0, 0, k_R_plot_x, k_R_plot_y,
             head_width=0.08, head_length=0.12,
             fc='red', ec='darkred', linewidth=2.5,
             label=f'Отражённый луч (kᵣ)\n|kᵣ| = {np.linalg.norm(k_R)*scale:.2f}',
             length_includes_head=True, zorder=5)
    
    # Преломлённый луч
    ax.arrow(0, 0, k_T_plot_x, k_T_plot_y,
             head_width=0.08, head_length=0.12,
             fc='dodgerblue', ec='darkblue', linewidth=2.5,
             label=f'Преломлённый луч (kₜ)\n|kₜ| = {np.linalg.norm(k_T)*scale:.2f}',
             length_includes_head=True, zorder=5)
    
    # Точка падения
    ax.plot(0, 0, 'o', color='black', markersize=10, zorder=10)
    
    # Углы
    theta_rad = np.arctan2(k_I_plot_x, -k_I_plot_y)
    theta_T_rad = np.arctan2(k_T_plot_x, -k_T_plot_y)
    arc_r = 0.6
    
    # Угол падения
    theta_start = np.pi/2
    theta_end = np.pi/2 - theta_rad
    theta_arc = np.linspace(theta_end, theta_start, 30)
    ax.plot(arc_r * np.cos(theta_arc), arc_r * np.sin(theta_arc),
            'red', linewidth=2, linestyle='--')
    ax.text(0.15, 0.55, f'θ = {theta:.1f}°', fontsize=11,
            color='darkred', fontweight='bold')
    
    # Угол отражения
    theta_refl_end = np.pi/2 + theta_rad
    theta_refl_arc = np.linspace(theta_start, theta_refl_end, 30)
    ax.plot(arc_r * np.cos(theta_refl_arc), arc_r * np.sin(theta_refl_arc),
            'green', linewidth=2, linestyle='--')
    ax.text(0.15, 0.45, f'θ = {theta:.1f}°', fontsize=11,
            color='darkgreen', fontweight='bold')
    
    # Угол преломления
    theta_T_start = -np.pi/2
    theta_T_end = -np.pi/2 + theta_T_rad
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
    plt.savefig('wave_vectors_diagram.png', dpi=CONFIG['dpi'], bbox_inches='tight')
    plt.show()


# ============================================================================
# ГРАФИК 2: АМПЛИТУДНЫЕ КОЭФФИЦИЕНТЫ ФРЕНЕЛЯ
# ============================================================================

def plot_amplitude_vs_angle(filename=None):
    """
    График зависимости амплитудных коэффициентов Френеля от угла падения.
    """
    if filename is None:
        filename = CONFIG['csv_filename']
    
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
    ax1.plot(angles, r_s_vals, 'o-', color='red', label='r_s (отражение)', markersize=6, linewidth=2)
    ax1.plot(angles, t_s_vals, 's-', color='blue', label='t_s (пропускание)', markersize=6, linewidth=2)
    ax1.set_xlabel('Угол падения θ, градусы', fontsize=12)
    ax1.set_ylabel('Амплитудный коэффициент', fontsize=12)
    ax1.set_title('s-поляризация (TE)', fontsize=13)
    ax1.legend(fontsize=11)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.set_ylim(-0.5, 1.2)
    ax1.axvline(x=CONFIG['theta_deg'], color='gray', linestyle=':', alpha=0.7, linewidth=1.5)
    
    # p-поляризация
    ax2.plot(angles, r_p_vals, 'o-', color='red', label='r_p (отражение)', markersize=6, linewidth=2)
    ax2.plot(angles, t_p_vals, 's-', color='blue', label='t_p (пропускание)', markersize=6, linewidth=2)
    ax2.set_xlabel('Угол падения θ, градусы', fontsize=12)
    ax2.set_ylabel('Амплитудный коэффициент', fontsize=12)
    ax2.set_title('p-поляризация (TM)', fontsize=13)
    ax2.legend(fontsize=11)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.set_ylim(-0.5, 1.2)
    ax2.axvline(x=CONFIG['theta_deg'], color='gray', linestyle=':', alpha=0.7, linewidth=1.5)
    
    fig.suptitle(f'Коэффициенты Френеля для границы воздух-стекло (n₁={CONFIG["n1"]}, n₂={CONFIG["n2"]})\n'
                 f'Вертикальная линия — выбранный угол θ = {CONFIG["theta_deg"]}°',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('amplitude_vs_angle.png', dpi=CONFIG['dpi'])
    plt.show()


# ============================================================================
# ГРАФИК 3: ЗАКОН СОХРАНЕНИЯ ЭНЕРГИИ
# ============================================================================

def plot_energy_conservation(filename=None):
    """
    График закона сохранения энергии R + T = 1.
    """
    if filename is None:
        filename = CONFIG['csv_filename']
    
    angles = []
    R_s, T_s = [], []
    R_p, T_p = [], []
    
    with open(filename, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            angles.append(float(row['theta_deg']))
            R_s.append(float(row['R_s']))
            T_s.append(float(row['T_s']))
            R_p.append(float(row['R_p']))
            T_p.append(float(row['T_p']))
    
    angles = np.array(angles)
    R_s = np.array(R_s)
    T_s = np.array(T_s)
    R_p = np.array(R_p)
    T_p = np.array(T_p)
    
    plt.figure(figsize=(10, 6))
    
    plt.plot(angles, R_s, 'o-', color='darkred', linewidth=2, markersize=6, label='R_s')
    plt.plot(angles, T_s, 's-', color='salmon', linewidth=2, markersize=6, label='T_s')
    plt.plot(angles, R_p, 'o-', color='darkblue', linewidth=2, markersize=6, label='R_p')
    plt.plot(angles, T_p, 's-', color='skyblue', linewidth=2, markersize=6, label='T_p')
    
    plt.plot(angles, R_s + T_s, '--', color='black', linewidth=2, label='R_s + T_s')
    plt.plot(angles, R_p + T_p, ':', color='black', linewidth=2, label='R_p + T_p')
    
    plt.axvline(x=CONFIG['theta_deg'], color='gray', linestyle=':', alpha=0.7, linewidth=1.5,
                label=f'θ = {CONFIG["theta_deg"]}°')
    
    plt.xlabel('Угол падения θ, градусы', fontsize=14, fontweight='bold')
    plt.ylabel('Энергетические коэффициенты (R, T)', fontsize=14, fontweight='bold')
    plt.title(f'Закон сохранения энергии: R + T = 1\n'
              f'воздух (n₁={CONFIG["n1"]}) → стекло (n₂={CONFIG["n2"]})',
              fontsize=14, fontweight='bold')
    
    plt.xlim(0, 90)
    plt.ylim(0, 1.05)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=11, loc='best')
    
    plt.tight_layout()
    plt.savefig('energy_conservation.png', dpi=CONFIG['dpi'])
    plt.show()


# ============================================================================
# ГРАФИК 4: СПЕКТРАЛЬНАЯ ЗАВИСИМОСТЬ (ДИСПЕРСИЯ)
# ============================================================================

def plot_wavelength_dependence(theta_deg=None):
    """
    Спектральная зависимость R(λ) и T(λ) для стекла с дисперсией по Коши.
    """
    if theta_deg is None:
        theta_deg = CONFIG['theta_deg']
    
    n1 = CONFIG['n1']
    theta_rad = np.radians(theta_deg)
    
    wavelengths_nm = np.linspace(CONFIG['wavelength_min'], CONFIG['wavelength_max'], 500)
    wavelengths_um = wavelengths_nm / 1000.0
    
    R_avg = []
    T_avg = []
    
    for lam_um in wavelengths_um:
        n2 = CONFIG['n2'] + 0.004 / (lam_um**2)
        
        sin_theta_T = n1 / n2 * np.sin(theta_rad)
        
        if sin_theta_T >= 1.0:
            R_avg.append(1.0)
            T_avg.append(0.0)
            continue
        
        cos_theta_I = np.cos(theta_rad)
        cos_theta_T = np.sqrt(1 - sin_theta_T**2)
        
        # s-поляризация
        r_s = (n1 * cos_theta_I - n2 * cos_theta_T) / (n1 * cos_theta_I + n2 * cos_theta_T)
        R_s = r_s**2
        t_s = (2 * n1 * cos_theta_I) / (n1 * cos_theta_I + n2 * cos_theta_T)
        T_s = (n2 * cos_theta_T / (n1 * cos_theta_I)) * t_s**2
        
        # p-поляризация
        r_p = (n2 * cos_theta_I - n1 * cos_theta_T) / (n2 * cos_theta_I + n1 * cos_theta_T)
        R_p = r_p**2
        t_p = (2 * n1 * cos_theta_I) / (n2 * cos_theta_I + n1 * cos_theta_T)
        T_p = (n2 * cos_theta_T / (n1 * cos_theta_I)) * t_p**2
        
        R_avg.append(0.5 * (R_s + R_p))
        T_avg.append(0.5 * (T_s + T_p))
    
    R_avg = np.array(R_avg)
    T_avg = np.array(T_avg)
    
    plt.figure(figsize=(10, 6))
    
    plt.plot(wavelengths_nm, R_avg, 'r-', linewidth=2, label='R(λ)')
    plt.plot(wavelengths_nm, T_avg, 'b-', linewidth=2, label='T(λ)')
    plt.plot(wavelengths_nm, R_avg + T_avg, 'k--', linewidth=2, label='R + T = 1')
    
    plt.xlabel('Длина волны λ, нм', fontsize=14, fontweight='bold')
    plt.ylabel('Коэффициент', fontsize=14, fontweight='bold')
    plt.title(f'Спектральная зависимость коэффициентов Френеля\n'
              f'Фиксированный угол падения θ = {theta_deg}°\n'
              f'n₁ = {n1} (воздух), n₂(λ) = {CONFIG["n2"]} + 0.004/λ² (λ в мкм)',
              fontsize=12, fontweight='bold')
    
    plt.xlim(CONFIG['wavelength_min'], CONFIG['wavelength_max'])
    plt.ylim(0, 1.05)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=11, loc='best')
    
    plt.tight_layout()
    plt.savefig('wavelength_dependence.png', dpi=CONFIG['dpi'])
    plt.show()


# ============================================================================
# ГРАФИК 5: НОРМИРОВАННАЯ СПЕКТРАЛЬНАЯ ЗАВИСИМОСТЬ
# ============================================================================

def plot_normalized_wavelength_dependence(theta_deg=None):
    """
    Нормированный график спектральной зависимости.
    """
    if theta_deg is None:
        theta_deg = CONFIG['theta_deg']
    
    theta_rad = np.radians(theta_deg)
    n1 = CONFIG['n1']
    
    wavelengths_nm = np.linspace(CONFIG['wavelength_min'], CONFIG['wavelength_max'], 300)
    wavelengths_um = wavelengths_nm / 1000.0
    
    R = []
    
    for lam_um in wavelengths_um:
        n2 = CONFIG['n2'] + 0.004 / lam_um**2
        sin_theta_T = n1 / n2 * np.sin(theta_rad)
        
        if sin_theta_T >= 1.0:
            R.append(1.0)
            continue
        
        cos_theta_I = np.cos(theta_rad)
        cos_theta_T = np.sqrt(1 - sin_theta_T**2)
        
        # s-поляризация
        r_s = (n1 * cos_theta_I - n2 * cos_theta_T) / (n1 * cos_theta_I + n2 * cos_theta_T)
        R_s = r_s**2
        
        # p-поляризация
        r_p = (n2 * cos_theta_I - n1 * cos_theta_T) / (n2 * cos_theta_I + n1 * cos_theta_T)
        R_p = r_p**2
        
        R.append(0.5 * (R_s + R_p))
    
    R = np.array(R)
    
    # Нормировка
    R_norm = (R - R.min()) / (R.max() - R.min())
    T_norm = 1.0 - R_norm
    
    plt.figure(figsize=(10, 6))
    
    plt.plot(wavelengths_nm, R_norm, 'r-', linewidth=2, label='Rₙₒᵣₘ(λ)')
    plt.plot(wavelengths_nm, T_norm, 'b-', linewidth=2, label='Tₙₒᵣₘ(λ)')
    plt.axhline(0.5, linestyle='--', linewidth=1, color='gray', alpha=0.6, label='R = T = 0.5')
    
    plt.xlabel('Длина волны λ, нм', fontsize=14, fontweight='bold')
    plt.ylabel('Нормированная величина', fontsize=14, fontweight='bold')
    plt.title(f'Нормированное представление коэффициентов\n'
              f'Фиксированный угол падения θ = {theta_deg}°',
              fontsize=14, fontweight='bold')
    
    plt.xlim(CONFIG['wavelength_min'], CONFIG['wavelength_max'])
    plt.ylim(0, 1)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend(fontsize=11, loc='best')
    
    plt.tight_layout()
    plt.savefig('normalized_wavelength_dependence.png', dpi=CONFIG['dpi'], bbox_inches='tight')
    plt.show()


# ============================================================================
# ОСНОВНАЯ ПРОГРАММА
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("ПОСТРОЕНИЕ ДИАГРАММ И ГРАФИКОВ")
    print("=" * 70)
    print(f"\nТЕКУЩАЯ КОНФИГУРАЦИЯ:")
    print(f"  Угол падения θ = {CONFIG['theta_deg']}°")
    print(f"  n₁ = {CONFIG['n1']} (воздух)")
    print(f"  n₂ = {CONFIG['n2']} (стекло)")
    print(f"  Диапазон длин волн: {CONFIG['wavelength_min']} - {CONFIG['wavelength_max']} нм")
    print("\n" + "=" * 70)
    
    print("\n1. Построение диаграммы волновых векторов...")
    try:
        plot_wave_vectors_from_log()
    except FileNotFoundError:
        print("   Ошибка: файл output_log.csv не найден. Сначала запустите первый файл.")
    
    print("\n2. Построение графиков коэффициентов Френеля от угла...")
    try:
        plot_amplitude_vs_angle()
    except FileNotFoundError:
        print("   Ошибка: файл output_log.csv не найден.")
    
    print("\n3. Построение графика закона сохранения энергии...")
    try:
        plot_energy_conservation()
    except FileNotFoundError:
        print("   Ошибка: файл output_log.csv не найден.")
    
    print("\n4. Построение спектральной зависимости (от длины волны)...")
    plot_wavelength_dependence()
    
    print("\n5. Построение нормированной спектральной зависимости...")
    plot_normalized_wavelength_dependence()
    
    print("\n" + "=" * 70)
    print("ВСЕ ГРАФИКИ СОЗДАНЫ!")
    print(f"Все спектральные графики построены для угла θ = {CONFIG['theta_deg']}°")
    print("=" * 70)
