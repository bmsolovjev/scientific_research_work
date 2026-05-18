import matplotlib.pyplot as plt
import numpy as np
import csv

def read_log(filename='output_log.csv'):
    """Чтение данных из CSV-файла"""
    angles = []
    r_amps = []
    t_amps = []
    with open(filename, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            angles.append(float(row['theta_deg']))
            r_amps.append(float(row['R_amp']))
            t_amps.append(float(row['T_amp']))
    return np.array(angles), np.array(r_amps), np.array(t_amps)

def plot_amplitude_vs_angle(angles, r_amps, t_amps):
    """
    График 1: Зависимость амплитудных коэффициентов отражения и пропускания
    от угла падения (по данным прототипа)
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(angles, r_amps, 'o-', color='red', label='R (отражение), прототип', markersize=6)
    ax.plot(angles, t_amps, 's-', color='blue', label='T (пропускание), прототип', markersize=6)

    ax.set_xlabel('Угол падения θ, градусы', fontsize=12)
    ax.set_ylabel('Амплитудный коэффициент', fontsize=12)
    ax.set_title('Рис. 2 — Зависимость амплитуд отражённой и прошедшей волн\nот угла падения (расчёт прототипа, n₁ = 1.0, n₂ = 1.5)', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_ylim(-0.3, 1.0)

    plt.tight_layout()
    plt.savefig('amplitude_vs_angle.png', dpi=150)
    plt.show()

def plot_wave_vectors_diagram():
    """
    График 2: Диаграмма направленности волновых векторов в плоскости xz
    для угла падения 30° (качественная схема)
    """
    theta = 30
    theta_rad = np.radians(theta)
    n1, n2 = 1.0, 1.5

    # Относительные длины векторов (k0 вынесен за скобку)
    k1_rel = n1
    k2_rel = n2

    # Падающий вектор
    k_Ix = k1_rel * np.cos(theta_rad)
    k_Iz = k1_rel * np.sin(theta_rad)

    # Отражённый вектор
    k_Rx = -k_Ix
    k_Rz = k_Iz

    # Прошедший вектор
    sin_theta_T = n1 / n2 * np.sin(theta_rad)
    theta_T_rad = np.arcsin(sin_theta_T)
    k_Tx = k2_rel * np.cos(theta_T_rad)
    k_Tz = k2_rel * np.sin(theta_T_rad)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_xlim(-2, 2)
    ax.set_ylim(-0.5, 2.5)
    ax.set_aspect('equal')

    # Граница раздела сред
    ax.axhline(y=0, color='black', linewidth=2, label='Граница x=0')
    # Нормаль
    ax.axvline(x=0, color='grey', linestyle='--', linewidth=1, alpha=0.5)

    # Стрелки волновых векторов
    ax.arrow(0, 0, k_Iz, k_Ix, head_width=0.08, head_length=0.12, fc='green', ec='green',
             label=f'k_I (падающий), |k|={k1_rel}', length_includes_head=True)
    ax.arrow(0, 0, k_Rz, k_Rx, head_width=0.08, head_length=0.12, fc='red', ec='red',
             label=f'k_R (отражённый), |k|={k1_rel}', length_includes_head=True)
    ax.arrow(0, 0, k_Tz, k_Tx, head_width=0.08, head_length=0.12, fc='blue', ec='blue',
             label=f'k_T (прошедший), |k|={k2_rel}', length_includes_head=True)

    # Подпись углов
    ax.text(0.3, 0.15, f'θ = {theta}°', fontsize=11, color='green')
    ax.text(1.0, 0.25, f'θ_T ≈ {np.degrees(theta_T_rad):.1f}°', fontsize=11, color='blue')

    ax.set_xlabel('Компонента k_z (отн. ед.)', fontsize=12)
    ax.set_ylabel('Компонента k_x (отн. ед.)', fontsize=12)
    ax.set_title('Рис. 3 — Диаграмма направленности волновых векторов\nв плоскости падения xz (θ = 30°, n₁ = 1.0, n₂ = 1.5)', fontsize=13)
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(True, linestyle=':', alpha=0.5)

    # Среда 1 / Среда 2
    ax.text(-1.8, 1.8, 'Среда 1 (воздух)', fontsize=11, style='italic')
    ax.text(-1.8, -0.3, 'Среда 2 (стекло)', fontsize=11, style='italic')

    plt.tight_layout()
    plt.savefig('wave_vectors_diagram.png', dpi=150)
    plt.show()

if __name__ == "__main__":
    # Попытка чтения данных
    try:
        angles, r_amps, t_amps = read_log()
        plot_amplitude_vs_angle(angles, r_amps, t_amps)
    except FileNotFoundError:
        print("Файл output_log.csv не найден. Строим только диаграмму волновых векторов.")
    plot_wave_vectors_diagram()
