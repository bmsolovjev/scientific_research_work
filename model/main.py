import numpy as np
import csv

def fresnel_coefficients(n1, n2, theta_rad):
    """
    Вычисление амплитудных коэффициентов Френеля для s- и p-поляризаций.
    Корректно обрабатывает нормальное падение (theta = 0).
    """
    # Проверка на нормальное падение
    if np.isclose(theta_rad, 0.0, atol=1e-10):
        # При нормальном падении s- и p-поляризации неразличимы
        r_s = (n1 - n2) / (n1 + n2)
        r_p = r_s
        t_s = (2 * n1) / (n1 + n2)
        t_p = t_s
        cos_theta_T = 1.0
        return r_s, r_p, t_s, t_p, cos_theta_T
    
    # Закон Снеллиуса для прошедшего угла
    sin_theta_T = n1 / n2 * np.sin(theta_rad)
    
    # Проверка на полное внутреннее отражение
    if sin_theta_T >= 1.0:
        # ПВО: прошедшая волна затухающая, косинус мнимый
        cos_theta_T = 1j * np.sqrt(sin_theta_T**2 - 1)
    else:
        cos_theta_T = np.sqrt(1 - sin_theta_T**2)
    
    cos_theta_I = np.cos(theta_rad)
    
    # Формулы Френеля для s-поляризации (TE)
    r_s = (n1 * cos_theta_I - n2 * cos_theta_T) / (n1 * cos_theta_I + n2 * cos_theta_T)
    t_s = (2 * n1 * cos_theta_I) / (n1 * cos_theta_I + n2 * cos_theta_T)
    
    # Формулы Френеля для p-поляризации (TM)
    r_p = (n2 * cos_theta_I - n1 * cos_theta_T) / (n2 * cos_theta_I + n1 * cos_theta_T)
    t_p = (2 * n1 * cos_theta_I) / (n2 * cos_theta_I + n1 * cos_theta_T)
    
    return r_s, r_p, t_s, t_p, cos_theta_T


def wave_fields(lambda_val, n1, n2, theta_deg, A_I):
    """
    Расчет отраженных и прошедших полей на границе x=0.
    x < 0: среда 1 (n1) - свет падает отсюда
    x > 0: среда 2 (n2) - свет проходит сюда
    Положительная ось X направлена вниз (в среду 2).
    Положительная ось Z направлена вправо.
    """
    theta_rad = np.radians(theta_deg)
    k0 = 2 * np.pi / lambda_val
    k1 = k0 * n1
    k2 = k0 * n2

    # Падающий луч k_I (в плоскости xz, y=0)
    k_Ix = k1 * np.cos(theta_rad)
    k_Iy = 0.0
    k_Iz = k1 * np.sin(theta_rad)

    k_I = np.array([k_Ix, k_Iy, k_Iz])

    # Закон Снеллиуса (k_z)
    k_Rz = k_Iz
    k_Tz = k_Iz

    # Отраженный луч k_R (x < 0, движется в отрицательную x)
    k_Rx = -k_Ix
    k_R = np.array([k_Rx, 0.0, k_Rz])

    # Прошедший луч k_T (x > 0)
    k_Tx_sq = k2**2 - k_Tz**2
    if k_Tx_sq < 0:
        k_Tx = 1j * np.sqrt(-k_Tx_sq)
    else:
        k_Tx = np.sqrt(k_Tx_sq)

    k_T = np.array([k_Tx, 0.0, k_Tz])

    # Вычисление коэффициентов Френеля
    r_s, r_p, t_s, t_p, cos_theta_T = fresnel_coefficients(n1, n2, theta_rad)

    # Амплитуды отражённой и прошедшей волн (s-поляризация)
    R_s_amp = A_I * r_s
    T_s_amp = A_I * t_s

    # Амплитуды отражённой и прошедшей волн (p-поляризация)
    R_p_amp = A_I * r_p
    T_p_amp = A_I * t_p

    return {
        "k_I": k_I, "k_R": k_R, "k_T": k_T,
        "A_I": A_I,
        # s-поляризация
        "r_s": r_s, "t_s": t_s,
        "R_s_amp": R_s_amp, "T_s_amp": T_s_amp,
        # p-поляризация
        "r_p": r_p, "t_p": t_p,
        "R_p_amp": R_p_amp, "T_p_amp": T_p_amp,
        # Углы
        "theta_deg": theta_deg,
        "theta_T_rad": np.arcsin(n1 / n2 * np.sin(theta_rad)) if n1 / n2 * np.sin(theta_rad) < 1 else None
    }


def log_results(filename, results_list):
    """Запись серии результатов в CSV-файл"""
    fieldnames = [
        'theta_deg',
        # Падающий вектор k_I (все 3 компоненты)
        'k_Ix', 'k_Iy', 'k_Iz',
        # Отражённый вектор k_R (все 3 компоненты)
        'k_Rx', 'k_Ry', 'k_Rz',
        # Прошедший вектор k_T (все 3 компоненты)
        'k_Tx', 'k_Ty', 'k_Tz',
        # Коэффициенты Френеля
        'r_s', 't_s', 'R_s_amp', 'T_s_amp',
        'r_p', 't_p', 'R_p_amp', 'T_p_amp'
    ]
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for res in results_list:
            writer.writerow({
                'theta_deg': res['theta_deg'],
                # k_I
                'k_Ix': res['k_I'][0], 'k_Iy': res['k_I'][1], 'k_Iz': res['k_I'][2],
                # k_R
                'k_Rx': res['k_R'][0], 'k_Ry': res['k_R'][1], 'k_Rz': res['k_R'][2],
                # k_T
                'k_Tx': np.real(res['k_T'][0]), 'k_Ty': np.real(res['k_T'][1]), 'k_Tz': np.real(res['k_T'][2]),
                # Коэффициенты
                'r_s': np.real(res['r_s']), 't_s': np.real(res['t_s']),
                'R_s_amp': res['R_s_amp'], 'T_s_amp': res['T_s_amp'],
                'r_p': np.real(res['r_p']), 't_p': np.real(res['t_p']),
                'R_p_amp': res['R_p_amp'], 'T_p_amp': res['T_p_amp']
            })


if __name__ == "__main__":
    # Параметры
    lam = 500e-9
    n1 = 1.0
    n2 = 1.5
    A_init = 1.0

    print("=" * 70)
    print("ТЕСТИРОВАНИЕ ПРОТОТИПА С ФОРМУЛАМИ ФРЕНЕЛЯ")
    print("=" * 70)

    # Тест 1: Нормальное падение
    print("\n--- Тест 1: Нормальное падение (θ = 0°) ---")
    res = wave_fields(lam, n1, n2, 0.0, A_init)
    print(f"k_I: {res['k_I']}")
    print(f"k_R: {res['k_R']}")
    print(f"k_T: {res['k_T']}")
    print(f"s-поляризация: r_s = {res['r_s']:.4f}, t_s = {res['t_s']:.4f}")
    print(f"p-поляризация: r_p = {res['r_p']:.4f}, t_p = {res['t_p']:.4f}")
    print(f"Амплитуда R_s: {res['R_s_amp']:.4f}, Амплитуда T_s: {res['T_s_amp']:.4f}")

    # Тест 2: Наклонное падение 30°
    print("\n--- Тест 2: Наклонное падение (θ = 30°) ---")
    res = wave_fields(lam, n1, n2, 30.0, A_init)
    print(f"k_I: {res['k_I']}")
    print(f"k_R: {res['k_R']}")
    print(f"k_T: {res['k_T']}")
    print(f"s-поляризация: r_s = {res['r_s']:.4f}, t_s = {res['t_s']:.4f}")
    print(f"p-поляризация: r_p = {res['r_p']:.4f}, t_p = {res['t_p']:.4f}")
    print(f"Амплитуда R_s: {res['R_s_amp']:.4f}, Амплитуда T_s: {res['T_s_amp']:.4f}")
    print(f"Амплитуда R_p: {res['R_p_amp']:.4f}, Амплитуда T_p: {res['T_p_amp']:.4f}")

    # Тест 3: Угол Брюстера
    theta_B = np.degrees(np.arctan(n2 / n1))
    print(f"\n--- Тест 3: Угол Брюстера (θ ≈ {theta_B:.1f}°) ---")
    res = wave_fields(lam, n1, n2, theta_B, A_init)
    print(f"s-поляризация: r_s = {res['r_s']:.4f}")
    print(f"p-поляризация: r_p = {res['r_p']:.6f} (должен быть ~0)")

    # Серия расчётов для CSV
    print("\n--- Сохранение серии расчётов в CSV ---")
    results = []
    for theta in [0, 15, 30, 45, 60, 75, 85]:
        res = wave_fields(lam, n1, n2, theta, A_init)
        results.append(res)
        print(f"θ = {theta:2d}°: r_s = {res['r_s']:.4f}, t_s = {res['t_s']:.4f}, "
              f"r_p = {res['r_p']:.4f}, t_p = {res['t_p']:.4f}")

    log_results('output_log.csv', results)
    print("\nРезультаты сохранены в output_log.csv")
    print("=" * 70)
