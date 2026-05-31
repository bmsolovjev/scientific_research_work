import numpy as np
import csv

def fresnel_coefficients(n1, n2, theta_rad):
    """
    Вычисление амплитудных коэффициентов Френеля для s- и p-поляризаций.
    Возвращает также cos_theta_T и коэффициенты интенсивности R, T.
    """
    # Проверка на нормальное падение
    if np.isclose(theta_rad, 0.0, atol=1e-10):
        r_s = (n1 - n2) / (n1 + n2)
        r_p = r_s
        t_s = (2 * n1) / (n1 + n2)
        t_p = t_s
        cos_theta_T = 1.0
        sin_theta_T = 0.0
        is_tir = False
    else:
        # Закон Снеллиуса для прошедшего угла
        sin_theta_T = n1 / n2 * np.sin(theta_rad)
        
        # Проверка на полное внутреннее отражение
        if sin_theta_T >= 1.0:
            is_tir = True
            cos_theta_T = 1j * np.sqrt(sin_theta_T**2 - 1)
        else:
            is_tir = False
            cos_theta_T = np.sqrt(1 - sin_theta_T**2)
    
    cos_theta_I = np.cos(theta_rad)
    
    # Формулы Френеля для s-поляризации (TE)
    denom_s = n1 * cos_theta_I + n2 * cos_theta_T
    if denom_s == 0:
        r_s = 1.0
        t_s = 0.0
    else:
        r_s = (n1 * cos_theta_I - n2 * cos_theta_T) / denom_s
        t_s = (2 * n1 * cos_theta_I) / denom_s
    
    # Формулы Френеля для p-поляризации (TM)
    denom_p = n2 * cos_theta_I + n1 * cos_theta_T
    if denom_p == 0:
        r_p = 1.0
        t_p = 0.0
    else:
        r_p = (n2 * cos_theta_I - n1 * cos_theta_T) / denom_p
        t_p = (2 * n1 * cos_theta_I) / denom_p
    
    # Энергетические коэффициенты (интенсивность)
    R_s = np.abs(r_s)**2
    R_p = np.abs(r_p)**2
    T_s = (n2 * np.real(cos_theta_T) / (n1 * cos_theta_I)) * np.abs(t_s)**2 if not is_tir else 0.0
    T_p = (n2 * np.real(cos_theta_T) / (n1 * cos_theta_I)) * np.abs(t_p)**2 if not is_tir else 0.0
    
    # Для ПВО: R = 1, T = 0
    if is_tir:
        R_s = 1.0
        R_p = 1.0
        T_s = 0.0
        T_p = 0.0
    
    return {
        'r_s': r_s, 'r_p': r_p,
        't_s': t_s, 't_p': t_p,
        'R_s': R_s, 'R_p': R_p,
        'T_s': T_s, 'T_p': T_p,
        'cos_theta_T': cos_theta_T,
        'sin_theta_T': sin_theta_T,
        'is_tir': is_tir
    }


def calculate_e_field(amplitude, k_vector, r, polarization_type, theta_rad, n):
    """
    Расчёт вектора электрического поля E для плоской волны.
    
    Параметры:
    - amplitude: комплексная амплитуда волны
    - k_vector: волновой вектор (kx, ky, kz)
    - r: радиус-вектор точки наблюдения (по умолчанию (0,0,0))
    - polarization_type: 's' (TE) или 'p' (TM)
    - theta_rad: угол падения (для определения направления поля)
    - n: показатель преломления среды
    """
    # считаем поле в начале координат r = (0,0,0)
    # Фаза exp(i·k·r) = 1
    
    kx, ky, kz = k_vector
    k_norm = np.sqrt(kx**2 + ky**2 + kz**2)
    
    # Единичный вектор направления распространения
    if k_norm > 0:
        direction = np.array([kx, ky, kz]) / k_norm
    else:
        direction = np.array([0, 0, 0])
    
    # Вектор поляризации зависит от типа поляризации
    if polarization_type == 's':
        # s-поляризация (TE): E перпендикулярен плоскости падения (ось Y)
        # Плоскость падения — XZ, поэтому E направлен по Y
        e_vector = np.array([0.0, 1.0, 0.0])
    else:  # 'p'
        # p-поляризация (TM): E лежит в плоскости падения (XZ)
        # E перпендикулярен направлению распространения и лежит в плоскости XZ
        # Нормализованный вектор в плоскости XZ, перпендикулярный direction
        if np.abs(direction[0]) > 1e-10 or np.abs(direction[2]) > 1e-10:
            # Вектор в плоскости XZ, перпендикулярный direction
            e_vector = np.array([direction[2], 0.0, -direction[0]])
            e_vector = e_vector / np.sqrt(e_vector[0]**2 + e_vector[2]**2)
        else:
            e_vector = np.array([1.0, 0.0, 0.0])
    
    # Комплексная амплитуда поля
    E_complex = amplitude * e_vector
    
    return {
        'E_vector': E_complex,
        'amplitude': amplitude,
        'polarization': e_vector,
        'direction': direction
    }


def wave_fields(lambda_val, n1, n2, theta_deg, A_I):
    """
    Расчет отраженных и прошедших полей на границе x=0.
    x < 0: среда 1 (n1) - свет падает отсюда
    x > 0: среда 2 (n2) - свет проходит сюда
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
        k_Tx = -1j * np.sqrt(-k_Tx_sq)  # затухающая волна
    else:
        k_Tx = np.sqrt(k_Tx_sq)
    k_T = np.array([k_Tx, 0.0, k_Tz])

    # Вычисление коэффициентов Френеля
    fresnel = fresnel_coefficients(n1, n2, theta_rad)
    
    r_s = fresnel['r_s']
    r_p = fresnel['r_p']
    t_s = fresnel['t_s']
    t_p = fresnel['t_p']
    R_s = fresnel['R_s']
    R_p = fresnel['R_p']
    T_s = fresnel['T_s']
    T_p = fresnel['T_p']
    is_tir = fresnel['is_tir']
    
    # Амплитуды отражённой и прошедшей волн
    R_s_amp = A_I * r_s
    R_p_amp = A_I * r_p
    T_s_amp = A_I * t_s
    T_p_amp = A_I * t_p
    
    # Расчёт векторов E для падающей, отражённой и прошедшей волн
    # Падающая волна (s-поляризация)
    E_inc_s = calculate_e_field(A_I, k_I, 's', theta_rad, n1)
    # Падающая волна (p-поляризация)
    E_inc_p = calculate_e_field(A_I, k_I, 'p', theta_rad, n1)
    
    # Отражённая волна (s-поляризация)
    E_ref_s = calculate_e_field(R_s_amp, k_R, 's', theta_rad, n1)
    # Отражённая волна (p-поляризация)
    E_ref_p = calculate_e_field(R_p_amp, k_R, 'p', theta_rad, n1)
    
    # Прошедшая волна (s-поляризация)
    E_trans_s = calculate_e_field(T_s_amp, k_T, 's', theta_rad, n2)
    # Прошедшая волна (p-поляризация)
    E_trans_p = calculate_e_field(T_p_amp, k_T, 'p', theta_rad, n2)
    
    # Проверка закона сохранения энергии
    energy_s = R_s + T_s
    energy_p = R_p + T_p
    energy_conservation_s = np.isclose(energy_s, 1.0, atol=1e-10) or is_tir
    energy_conservation_p = np.isclose(energy_p, 1.0, atol=1e-10) or is_tir
    
    # Угол преломления (если нет ПВО)
    if not is_tir:
        theta_T_rad = np.arcsin(n1 / n2 * np.sin(theta_rad)) if n1 / n2 * np.sin(theta_rad) < 1 else np.pi/2
        theta_T_deg = np.degrees(theta_T_rad)
    else:
        theta_T_rad = None
        theta_T_deg = None

    return {
        # Волновые векторы
        "k_I": k_I, "k_R": k_R, "k_T": k_T,
        "A_I": A_I,
        # Углы
        "theta_deg": theta_deg,
        "theta_T_deg": theta_T_deg,
        "is_tir": is_tir,
        # s-поляризация
        "r_s": r_s, "t_s": t_s,
        "R_s": R_s, "T_s": T_s,
        "R_s_amp": R_s_amp, "T_s_amp": T_s_amp,
        "E_inc_s": E_inc_s,
        "E_ref_s": E_ref_s,
        "E_trans_s": E_trans_s,
        # p-поляризация
        "r_p": r_p, "t_p": t_p,
        "R_p": R_p, "T_p": T_p,
        "R_p_amp": R_p_amp, "T_p_amp": T_p_amp,
        "E_inc_p": E_inc_p,
        "E_ref_p": E_ref_p,
        "E_trans_p": E_trans_p,
        # Проверка энергии
        "energy_s": energy_s,
        "energy_p": energy_p,
        "energy_conservation_s": energy_conservation_s,
        "energy_conservation_p": energy_conservation_p
    }


def log_results(filename, results_list):
    """Запись серии результатов в CSV-файл"""
    fieldnames = [
        'theta_deg', 'theta_T_deg', 'is_tir',
        # Волновые векторы
        'k_Ix', 'k_Iy', 'k_Iz',
        'k_Rx', 'k_Ry', 'k_Rz',
        'k_Tx', 'k_Ty', 'k_Tz',
        # s-поляризация
        'r_s', 't_s', 'R_s', 'T_s', 'energy_s', 'energy_conservation_s',
        # p-поляризация
        'r_p', 't_p', 'R_p', 'T_p', 'energy_p', 'energy_conservation_p',
        # Амплитуды
        'R_s_amp_re', 'R_s_amp_im', 'T_s_amp_re', 'T_s_amp_im',
        'R_p_amp_re', 'R_p_amp_im', 'T_p_amp_re', 'T_p_amp_im',
        # Векторы E (компоненты)
        'E_inc_s_x', 'E_inc_s_y', 'E_inc_s_z',
        'E_ref_s_x', 'E_ref_s_y', 'E_ref_s_z',
        'E_trans_s_x', 'E_trans_s_y', 'E_trans_s_z',
        'E_inc_p_x', 'E_inc_p_y', 'E_inc_p_z',
        'E_ref_p_x', 'E_ref_p_y', 'E_ref_p_z',
        'E_trans_p_x', 'E_trans_p_y', 'E_trans_p_z'
    ]
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for res in results_list:
            writer.writerow({
                'theta_deg': res['theta_deg'],
                'theta_T_deg': res['theta_T_deg'],
                'is_tir': res['is_tir'],
                # Волновые векторы
                'k_Ix': res['k_I'][0], 'k_Iy': res['k_I'][1], 'k_Iz': res['k_I'][2],
                'k_Rx': res['k_R'][0], 'k_Ry': res['k_R'][1], 'k_Rz': res['k_R'][2],
                'k_Tx': np.real(res['k_T'][0]), 'k_Ty': np.real(res['k_T'][1]), 'k_Tz': np.real(res['k_T'][2]),
                # s-поляризация
                'r_s': np.real(res['r_s']), 't_s': np.real(res['t_s']),
                'R_s': res['R_s'], 'T_s': res['T_s'],
                'energy_s': res['energy_s'],
                'energy_conservation_s': res['energy_conservation_s'],
                # p-поляризация
                'r_p': np.real(res['r_p']), 't_p': np.real(res['t_p']),
                'R_p': res['R_p'], 'T_p': res['T_p'],
                'energy_p': res['energy_p'],
                'energy_conservation_p': res['energy_conservation_p'],
                # Амплитуды
                'R_s_amp_re': np.real(res['R_s_amp']), 'R_s_amp_im': np.imag(res['R_s_amp']),
                'T_s_amp_re': np.real(res['T_s_amp']), 'T_s_amp_im': np.imag(res['T_s_amp']),
                'R_p_amp_re': np.real(res['R_p_amp']), 'R_p_amp_im': np.imag(res['R_p_amp']),
                'T_p_amp_re': np.real(res['T_p_amp']), 'T_p_amp_im': np.imag(res['T_p_amp']),
                # Векторы E (s-поляризация)
                'E_inc_s_x': res['E_inc_s']['E_vector'][0], 'E_inc_s_y': res['E_inc_s']['E_vector'][1], 'E_inc_s_z': res['E_inc_s']['E_vector'][2],
                'E_ref_s_x': res['E_ref_s']['E_vector'][0], 'E_ref_s_y': res['E_ref_s']['E_vector'][1], 'E_ref_s_z': res['E_ref_s']['E_vector'][2],
                'E_trans_s_x': res['E_trans_s']['E_vector'][0], 'E_trans_s_y': res['E_trans_s']['E_vector'][1], 'E_trans_s_z': res['E_trans_s']['E_vector'][2],
                # Векторы E (p-поляризация)
                'E_inc_p_x': res['E_inc_p']['E_vector'][0], 'E_inc_p_y': res['E_inc_p']['E_vector'][1], 'E_inc_p_z': res['E_inc_p']['E_vector'][2],
                'E_ref_p_x': res['E_ref_p']['E_vector'][0], 'E_ref_p_y': res['E_ref_p']['E_vector'][1], 'E_ref_p_z': res['E_ref_p']['E_vector'][2],
                'E_trans_p_x': res['E_trans_p']['E_vector'][0], 'E_trans_p_y': res['E_trans_p']['E_vector'][1], 'E_trans_p_z': res['E_trans_p']['E_vector'][2],
            })


def print_energy_check(res):
    """Красивый вывод проверки закона сохранения энергии"""
    print("-" * 50)
    print(f"Угол падения: {res['theta_deg']}°")
    if res['is_tir']:
        print("Режим полного внутреннего отражения")
    print(f"s-поляризация: R = {res['R_s']:.6f}, T = {res['T_s']:.6f}, R+T = {res['energy_s']:.6f} → {res['energy_conservation_s']}")
    print(f"p-поляризация: R = {res['R_p']:.6f}, T = {res['T_p']:.6f}, R+T = {res['energy_p']:.6f} → {res['energy_conservation_p']}")
    print("-" * 50)


if __name__ == "__main__":
    # Параметры
    lam = 500e-9
    n1 = 1.0
    n2 = 1.5
    A_init = 1.0

    print("=" * 70)
    print("ТЕСТИРОВАНИЕ МОДЕЛИ С ФОРМУЛАМИ ФРЕНЕЛЯ")
    print("Проверка закона сохранения энергии + расчёт вектора E")
    print("=" * 70)

    # Тест 1: Нормальное падение
    print("\n--- Тест 1: Нормальное падение (θ = 0°) ---")
    res = wave_fields(lam, n1, n2, 0.0, A_init)
    print(f"Падающая волна (s): E = {res['E_inc_s']['E_vector']}")
    print(f"Отражённая волна (s): E = {res['E_ref_s']['E_vector']}")
    print(f"Прошедшая волна (s): E = {res['E_trans_s']['E_vector']}")
    print_energy_check(res)

    # Тест 2: Наклонное падение 30°
    print("\n--- Тест 2: Наклонное падение (θ = 30°) ---")
    res = wave_fields(lam, n1, n2, 30.0, A_init)
    print(f"Угол преломления: {res['theta_T_deg']:.2f}°")
    print(f"s-поляризация: r_s = {res['r_s']:.4f}, t_s = {res['t_s']:.4f}")
    print(f"p-поляризация: r_p = {res['r_p']:.4f}, t_p = {res['t_p']:.4f}")
    print_energy_check(res)

    # Тест 3: Угол Брюстера
    theta_B = np.degrees(np.arctan(n2 / n1))
    print(f"\n--- Тест 3: Угол Брюстера (θ ≈ {theta_B:.1f}°) ---")
    res = wave_fields(lam, n1, n2, theta_B, A_init)
    print(f"s-поляризация: r_s = {res['r_s']:.4f}")
    print(f"p-поляризация: r_p = {res['r_p']:.8f} (должен быть ~0)")
    print_energy_check(res)

    # Тест 4: Полное внутреннее отражение (обратный ход)
    print("\n--- Тест 4: Полное внутреннее отражение (стекло → воздух, θ = 60°) ---")
    res_tir = wave_fields(lam, n2, n1, 60.0, A_init)
    print(f"Падение из стекла (n={n2}) в воздух (n={n1})")
    print(f"Критический угол: {np.degrees(np.arcsin(n1/n2)):.1f}°")
    print_energy_check(res_tir)

    # Серия расчётов для CSV
    print("\n--- Сохранение серии расчётов в CSV ---")
    results = []
    print("\nРезультаты проверки энергии для разных углов:")
    for theta in [0, 15, 30, 45, 60, 75, 85]:
        res = wave_fields(lam, n1, n2, theta, A_init)
        results.append(res)
        status_s = "✓" if res['energy_conservation_s'] else "✗"
        status_p = "✓" if res['energy_conservation_p'] else "✗"
        print(f"θ = {theta:2d}°: R_s+T_s = {res['energy_s']:.6f} {status_s}, "
              f"R_p+T_p = {res['energy_p']:.6f} {status_p}")

    log_results('output_log.csv', results)
    print("\nРезультаты сохранены в output_log.csv")
    print("=" * 70)
