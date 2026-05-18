import numpy as np
import csv

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

    # Отраженный луч k_R
    k_Rx = -k_Ix
    k_R = np.array([k_Rx, 0.0, k_Iz])

    # Прошедший луч k_T
    k_Tx_sq = k2**2 - k_Iz**2
    if k_Tx_sq < 0:
        k_Tx = 1j * np.sqrt(-k_Tx_sq)
    else:
        k_Tx = np.sqrt(k_Tx_sq)
    k_T = np.array([k_Tx, 0.0, k_Iz])

    # Коэффициенты для нормального падения (упрощение)
    r_normal = (n1 - n2) / (n1 + n2)
    t_normal = (2 * n1) / (n1 + n2)

    R_amp = A_I * r_normal
    T_amp = A_I * t_normal

    return {
        "k_I": k_I, "k_R": k_R, "k_T": k_T,
        "A_I": A_I, "R": R_amp, "T": T_amp,
        "theta_deg": theta_deg
    }

def log_results(filename, results_list):
    """Запись серии результатов в CSV-файл"""
    fieldnames = ['theta_deg', 'k_Ix', 'k_Iz', 'k_Rx', 'k_Rz', 'k_Tx', 'k_Tz', 'R_amp', 'T_amp']
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for res in results_list:
            writer.writerow({
                'theta_deg': res['theta_deg'],
                'k_Ix': res['k_I'][0], 'k_Iz': res['k_I'][2],
                'k_Rx': res['k_R'][0], 'k_Rz': res['k_R'][2],
                'k_Tx': np.real(res['k_T'][0]), 'k_Tz': np.real(res['k_T'][2]),
                'R_amp': res['R'], 'T_amp': res['T']
            })

if __name__ == "__main__":
    lam = 500e-9
    n1 = 1.0
    n2 = 1.5
    A_init = 1.0

    # Серия расчётов для разных углов
    results = []
    for theta in [0, 15, 30, 45, 60, 75]:
        res = wave_fields(lam, n1, n2, theta, A_init)
        results.append(res)
        print(f"θ = {theta}°: R = {res['R']:.4f}, T = {res['T']:.4f}")

    # Сохранение в CSV
    log_results('output_log.csv', results)
    print("Результаты сохранены в output_log.csv")
