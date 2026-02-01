"""
Радиояркостная карта - это «истина»,
а радиометрическое измерение — это «то, что реально видит БПЛА/ракета в процессе полета над этой картой»

С полноразмерной карты(со спутника) берется небольшой квадратный участок и летательный аппарат измеряет радиояркость
в каждой точке этого участка с некоторым шумом.

Для каждого положения летательного аппарата:
РЯК(радиояркостная карта) -> окно -> взвешивание -> шум -> измерение

Вход: полная радиояркостная карта - как сфотографировал спутник с орбиты
Выход: набор радиометрических измерений с шумом, полученных летательным аппаратом - как увидел бы летательный
аппарат, летящий в заданном квадрате.
"""

import numpy as np
import cv2

def extract_fov(rmap, center, fov_size=64):
    """
    rmap     : радиояркостная карта
    center   : (x, y) положение летательного аппарата
    fov_size : размер окна радиометра
    """
    x, y = int(center[0]), int(center[1])
    half = fov_size // 2

    h, w = rmap.shape

    x1 = np.clip(x - half, 0, w)
    x2 = np.clip(x + half, 0, w)
    y1 = np.clip(y - half, 0, h)
    y2 = np.clip(y + half, 0, h)

    patch = rmap[y1:y2, x1:x2]

    # дополняем до нужного размера
    patch = cv2.copyMakeBorder(
        patch,
        top=max(0, half - y),
        bottom=max(0, y + half - h),
        left=max(0, half - x),
        right=max(0, x + half - w),
        borderType=cv2.BORDER_REFLECT
    )
    return patch

# диаграмма направленности антенны
# Радиометр сильнее чувствителен к центру окна, чем к краям
def antenna_pattern(size, sigma=0.4):
    """
    size  : размер окна
    sigma : ширина диаграммы направленности
    """
    ax = np.linspace(-1, 1, size)
    xx, yy = np.meshgrid(ax, ax)
    pattern = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return pattern / pattern.sum()

# интегрирование по апертуре
def apply_antenna(patch, pattern):
    return patch * pattern

# добавление шума
def add_radiometric_noise(signal, snr_db=20):
    signal_power = np.mean(signal**2)
    snr = 10**(snr_db / 10)
    noise_power = signal_power / snr

    noise = np.random.normal(
        0,
        np.sqrt(noise_power),
        size=signal.shape
    )
    return signal + noise

# квантование сигнала
def quantize(signal, levels=256):
    signal = np.clip(signal, 0, 1)
    return np.round(signal * (levels - 1)) / (levels - 1)

def normalize_patch(patch, eps=1e-6):
    p_min = patch.min()
    p_max = patch.max()
    return (patch - p_min) / (p_max - p_min + eps)

def radiometer_measurement(
    rmap,
    position,
    fov_size=500,
    snr_db=10 # параметре отвечает за эффект "прожектора" из-за ДН
):
    patch = extract_fov(rmap, position, fov_size)
    pattern = antenna_pattern(fov_size)
    patch = apply_antenna(patch, pattern)
    patch = add_radiometric_noise(patch, snr_db)
    patch = normalize_patch(patch)
    patch = quantize(patch)

    return patch

