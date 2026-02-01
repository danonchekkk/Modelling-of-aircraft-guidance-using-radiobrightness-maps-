import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random

from radiomap import generate_radiomap
from sensor import radiometer_measurement

#параметры
map_size = (2000, 2000)
pos = (random.randint(0, map_size[0]), random.randint(0, map_size[1])) #случайная позиция ЛА
fov_size = 200 #разрешение с сенсора ЛА

#генерация карты
rmap = generate_radiomap(map_size)

# измерение радиометра
patch = radiometer_measurement(rmap, pos, fov_size=fov_size)

# визуализация
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# ===== ЛЕВАЯ ПАНЕЛЬ: карта =====
ax_map = axes[0]
im_map = ax_map.imshow(rmap, cmap='gray')
ax_map.set_title("Radiobrightness Map")
ax_map.set_xlabel("X")
ax_map.set_ylabel("Y")

# положение ЛА
ax_map.plot(pos[0], pos[1], 'ro', label="Реальные координаты ЛА")

# окно обзора
half = fov_size // 2
rect = patches.Rectangle(
    (pos[0] - half, pos[1] - half),
    fov_size,
    fov_size,
    linewidth=2,
    edgecolor='red',
    facecolor='none',
    label="площадь \"обзора\" ЛА"
)
ax_map.add_patch(rect)

ax_map.legend()
fig.colorbar(im_map, ax=ax_map, fraction=0.046)

# ===== ПРАВАЯ ПАНЕЛЬ: вид с ЛА =====
ax_patch = axes[1]
im_patch = ax_patch.imshow(patch, cmap='gray')
ax_patch.set_title("Radiometer Measurement (FOV)")
ax_patch.set_xlabel("Local X")
ax_patch.set_ylabel("Local Y")
fig.colorbar(im_patch, ax=ax_patch, fraction=0.046)

plt.tight_layout()
plt.show()
