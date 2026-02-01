#Моделирование радиояркостной карты
import numpy as np
import cv2


# sigma - радиус пространственной корреляции(чем больше - тем "гладче" местность)
def generate_base_map(size=(512, 512), sigma=10):
    noise = np.random.randn(*size)
    smooth = cv2.GaussianBlur(noise, (0, 0), sigma)
    return smooth

#--------------------------------------------------
#ПРИРОДНЫЕ ОБЪЕКТЫ
# функция добавления рек(извилистые линии) lakes-параметр впадания реки в озеро
def add_rivers(rmap, num_rivers=2, lakes=None):
    """
    rivers go across the map or flow into lakes
    lakes: list of (x, y, radius) or None
    """
    h, w = rmap.shape
    
    for river_idx in range(num_rivers):
        # старт на границе карты
        side = np.random.choice(["top", "bottom", "left", "right"])
        
        # Начальная точка
        if side == "top":
            x, y = np.random.randint(50, w-50), 0
            direction = np.array([np.random.uniform(-0.3, 0.3), 1.0])  # немного влево/вправо, но в основном вниз
        elif side == "bottom":
            x, y = np.random.randint(50, w-50), h - 1
            direction = np.array([np.random.uniform(-0.3, 0.3), -1.0])  # вверх
        elif side == "left":
            x, y = 0, np.random.randint(50, h-50)
            direction = np.array([1.0, np.random.uniform(-0.3, 0.3)])  # вправо
        else:  # right
            x, y = w - 1, np.random.randint(50, h-50)
            direction = np.array([-1.0, np.random.uniform(-0.3, 0.3)])  # влево
        
        # Нормализуем направление
        direction = direction / np.linalg.norm(direction)
        direction = direction.astype(np.float32)
        
        # Цель
        target = None
        if lakes is not None and len(lakes) > 0 and np.random.rand() < 0.5:
            # Впадает в озеро
            lx, ly, lr = lakes[np.random.randint(len(lakes))]
            target = np.array([lx, ly], dtype=np.float32)
        else:
            # Течет к противоположной или соседней стороне
            if np.random.rand() < 0.7:
                # Противоположная сторона
                if side == "top":
                    target = np.array([np.random.randint(w//4, 3*w//4), h-1], dtype=np.float32)
                elif side == "bottom":
                    target = np.array([np.random.randint(w//4, 3*w//4), 0], dtype=np.float32)
                elif side == "left":
                    target = np.array([w-1, np.random.randint(h//4, 3*h//4)], dtype=np.float32)
                else:  # right
                    target = np.array([0, np.random.randint(h//4, 3*h//4)], dtype=np.float32)
            else:
                # Соседняя сторона (поворот)
                if side == "top":
                    if np.random.rand() < 0.5:
                        target = np.array([w-1, np.random.randint(h//2, 3*h//4)], dtype=np.float32)  # правая
                    else:
                        target = np.array([0, np.random.randint(h//2, 3*h//4)], dtype=np.float32)    # левая
                elif side == "bottom":
                    if np.random.rand() < 0.5:
                        target = np.array([w-1, np.random.randint(h//4, h//2)], dtype=np.float32)
                    else:
                        target = np.array([0, np.random.randint(h//4, h//2)], dtype=np.float32)
                elif side == "left":
                    if np.random.rand() < 0.5:
                        target = np.array([np.random.randint(w//2, 3*w//4), h-1], dtype=np.float32)  # низ
                    else:
                        target = np.array([np.random.randint(w//2, 3*w//4), 0], dtype=np.float32)    # верх
                else:  # right
                    if np.random.rand() < 0.5:
                        target = np.array([np.random.randint(w//4, w//2), h-1], dtype=np.float32)
                    else:
                        target = np.array([np.random.randint(w//4, w//2), 0], dtype=np.float32)
        
        points = []
        pos = np.array([x, y], dtype=np.float32)
        
        max_steps = 1500
        reached_target = False
        
        for step in range(max_steps):
            # Сохраняем точку
            px, py = int(pos[0]), int(pos[1])
            points.append((px, py))
            
            # Проверяем достижение цели (озера)
            if target is not None and lakes is not None:
                # Ищем ближайшее озеро
                for lx, ly, lr in lakes:
                    dist_to_lake = np.sqrt((pos[0] - lx)**2 + (pos[1] - ly)**2)
                    if dist_to_lake < lr + 15:
                        reached_target = True
                        break
                if reached_target:
                    break
            
            # Проверяем достижение границы (если цель - граница)
            if target is not None and lakes is None:
                dist_to_target = np.linalg.norm(pos - target)
                if dist_to_target < 30:
                    reached_target = True
                    break
            
            # Корректируем направление к цели (если есть)
            if target is not None:
                to_target = target - pos
                dist_to_target = np.linalg.norm(to_target)
                
                if dist_to_target > 0:
                    to_target_normalized = to_target / dist_to_target
                    # Сила притяжения зависит от расстояния
                    strength = 0.15 if dist_to_target > 100 else 0.3
                    direction = (1 - strength) * direction + strength * to_target_normalized
            
            # Добавляем извилины
            meander = np.random.normal(0, 0.25, size=2)
            direction = direction + meander
            
            # Нормализуем направление
            dir_norm = np.linalg.norm(direction)
            if dir_norm > 0:
                direction = direction / dir_norm
            
            # Двигаемся вперед
            pos += direction * 2.0
            
            # Проверяем границы
            if pos[0] < 0 or pos[0] >= w or pos[1] < 0 or pos[1] >= h:
                # Вышли за границу - добавляем последнюю точку на границе
                pos[0] = max(0, min(pos[0], w-1))
                pos[1] = max(0, min(pos[1], h-1))
                points.append((int(pos[0]), int(pos[1])))
                break
        
        # Убираем дубликаты точек
        unique_points = []
        for point in points:
            if len(unique_points) == 0 or point != unique_points[-1]:
                unique_points.append(point)
        
        # Рисуем реку (только если есть достаточно точек)
        if len(unique_points) > 20:
            # Рисуем основное русло
            for i in range(len(unique_points) - 1):
                # Толщина уменьшается к концу
                progress = i / len(unique_points)
                thickness = int(10 * (1 - progress * 0.5))
                thickness = max(3, thickness)
                
                cv2.line(
                    rmap,
                    unique_points[i],
                    unique_points[i + 1],
                    color=-2.5,
                    thickness=thickness,
                    lineType=cv2.LINE_AA  # сглаженные линии
                )
            
            # Добавляем ответвления (только для длинных рек)
            if len(unique_points) > 100 and np.random.rand() < 0.7:
                num_branches = np.random.randint(1, 4)
                for _ in range(num_branches):
                    # Выбираем случайную точку в средней части реки
                    branch_start_idx = np.random.randint(len(unique_points)//3, 2*len(unique_points)//3)
                    start_point = unique_points[branch_start_idx]
                    
                    # Направление ответвления
                    branch_dir = np.array([
                        np.random.uniform(-1, 1),
                        np.random.uniform(-1, 1)
                    ])
                    branch_dir = branch_dir / np.linalg.norm(branch_dir)
                    
                    # Рисуем ответвление
                    branch_points = []
                    branch_pos = np.array(start_point, dtype=np.float32)
                    branch_length = np.random.randint(30, 80)
                    
                    for _ in range(branch_length):
                        branch_points.append((int(branch_pos[0]), int(branch_pos[1])))
                        
                        # Немного меняем направление
                        branch_dir = branch_dir + np.random.normal(0, 0.2, size=2)
                        branch_dir = branch_dir / (np.linalg.norm(branch_dir) + 1e-6)
                        
                        branch_pos += branch_dir * 1.5
                        
                        # Проверяем границы
                        if (branch_pos[0] < 0 or branch_pos[0] >= w or 
                            branch_pos[1] < 0 or branch_pos[1] >= h):
                            break
                    
                    # Рисуем линии ответвления
                    if len(branch_points) > 2:
                        for j in range(len(branch_points) - 1):
                            cv2.line(
                                rmap,
                                branch_points[j],
                                branch_points[j + 1],
                                color=-2.2,  # немного светлее
                                thickness=max(1, thickness//2),
                                lineType=cv2.LINE_AA
                            )
    return rmap


# функция добавления озер(случайные замкнутые фигуры)
def add_lakes(rmap, num_lakes=3):
    h, w = rmap.shape

    for _ in range(num_lakes):
        center = (np.random.randint(0, w), np.random.randint(0, h))
        num_points = np.random.randint(6, 12)

        angles = np.linspace(0, 2*np.pi, num_points)
        radii = np.random.randint(30, 80, size=num_points)

        pts = []
        for a, r in zip(angles, radii):
            x = int(center[0] + r * np.cos(a))
            y = int(center[1] + r * np.sin(a))
            pts.append([np.clip(x, 0, w-1), np.clip(y, 0, h-1)])

        pts = np.array([pts], dtype=np.int32)
        cv2.fillPoly(rmap, pts, color=-3.0)
    return rmap

# функция добавления лесов(эллиптические области с текстурой)
def add_forests(rmap, num_forests=6):
    h, w = rmap.shape

    for _ in range(num_forests):
        # --- центр леса ---
        cx = np.random.randint(0, w)
        cy = np.random.randint(0, h)

        # --- характерный размер (НЕ зависит от карты) ---
        base_radius = np.random.randint(40, 90)

        # --- форма: нерегулярный полигон ---
        num_points = np.random.randint(8, 16)
        angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
        radii = base_radius + np.random.randint(-15, 15, size=num_points)

        pts = []
        for a, r in zip(angles, radii):
            x = int(cx + r * np.cos(a))
            y = int(cy + r * np.sin(a))
            pts.append([
                np.clip(x, 0, w - 1),
                np.clip(y, 0, h - 1)
            ])

        pts = np.array([pts], dtype=np.int32)

        # --- маска леса ---
        mask = np.zeros_like(rmap, dtype=np.float32)
        cv2.fillPoly(mask, pts, 1.0)

        # --- сглаживание краёв ---
        mask = cv2.GaussianBlur(
            mask,
            (0, 0),
            sigmaX=6
        )

        # --- внутренняя текстура ---
        texture = np.random.normal(
            loc=0.6,
            scale=0.12,
            size=rmap.shape
        ).astype(np.float32)

        rmap += mask * texture
    return rmap

# функция добавления гор(эллипсоидные возвышенности)
def add_mountains(rmap, num_mountains=3):
    h, w = rmap.shape

    for _ in range(num_mountains):
        cx = np.random.randint(0, w)
        cy = np.random.randint(0, h)

        cluster = np.zeros_like(rmap)

        num_peaks = np.random.randint(3, 7)

        for _ in range(num_peaks):
            px = cx + np.random.randint(-80, 280)
            py = cy + np.random.randint(-80, 280)
            rx = np.random.randint(40, 120)
            ry = np.random.randint(40, 120)
            height = np.random.uniform(1.5, 3.5)

            y, x = np.ogrid[:h, :w]
            peak = np.exp(
                -(((x - px) / rx) ** 2 + ((y - py) / ry) ** 2)
            ) * height

            cluster += peak

        roughness = cv2.GaussianBlur(
            np.random.randn(h, w).astype(np.float32),
            (0, 0),
            sigmaX=10
        )

        rmap += cluster + 0.3 * roughness
    return rmap

#КОНЕЦ ПРИРОДНЫХ ОБЪЕКТОВ
#--------------------------------------------------


#--------------------------------------------------
#АНТРОПОГЕННЫЕ ОБЪЕКТЫ

# дороги(извилистые линии с повышенной радиояркостью в отличие от рек)
def add_roads(rmap, num_highways=2, num_local_roads=20):
    h, w = rmap.shape

    highways = []

    # ---------- МАГИСТРАЛИ ----------
    for _ in range(num_highways):

        # старт на границе
        side = np.random.choice(["top", "bottom", "left", "right"])

        if side in ["top", "bottom"]:
            x = np.random.randint(0, w)
            y = 0 if side == "top" else h - 1
            direction = np.array([0, 1 if side == "top" else -1])
        else:
            y = np.random.randint(0, h)
            x = 0 if side == "left" else w - 1
            direction = np.array([1 if side == "left" else -1, 0])

        direction = direction.astype(np.float32)
        pos = np.array([x, y], dtype=np.float32)

        points = [tuple(pos.astype(int))]

        for _ in range(1200):

            # редкие повороты
            if np.random.rand() < 0.02:
                angle = np.random.choice([-30, -15, 15, 30])
                theta = np.deg2rad(angle)
                rot = np.array([
                    [np.cos(theta), -np.sin(theta)],
                    [np.sin(theta),  np.cos(theta)]
                ])
                direction = rot @ direction

            direction /= (np.linalg.norm(direction) + 1e-6)
            pos += direction * 2.0

            if pos[0] < 0 or pos[0] >= w or pos[1] < 0 or pos[1] >= h:
                break

            points.append(tuple(pos.astype(int)))

        highways.append(points)

        # рисуем магистраль
        for i in range(len(points) - 1):
            cv2.line(rmap, points[i], points[i + 1],
                     color=3.5, thickness=5)

    # ---------- ЛОКАЛЬНЫЕ ДОРОГИ ----------
    for _ in range(num_local_roads):

        # выбираем случайную магистраль
        base = highways[np.random.randint(len(highways))]
        idx = np.random.randint(len(base))
        x, y = base[idx]

        pos = np.array([x, y], dtype=np.float32)

        # ортогональное направление
        if np.random.rand() < 0.5:
            direction = np.array([1, 0])
        else:
            direction = np.array([0, 1])

        if np.random.rand() < 0.5:
            direction *= -1

        length = np.random.randint(50, 200)

        for _ in range(length):
            next_pos = pos + direction

            if (0 <= next_pos[0] < w and 0 <= next_pos[1] < h):
                cv2.line(rmap,
                         tuple(pos.astype(int)),
                         tuple(next_pos.astype(int)),
                         color=3.0,
                         thickness=3)
                pos = next_pos
            else:
                break
    return rmap


# здания(прямоугольные объекты с высокой радиояркостью)
def add_buildings(rmap, num_buildings=40):
    h, w = rmap.shape

    for _ in range(num_buildings):
        x = np.random.randint(0, w - 20)
        y = np.random.randint(0, h - 20)
        bw = np.random.randint(10, 25)
        bh = np.random.randint(10, 25)

        cv2.rectangle(rmap,
                      (x, y),
                      (x + bw, y + bh),
                      color=7.0,
                      thickness=-1)
    return rmap

# транспортные средства(маленькие прямоугольники с очень высокой радиояркостью)
def add_vehicles(rmap, num_vehicles=80):
    h, w = rmap.shape

    for _ in range(num_vehicles):
        x = np.random.randint(0, w - 8)
        y = np.random.randint(0, h - 4)

        cv2.rectangle(rmap,
                      (x, y),
                      (x + 6, y + 3),
                      color=14.0,
                      thickness=-1)
    return rmap


#КОНЕЦ АНТРОПОГЕННЫХ ОБЪЕКТОВ
#--------------------------------------------------


# нормализация радиояркости до физически понятных пределов
def normalize_map(rmap):
    rmap -= rmap.min()
    rmap /= rmap.max()
    return rmap

# !!!финальная сборка карты со всеми объектами!!!
def generate_radiomap(size=(2000, 2000)):
    rmap = generate_base_map(size)

    # природные объекты
    rmap = add_rivers(rmap, num_rivers=1)
    rmap = add_lakes(rmap, num_lakes=2)
    rmap = add_forests(rmap, num_forests=6)
    rmap = add_mountains(rmap, num_mountains=8)
    # антропогенные объекты
    rmap = add_roads(rmap, num_highways=2, num_local_roads=15)
    rmap = add_buildings(rmap, num_buildings=50)
    rmap = add_vehicles(rmap, num_vehicles=25)

    rmap = normalize_map(rmap)
    return rmap

