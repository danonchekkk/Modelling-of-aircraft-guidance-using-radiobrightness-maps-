from radiomap import generate_radiomap
import matplotlib.pyplot as plt

rmap = generate_radiomap((2000, 2000))
x=5
print(f'ingo: {x}')
plt.figure(figsize=(8, 8))
plt.imshow(rmap, cmap='gray')
plt.colorbar(label='Radiobrightness')
plt.title('Simulated Radiobrightness Map')
plt.show()

# 1. поменять поведение генерации дорог
# 2. поменять поведение генерации зданий