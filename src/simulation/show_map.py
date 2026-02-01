from radiomap import generate_radiomap
import matplotlib.pyplot as plt

rmap = generate_radiomap((2000, 2000))

plt.figure(figsize=(8, 8))
plt.imshow(rmap, cmap='gray')
plt.colorbar(label='Radiobrightness')
plt.title('Simulated Radiobrightness Map')
plt.show()