import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import DBSCAN
from pyransac3d import Plane

# Funkcja do wczytywania pliku XYZ
def load_xyz_file(filename):
    points = []
    with open(filename, 'r') as f:
        for line in f:
            x, y, z = map(float, line.split())
            points.append([x, y, z])
    return np.array(points)

# Funkcja do wyświetlania chmur punktów w przestrzeni 3D
def plot_clusters(clusters):
    fig = plt.figure(figsize=(15, 5))
    for i, cluster in enumerate(clusters):
        ax = fig.add_subplot(1, len(clusters), i+1, projection='3d')
        ax.scatter(cluster[:, 0], cluster[:, 1], cluster[:, 2], marker='o')
        ax.set_title(f'Chmura punktów Klastra {i+1}')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
    plt.show()

# Wczytanie punktów z plików XYZ
poziome = load_xyz_file("poziome.xyz")
pionowe = load_xyz_file("pionowe.xyz")
cylindryczne = load_xyz_file("cylindryczne.xyz")

# Połączenie punktów w jedną macierz
all_points = np.concatenate((poziome, pionowe, cylindryczne))

# Zastosowanie DBSCAN do znajdowania rozłącznych chmur punktów
dbscan = DBSCAN(eps=0.2, min_samples=10)
labels = dbscan.fit_predict(all_points)
unique_labels = np.unique(labels)

# Podział punktów na klastry na podstawie etykiet DBSCAN
clusters = [all_points[labels == label] for label in unique_labels if label != -1]

# Wyświetlenie chmur punktów
plot_clusters(clusters)

# Dopasowanie płaszczyzny dla każdej chmury punktów za pomocą RANSAC
for i, cluster in enumerate(clusters):
    plane = Plane()
    plane.fit(cluster)
    plane_normal = plane.params[:3]
    inliers = plane.inliers
    if len(inliers) > 0:
        print(f"Współrzędne wektora normalnego płaszczyzny dla klastra {i+1}: {plane_normal}")
        print(f"Średnia odległość punktów do płaszczyzny dla klastra {i+1}: {plane.residuals.mean()}")
        if np.abs(plane_normal[2]) > np.abs(plane_normal[0]) and np.abs(plane_normal[2]) > np.abs(plane_normal[1]):
            print("Płaszczyzna jest pionowa")
        elif np.abs(plane_normal[0]) > np.abs(plane_normal[1]) and np.abs(plane_normal[0]) > np.abs(plane_normal[2]):
            print("Płaszczyzna jest pozioma")
        else:
            print("Płaszczyzna nie jest ani pozioma, ani pionowa")
    else:
        print(f"Nie udało się dopasować płaszczyzny dla klastra {i+1}.")
