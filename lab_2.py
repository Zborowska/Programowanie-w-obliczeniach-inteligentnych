import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def load_xyz_file(filename):
    points = []
    with open(filename, 'r') as f:
        for line in f:
            x, y, z = map(float, line.split())
            points.append([x, y, z])
    return np.array(points)


def kmeans(points, k=3, max_iter=100):
    centroids = points[np.random.choice(points.shape[0], k, replace=False)]
    for _ in range(max_iter):
        distances = np.sqrt(((points - centroids[:, np.newaxis]) ** 2).sum(axis=2))
        labels = np.argmin(distances, axis=0)
        new_centroids = np.array([points[labels == i].mean(axis=0) for i in range(k)])
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids
    return labels, centroids


def fit_plane_ransac(points, n_iterations=1000, threshold_distance=0.1):
    best_plane = None
    best_inliers = None
    best_error = np.inf

    for _ in range(n_iterations):
        while True:
            # Wybierz losowe trzy punkty
            indices = np.random.choice(points.shape[0], 3, replace=False)
            random_points = points[indices]

            # Dopasuj płaszczyznę do wybranych punktów
            try:
                plane = np.linalg.solve(random_points, np.ones(3))
            except np.linalg.LinAlgError:
                # Jeśli macierz jest osobliwa, spróbuj ponownie
                continue
            break

        # Oblicz odległość punktów do płaszczyzny
        distances = np.abs(points.dot(plane) + plane[-1]) / np.linalg.norm(plane, keepdims=True)

        # Znajdź punkty, które są blisko płaszczyzny (inliers)
        inliers = points[distances < threshold_distance]

        # Oblicz błąd dla inliers
        error = np.mean(distances[distances < threshold_distance])

        # Jeśli liczba inliers jest większa od aktualnego najlepszego
        if best_inliers is None or (len(inliers) > len(best_inliers) and error < best_error):
            best_plane = plane
            best_inliers = inliers
            best_error = error

    if best_plane is None:
        return None, None

    return best_plane, best_inliers


def is_plane_horizontal(plane_normal):
    return np.abs(plane_normal[2]) > np.abs(plane_normal[0]) and np.abs(plane_normal[2]) > np.abs(plane_normal[1])


def is_plane_vertical(plane_normal):
    return np.abs(plane_normal[0]) > np.abs(plane_normal[1]) and np.abs(plane_normal[0]) > np.abs(plane_normal[2])


# Wczytanie punktów z plików XYZ
poziome = load_xyz_file(
    r"C:\Users\Paulina\PycharmProjects\Programowanie_w_obliczeniach_inteligentnych\LAB1\poziome.xyz")
pionowe = load_xyz_file(
    r"C:\Users\Paulina\PycharmProjects\Programowanie_w_obliczeniach_inteligentnych\LAB1\pionowe.xyz")
cylindryczne = load_xyz_file(
    r"C:\Users\Paulina\PycharmProjects\Programowanie_w_obliczeniach_inteligentnych\LAB1\cylindryczne.xyz")

# Połączenie wszystkich punktów w jedną macierz
all_points = np.concatenate((poziome, pionowe, cylindryczne))

# Zastosowanie algorytmu k-średnich
labels, centroids = kmeans(all_points)

# Podział punktów na klastry na podstawie etykiet
cluster1 = all_points[labels == 0]
cluster2 = all_points[labels == 1]
cluster3 = all_points[labels == 2]

# Wyświetlenie wyników
fig = plt.figure(figsize=(15, 5))

ax1 = fig.add_subplot(131, projection='3d')
ax1.scatter(cluster1[:, 0], cluster1[:, 1], cluster1[:, 2], c='b', marker='o')
ax1.set_title('Chmura punktów Klastra 1')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')

ax2 = fig.add_subplot(132, projection='3d')
ax2.scatter(cluster2[:, 0], cluster2[:, 1], cluster2[:, 2], c='r', marker='o')
ax2.set_title('Chmura punktów Klastra 2')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('Z')

ax3 = fig.add_subplot(133, projection='3d')
ax3.scatter(cluster3[:, 0], cluster3[:, 1], cluster3[:, 2], c='g', marker='o')
ax3.set_title('Chmura punktów Klastra 3')
ax3.set_xlabel('X')
ax3.set_ylabel('Y')
ax3.set_zlabel('Z')

plt.show()

# Dopasowanie płaszczyzny za pomocą algorytmu RANSAC dla każdej chmury punktów
for cluster in [cluster1, cluster2, cluster3]:
    plane_normal, inliers = fit_plane_ransac(cluster)
    if plane_normal is not None:
        print("Współrzędne wektora normalnego płaszczyzny:", plane_normal)
        if len(inliers) > 0:
            print("Średnia odległość punktów do płaszczyzny:",
                  np.mean(np.linalg.norm(inliers - np.mean(inliers, axis=0), axis=1)))
            if is_plane_horizontal(plane_normal):
                print("Płaszczyzna jest pozioma")
            elif is_plane_vertical(plane_normal):
                print("Płaszczyzna jest pionowa")
            else:
                print("Płaszczyzna nie jest ani pozioma, ani pionowa")
        else:
            print("Nie udało się dopasować płaszczyzny.")
    else:
        print("Nie udało się dopasować płaszczyzny.")

