import numpy as np
import matplotlib.pyplot as plt


def generuj_punkty_poziome(szerokosc, dlugosc, liczba_punktow):
    x = np.random.uniform(0, szerokosc, liczba_punktow)
    y = np.random.uniform(0, dlugosc, liczba_punktow)
    z = np.zeros(liczba_punktow)
    return x, y, z

def generuj_punkty_pionowe(szerokosc, wysokosc, liczba_punktow):
    x = np.random.uniform(0, szerokosc, liczba_punktow)
    y = np.zeros(liczba_punktow)
    z = np.random.uniform(0, wysokosc, liczba_punktow)
    return x, y, z

def generuj_punkty_cylindryczne(promien, wysokosc, liczba_punktow):
    theta = np.random.uniform(0, 2*np.pi, liczba_punktow)
    z = np.random.uniform(0, wysokosc, liczba_punktow)
    x = promien * np.cos(theta)
    y = promien * np.sin(theta)
    return x, y, z

def zapisz_punkty_do_pliku_xyz(nazwa_pliku, x, y, z):
    with open(nazwa_pliku, 'w') as f:
        for i in range(len(x)):
            f.write(f"{x[i]} {y[i]} {z[i]}\n")

def wykres_punktow_z_pliku_xyz(nazwa_pliku, tytul):
    dane = np.loadtxt(nazwa_pliku)
    x = dane[:, 0]
    y = dane[:, 1]
    z = dane[:, 2]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, c='b', marker='o')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(tytul)
    plt.show()

# Parametry dla przypadków
szerokosc = 10
dlugosc = 15
wysokosc = 8
promien = 5
liczba_punktow = 100

# Generowanie i zapis punktów
x, y, z = generuj_punkty_poziome(szerokosc, dlugosc, liczba_punktow)
zapisz_punkty_do_pliku_xyz("poziome.xyz", x, y, z)
wykres_punktow_z_pliku_xyz("poziome.xyz", "Powierzchnia Pozioma")

x, y, z = generuj_punkty_pionowe(szerokosc, wysokosc, liczba_punktow)
zapisz_punkty_do_pliku_xyz("pionowe.xyz", x, y, z)
wykres_punktow_z_pliku_xyz("pionowe.xyz", "Powierzchnia Pionowa")

x, y, z = generuj_punkty_cylindryczne(promien, wysokosc, liczba_punktow)
zapisz_punkty_do_pliku_xyz("cylindryczne.xyz", x, y, z)
wykres_punktow_z_pliku_xyz("cylindryczne.xyz", "Powierzchnia Cylindryczna")
