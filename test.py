import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Percorso dell'immagine
image_path = './annotation-images/out6.jpg'  # Sostituisci con il tuo percorso

# Dati delle coordinate dell'immagine
image_coordinates = [
    [860, 945],
    [270, 1636],
    [1208, 1753],
    [2578, 1758],
    [3515, 1673],
    [2944, 966],
    [2281, 946],
    [1518, 934],
    [610, 850],
    [564, 886],
    [341, 1065],
    [836, 1046],
    [801, 1089],
    [564, 1366],
    [521, 1416],
    [25, 1372],
    [3768, 1409],
    [3272, 1451],
    [3234, 1395],
    [3006, 1111],
    [2969, 1073],
    [3458, 1093],
    [3239, 914],
    [3197, 875]
]

# Carica l'immagine
img = mpimg.imread(image_path)

# Creazione della figura
plt.figure(figsize=(10, 10))

# Mostra l'immagine
plt.imshow(img)

# Plottare i punti
for coord in image_coordinates:
    plt.scatter(coord[0], coord[1], color='red')  # Cambia il colore se necessario

# Aggiungere etichette e titolo
plt.title('Punti delle Coordinate dell\'Immagine')
plt.xlabel('Coordinate X')
plt.ylabel('Coordinate Y')

# Mostrare la griglia
plt.grid()

# Mostrare il grafico
plt.show()
