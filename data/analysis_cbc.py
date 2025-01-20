"""
@author: Daniel A. Triana-Camacho

Este algoritmo se creo para comparar la resistencia eléctrica obtenida con 
los dispositivos PXIe y SME. La resistencia se obtuvo sometiendo a cargas
cíclicas.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import AutoLocator, MaxNLocator
from matplotlib import rcParams

def scatter_with_boxes_dynamic(x, data, label, ax=None, base_box_width=0.4, scatter_color='white', 
                               box_color='blue', median_color='red', alpha=0.5, scatter_size=10):
    """
    Dibuja un gráfico de dispersión con cajas y bigotes en el eje horizontal,
    ajustando dinámicamente el tamaño de las cajas según el tamaño de los números.

    Parámetros:
    - x: lista o array de posiciones en el eje x.
    - data: lista de arrays o listas con los datos para cada posición de x.
    - ax: objeto de ejes (opcional). Si es None, usa plt.gca() (el eje actual).
    - base_box_width: ancho base de las cajas.
    - scatter_color: color de los puntos dispersos.
    - box_color: color de las cajas y bigotes.
    - median_color: color de la línea de la mediana.
    - alpha: transparencia de los puntos dispersos.
    - scatter_size: tamaño de los puntos dispersos.
    """
    if ax is None:
        ax = plt.gca()  # Usar el eje actual si no se pasa uno explícito
    
    # Normalizar el ancho dinámico para que no haya barras invisibles
    dynamic_widths = base_box_width * ((x - np.min(x)) / (np.max(x) - np.min(x)) + 0.5)
    yline = []
    for xi, d, width in zip(x, data, dynamic_widths):
        # Calcular estadísticas
        q1 = np.percentile(d, 25)
        q3 = np.percentile(d, 75)
        iqr = q3 - q1
        mediana = np.median(d)
        bigote_inf = np.min(d[d >= q1 - 1.5 * iqr])
        bigote_sup = np.max(d[d <= q3 + 1.5 * iqr])

        # Dibujar caja
        ax.plot([xi - width / 2, xi + width / 2], [q1, q1], color=box_color)  # Límite inferior
        ax.plot([xi - width / 2, xi + width / 2], [q3, q3], color=box_color)  # Límite superior
        ax.plot([xi - width / 2, xi - width / 2], [q1, q3], color=box_color)  # Lado izquierdo
        ax.plot([xi + width / 2, xi + width / 2], [q1, q3], color=box_color)  # Lado derecho

        # Dibujar mediana
        ax.plot([xi - width / 2, xi + width / 2], [mediana, mediana], color=median_color, lw=2)
        yline.append(np.mean(d))
        
        # Dibujar bigotes
        ax.plot([xi, xi], [bigote_inf, q1], color=box_color)  # Bigote inferior
        ax.plot([xi, xi], [q3, bigote_sup], color=box_color)  # Bigote superior

        # Dibujar extremos de los bigotes
        ax.plot([xi - width / 4, xi + width / 4], [bigote_inf, bigote_inf], color=box_color)
        ax.plot([xi - width / 4, xi + width / 4], [bigote_sup, bigote_sup], color=box_color)

        # Dibujar puntos de dispersión
        ax.scatter([xi] * len(d), d, color=scatter_color, alpha=alpha, s=scatter_size)
    
    ax.plot(x, yline, '.-', color = box_color, label = label)


file = "parameters_cbs.xlsx"
path = "G:/My Drive/Papers/2024_Software_KDssZ/data/"
tag = ['Au NPs', 'MWCNTs', 'Graphite', 'rGO']
dpi = 300

df_Au = pd.read_excel(path+file, sheet_name=tag[0])
x_Au = df_Au['material'][67:72]

df_CNT = pd.read_excel(path+file, sheet_name=tag[1])
x_CNT = df_CNT['material'][67:75]

df_G = pd.read_excel(path+file, sheet_name=tag[2])
x_G = df_G['material'][67:71]

df_rGO = pd.read_excel(path+file, sheet_name=tag[3])
x_rGO = df_rGO['material'][67:71]

# Polarization resistance------------------------------------------------------
b = 2.6  
fig, ax = plt.subplots(figsize=(6.6875, b))
# Datos Au
x = np.float16(x_Au.values)  # Posiciones en el eje x
y = []
for sj in (67, 68, 69, 70, 71):
    yy = df_Au.loc[sj][1:4]
    y.append(yy.values)

data = y
# Llamar a la función
scatter_with_boxes_dynamic(x, data, label=tag[0], ax=ax, base_box_width=10, 
                               box_color='orange', 
                               median_color='red', 
                               alpha=0.5, 
                               scatter_size=10)

# Datos MWCNTs
x = np.float16(x_CNT.values)  # Posiciones en el eje x
y = []
for sj in (67, 68, 69, 70, 71, 72, 73, 74):
    yy = df_CNT.loc[sj][1:4]
    y.append(yy.values)

data = y
# Llamar a la función
scatter_with_boxes_dynamic(x, data, label=tag[1], ax=ax, base_box_width=10, 
                               box_color='lightblue', 
                               median_color='red', 
                               alpha=0.5, 
                               scatter_size=10)

# Datos graphite
x = np.float16(x_G.values)  # Posiciones en el eje x
y = []
for sj in (67, 68, 69, 70):
    yy = df_G.loc[sj][1:4]
    y.append(yy.values)

data = y
# Llamar a la función
scatter_with_boxes_dynamic(x, data, label=tag[2], ax=ax, base_box_width=10, 
                               box_color='blue', 
                               median_color='red', 
                               alpha=0.5, 
                               scatter_size=10)

# Datos reduced graphene oxide
x = np.float16(x_rGO.values)  # Posiciones en el eje x
y = []
for sj in (67, 68, 69, 70):
    yy = df_rGO.loc[sj][1:4]
    y.append(yy.values)

data = y
# Llamar a la función
scatter_with_boxes_dynamic(x, data, label=tag[3], ax=ax, base_box_width=10, 
                               box_color='darkblue', 
                               median_color='red', 
                               alpha=0.5, 
                               scatter_size=10)


# Configurar los ejes
# ax.xaxis.set_major_locator(AutoLocator())
ax.set_xlabel('Time [days]', fontname="Times New Roman", color='k', fontsize=11)
ax.set_ylabel('$\Delta$R [$\Omega$]', fontname="Times New Roman", color='k', fontsize=11)
plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
# Create custom legend handles
plt.legend(prop={'family': 'Times New Roman', 'size': 11}, loc='best')
plt.xticks(fontsize=9, fontname="Times New Roman", rotation=0)
plt.yticks(fontsize=9, fontname="Times New Roman", color='k') # cambia color y fuente de los ticks
plt.tight_layout() # to fix the size
# ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
# ax.set_xticks(0,7,28,38,46,88,114,135,179,203,226,262,273,351,442)

# Capacitance C3 at low frequency----------------------------------------------  
fig, ax = plt.subplots(figsize=(6.6875, b))
# Datos Au
x = np.float16(x_Au.values)  # Posiciones en el eje x
y = []
for sj in (52, 53, 54, 55, 56):
    yy = df_Au.loc[sj][1:4]
    y.append(yy.values)

data = y
# Llamar a la función
scatter_with_boxes_dynamic(x, data, label=tag[0], ax=ax, base_box_width=10, 
                               box_color='orange', 
                               median_color='red', 
                               alpha=0.5, 
                               scatter_size=10)

# Datos MWCNTs
x = np.float16(x_CNT.values)  # Posiciones en el eje x
y = []
for sj in (52, 53, 54, 55, 56, 57, 58, 59):
    yy = df_CNT.loc[sj][1:4]
    y.append(yy.values)

data = y
# Llamar a la función
scatter_with_boxes_dynamic(x, data, label=tag[1], ax=ax, base_box_width=10, 
                               box_color='lightblue', 
                               median_color='red', 
                               alpha=0.5, 
                               scatter_size=10)

# Datos graphite
x = np.float16(x_G.values)  # Posiciones en el eje x
y = []
for sj in (52, 53, 54, 55):
    yy = df_G.loc[sj][1:4]
    y.append(yy.values)

data = y
# Llamar a la función
scatter_with_boxes_dynamic(x, data, label=tag[2], ax=ax, base_box_width=10, 
                               box_color='blue', 
                               median_color='red', 
                               alpha=0.5, 
                               scatter_size=10)

# Datos reduced graphene oxide
x = np.float16(x_rGO.values)  # Posiciones en el eje x
y = []
for sj in (52, 53, 54, 55):
    yy = df_rGO.loc[sj][1:4]
    y.append(yy.values)

data = y
# Llamar a la función
scatter_with_boxes_dynamic(x, data, label=tag[3], ax=ax, base_box_width=10, 
                               box_color='darkblue', 
                               median_color='red', 
                               alpha=0.5, 
                               scatter_size=10)


# Configurar los ejes
# ax.xaxis.set_major_locator(AutoLocator())
ax.set_xlabel('Time [days]', fontname="Times New Roman", color='k', fontsize=11)
ax.set_ylabel('C$_3$ [F]', fontname="Times New Roman", color='k', fontsize=11)
plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
# Create custom legend handles
# plt.legend(prop={'family': 'Times New Roman', 'size': 11}, loc='best')
plt.xticks(fontsize=9, fontname="Times New Roman", rotation=0)
plt.yticks(fontsize=9, fontname="Times New Roman", color='k') # cambia color y fuente de los ticks
plt.tight_layout() # to fix the size
# ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
# ax.set_xticks(0,7,28,38,46,88,114,135,179,203,226,262,273,351,442)

# Admitance Q2 at midle frequency----------------------------------------------  
fig, ax = plt.subplots(figsize=(6.6875, b))
# Datos Au
x = np.float16(x_Au.values)  # Posiciones en el eje x
y = []
for sj in (67, 68, 69, 70, 71):
    yy = df_Au.loc[sj][12:15]
    y.append(yy.values)

data = y
# Llamar a la función
scatter_with_boxes_dynamic(x, data, label=tag[0], ax=ax, base_box_width=10, 
                               box_color='orange', 
                               median_color='red', 
                               alpha=0.5, 
                               scatter_size=10)

# Datos MWCNTs
x = np.float16(x_CNT.values)  # Posiciones en el eje x
y = []
for sj in (67, 68, 69, 70, 71, 72, 73, 74):
    yy = df_CNT.loc[sj][12:15]
    y.append(yy.values)

data = y
# Llamar a la función
scatter_with_boxes_dynamic(x, data, label=tag[1], ax=ax, base_box_width=10, 
                               box_color='lightblue', 
                               median_color='red', 
                               alpha=0.5, 
                               scatter_size=10)

# Datos graphite
x = np.float16(x_G.values)  # Posiciones en el eje x
y = []
for sj in (67, 68, 69, 70):
    yy = df_G.loc[sj][14:17]
    y.append(yy.values)

data = y
# Llamar a la función
scatter_with_boxes_dynamic(x, data, label=tag[2], ax=ax, base_box_width=10, 
                               box_color='blue', 
                               median_color='red', 
                               alpha=0.5, 
                               scatter_size=10)

# Datos reduced graphene oxide
x = np.float16(x_rGO.values)  # Posiciones en el eje x
y = []
for sj in (67, 68, 69, 70):
    yy = df_rGO.loc[sj][14:17]
    y.append(yy.values)

data = y
# Llamar a la función
scatter_with_boxes_dynamic(x, data, label=tag[3], ax=ax, base_box_width=10, 
                               box_color='darkblue', 
                               median_color='red', 
                               alpha=0.5, 
                               scatter_size=10)


# Configurar los ejes
# ax.xaxis.set_major_locator(AutoLocator())
ax.set_xlabel('Time [days]', fontname="Times New Roman", color='k', fontsize=11)
ax.set_ylabel(r'Q$_2$ [$\Omega^{-1}s^{1-\alpha}$]', fontname="Times New Roman", color='k', fontsize=11)
plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
# Create custom legend handles
# plt.legend(prop={'family': 'Times New Roman', 'size': 11}, loc='best')
plt.xticks(fontsize=9, fontname="Times New Roman", rotation=0)
plt.yticks(fontsize=9, fontname="Times New Roman", color='k') # cambia color y fuente de los ticks
plt.tight_layout() # to fix the size
# ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
# ax.set_xticks(0,7,28,38,46,88,114,135,179,203,226,262,273,351,442)

# Admitance Q1 at midle frequency----------------------------------------------  
fig, ax = plt.subplots(figsize=(6.6875, b))
# Datos Au
x = np.float16(x_Au.values)  # Posiciones en el eje x
y = []
for sj in (52, 53, 54, 55, 56):
    yy = df_Au.loc[sj][12:15]
    y.append(yy.values)

data = y
# Llamar a la función
scatter_with_boxes_dynamic(x, data, label=tag[0], ax=ax, base_box_width=10, 
                               box_color='orange', 
                               median_color='red', 
                               alpha=0.5, 
                               scatter_size=10)

# Datos MWCNTs
x = np.float16(x_CNT.values)  # Posiciones en el eje x
y = []
for sj in (52, 53, 54, 55, 56, 57, 58, 59):
    yy = df_CNT.loc[sj][12:15]
    y.append(yy.values)

data = y
# Llamar a la función
scatter_with_boxes_dynamic(x, data, label=tag[1], ax=ax, base_box_width=10, 
                               box_color='lightblue', 
                               median_color='red', 
                               alpha=0.5, 
                               scatter_size=10)

# Datos graphite
x = np.float16(x_G.values)  # Posiciones en el eje x
y = []
for sj in (52, 53, 54, 55):
    yy = df_G.loc[sj][14:17]
    y.append(yy.values)

data = y
# Llamar a la función
scatter_with_boxes_dynamic(x, data, label=tag[2], ax=ax, base_box_width=10, 
                               box_color='blue', 
                               median_color='red', 
                               alpha=0.5, 
                               scatter_size=10)

# Datos reduced graphene oxide
x = np.float16(x_rGO.values)  # Posiciones en el eje x
y = []
for sj in (52, 53, 54, 55):
    yy = df_rGO.loc[sj][14:17]
    y.append(yy.values)

data = y
# Llamar a la función
scatter_with_boxes_dynamic(x, data, label=tag[3], ax=ax, base_box_width=10, 
                               box_color='darkblue', 
                               median_color='red', 
                               alpha=0.5, 
                               scatter_size=10)


# Configurar los ejes
# ax.xaxis.set_major_locator(AutoLocator())
ax.set_xlabel('Time [days]', fontname="Times New Roman", color='k', fontsize=11)
ax.set_ylabel(r'Q$_1$ [$\Omega^{-1}s^{1-\alpha}$]', fontname="Times New Roman", color='k', fontsize=11)
plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
# Create custom legend handles
# plt.legend(prop={'family': 'Times New Roman', 'size': 11}, loc='best')
plt.xticks(fontsize=9, fontname="Times New Roman", rotation=0)
plt.yticks(fontsize=9, fontname="Times New Roman", color='k') # cambia color y fuente de los ticks
plt.tight_layout() # to fix the size
# ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
# ax.set_xticks(0,7,28,38,46,88,114,135,179,203,226,262,273,351,442)