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

text_size = 11
numbers_size = 9
a = 8/2.53 # x length of the figure
b = 7/2.53 # y length of the figure
position_y1 = 'upper left'
position_y2 = 'lower right'
ss = 10 # scatter size
alp = 0.5
bbw = 1 # base box width
box_color_RCP='orange'
box_color_RCB='darkblue'
median_color='red'

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


file = "parameters_rss.xlsx"
path = "G:/My Drive/Papers/2024_Software_KDssZ/data/"
tag = ['RCP', 'RCB']
dpi = 300

df_RCP = pd.read_excel(path+file, sheet_name=tag[0])
x_RCP = df_RCP['material'][55:59]
Temperature_RCP = df_RCP['f_midle_0 [Hz]'][55:59]

df_RCB = pd.read_excel(path+file, sheet_name=tag[1])
x_RCB = df_RCB['material'][74:78]
Temperature_RCB = df_RCB['𝜏1 [s] | hight freq'][74:78]

# Polarization resistance------------------------------------------------------ 
fig, ax = plt.subplots(figsize=(a, b))
# Datos RCP
x = np.float16(x_RCP.values)  # Posiciones en el eje x
y = []
for sj in (55, 56, 57, 58, 59):
    yy = df_RCP.loc[sj][1:4]
    y.append(yy.values)

data = y
# Llamar a la función
scatter_with_boxes_dynamic(x, data, label="$\Delta$R "+"("+tag[0]+")", ax=ax, base_box_width=bbw, 
                               box_color=box_color_RCP, 
                               median_color=median_color, 
                               alpha=alp, 
                               scatter_size=ss)

# Datos RCB
x = np.float16(x_RCB.values)  # Posiciones en el eje x
y = []
for sj in (74, 75, 76, 77, 78):
    yy = df_RCB.loc[sj][3:6]
    y.append(yy.values)

data = y
# Llamar a la función
scatter_with_boxes_dynamic(x, data, label="$\Delta$R "+"("+tag[1]+")", ax=ax, base_box_width=bbw, 
                               box_color=box_color_RCB, 
                               median_color=median_color, 
                               alpha=alp, 
                               scatter_size=ss)


# Configurar los ejes
# ax.xaxis.set_major_locator(AutoLocator())
ax.set_xlabel('Time [days]', fontname="Times New Roman", color='k', fontsize=text_size)
ax.set_ylabel('$\Delta$R [$\Omega$]', fontname="Times New Roman", color='k', fontsize=text_size)
plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
# Create custom legend handles
plt.legend(prop={'family': 'Times New Roman', 'size': text_size}, loc=position_y1)
plt.xticks(fontsize=numbers_size, fontname="Times New Roman", rotation=0)
plt.yticks(fontsize=numbers_size, fontname="Times New Roman", color='k') # cambia color y fuente de los ticks
# ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
# ax.set_xticks(0,7,28,38,46,88,114,135,179,203,226,262,273,351,442)

# Graficar la segunda curva en el eje derecho
ax1 = plt.twinx()
ax1.plot(x, Temperature_RCB, 'k-.', label="T "+"("+tag[0]+")")
ax1.plot(x, Temperature_RCP, 'k--', label="T "+"("+tag[1]+")")
ax1.set_ylabel("T [°C]", fontname="Times New Roman", color='k', fontsize=text_size)
# Ajustar los parámetros de los ticks del eje y derecho
ax1.tick_params(axis='y', colors='k', labelsize=numbers_size)
for label in ax1.get_yticklabels():
    label.set_fontname("Times New Roman")

plt.legend(prop={'family': 'Times New Roman', 'size': text_size}, loc=position_y2)
plt.tight_layout() # to fix the size

# Capacitance C3 at low frequency----------------------------------------------  
fig, ax = plt.subplots(figsize=(a, b))
# Datos RCP
x = np.float16(x_RCP.values)  # Posiciones en el eje x
y = []
for sj in (33, 34, 35, 36, 37):
    yy = df_RCP.loc[sj][1:4]
    y.append(yy.values)

data = y
# Llamar a la función
scatter_with_boxes_dynamic(x, data, label="C$_3$ "+"("+tag[0]+")", ax=ax, base_box_width=bbw, 
                               box_color=box_color_RCP, 
                               median_color=median_color, 
                               alpha=alp, 
                               scatter_size=ss)

# Datos RCB
x = np.float16(x_RCB.values)  # Posiciones en el eje x
y = []
for sj in (52, 53, 54, 55, 56):
    yy = df_RCB.loc[sj][3:6]
    y.append(yy.values)

data = y
# Llamar a la función
scatter_with_boxes_dynamic(x, data, label="C$_3$ "+"("+tag[1]+")", ax=ax, base_box_width=bbw, 
                               box_color=box_color_RCB, 
                               median_color=median_color, 
                               alpha=alp, 
                               scatter_size=ss)

# Configurar los ejes
# ax.xaxis.set_major_locator(AutoLocator())
ax.set_xlabel('Time [days]', fontname="Times New Roman", color='k', fontsize=text_size)
ax.set_ylabel('C$_3$ [F]', fontname="Times New Roman", color='k', fontsize=text_size)
plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
# Create custom legend handles
# plt.legend(prop={'family': 'Times New Roman', 'size': text_size}, loc=position_y1)
plt.xticks(fontsize=numbers_size, fontname="Times New Roman", rotation=0)
plt.yticks(fontsize=numbers_size, fontname="Times New Roman", color='k') # cambia color y fuente de los ticks
# ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
# ax.set_xticks(0,7,28,38,46,88,114,135,179,203,226,262,273,351,442)

# Graficar la segunda curva en el eje derecho
ax1 = plt.twinx()
ax1.plot(x, Temperature_RCB, 'k-.', label="T "+"("+tag[0]+")")
ax1.plot(x, Temperature_RCP, 'k--', label="T "+"("+tag[1]+")")
ax1.set_ylabel("T [°C]", fontname="Times New Roman", color='k', fontsize=text_size)
# Ajustar los parámetros de los ticks del eje y derecho
ax1.tick_params(axis='y', colors='k', labelsize=numbers_size)
for label in ax1.get_yticklabels():
    label.set_fontname("Times New Roman")

# plt.legend(prop={'family': 'Times New Roman', 'size': text_size}, loc=position_y2)
plt.tight_layout() # to fix the size

# Admitance Q2 at midle frequency----------------------------------------------  
fig, ax = plt.subplots(figsize=(a, b))
# Datos RCP
x = np.float16(x_RCP.values)  # Posiciones en el eje x
y = []
for sj in (55, 56, 57, 58, 59):
    yy = df_RCP.loc[sj][12:15]
    y.append(yy.values)

data = y
# Llamar a la función
scatter_with_boxes_dynamic(x, data, label="Q$_2$ "+"("+tag[0]+")", ax=ax, base_box_width=bbw, 
                               box_color=box_color_RCP, 
                               median_color=median_color, 
                               alpha=alp, 
                               scatter_size=ss)

# Datos RCB
x = np.float16(x_RCB.values)  # Posiciones en el eje x
y = []
for sj in (74, 75, 76, 77, 78):
    yy = df_RCB.loc[sj][14:17]
    y.append(yy.values)

data = y
# Llamar a la función
scatter_with_boxes_dynamic(x, data, label="Q$_2$ "+"("+tag[1]+")", ax=ax, base_box_width=bbw, 
                               box_color=box_color_RCB, 
                               median_color=median_color, 
                               alpha=alp, 
                               scatter_size=ss)

# Configurar los ejes
# ax.xaxis.set_major_locator(AutoLocator())
ax.set_xlabel('Time [days]', fontname="Times New Roman", color='k', fontsize=text_size)
ax.set_ylabel(r'Q$_2$ [$\Omega^{-1}s^{1-\alpha}$]', fontname="Times New Roman", color='k', fontsize=text_size)
plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
# Create custom legend handles
# plt.legend(prop={'family': 'Times New Roman', 'size': text_size}, loc=position_y1)
plt.xticks(fontsize=numbers_size, fontname="Times New Roman", rotation=0)
plt.yticks(fontsize=numbers_size, fontname="Times New Roman", color='k') # cambia color y fuente de los ticks
# ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
# ax.set_xticks(0,7,28,38,46,88,114,135,179,203,226,262,273,351,442)

# Graficar la segunda curva en el eje derecho
ax1 = plt.twinx()
ax1.plot(x, Temperature_RCB, 'k-.', label="T "+"("+tag[0]+")")
ax1.plot(x, Temperature_RCP, 'k--', label="T "+"("+tag[1]+")")
ax1.set_ylabel("T [°C]", fontname="Times New Roman", color='k', fontsize=text_size)
# Ajustar los parámetros de los ticks del eje y derecho
ax1.tick_params(axis='y', colors='k', labelsize=numbers_size)
for label in ax1.get_yticklabels():
    label.set_fontname("Times New Roman")

# plt.legend(prop={'family': 'Times New Roman', 'size': text_size}, loc=position_y2)
plt.tight_layout() # to fix the size

# Admitance Q1 at midle frequency----------------------------------------------  
fig, ax = plt.subplots(figsize=(a, b))
# Datos RCP
x = np.float16(x_RCP.values)  # Posiciones en el eje x
y = []
for sj in (33, 34, 35, 36, 37):
    yy = df_RCP.loc[sj][12:15]
    y.append(yy.values)

data = y
# Llamar a la función
scatter_with_boxes_dynamic(x, data, label="Q$_1$ "+"("+tag[0]+")", ax=ax, base_box_width=bbw, 
                               box_color=box_color_RCP, 
                               median_color=median_color, 
                               alpha=alp, 
                               scatter_size=ss)

# Datos RCB
x = np.float16(x_RCB.values)  # Posiciones en el eje x
y = []
for sj in (52, 53, 54, 55, 56):
    yy = df_RCB.loc[sj][14:17]
    y.append(yy.values)

data = y
# Llamar a la función
scatter_with_boxes_dynamic(x, data, label="Q$_1$ "+"("+tag[1]+")", ax=ax, base_box_width=bbw, 
                               box_color=box_color_RCB, 
                               median_color=median_color,
                               alpha=alp, 
                               scatter_size=ss)


# Configurar los ejes
# ax.xaxis.set_major_locator(AutoLocator())
ax.set_xlabel('Time [days]', fontname="Times New Roman", color='k', fontsize=text_size)
ax.set_ylabel(r'Q$_1$ [$\Omega^{-1}s^{1-\alpha}$]', fontname="Times New Roman", color='k', fontsize=text_size)
plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
# Create custom legend handles
# plt.legend(prop={'family': 'Times New Roman', 'size': text_size}, loc=position_y1)
plt.xticks(fontsize=numbers_size, fontname="Times New Roman", rotation=0)
plt.yticks(fontsize=numbers_size, fontname="Times New Roman", color='k') # cambia color y fuente de los ticks
# ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
# ax.set_xticks(0,7,28,38,46,88,114,135,179,203,226,262,273,351,442)

# Graficar la segunda curva en el eje derecho
ax1 = plt.twinx()
ax1.plot(x, Temperature_RCB, 'k-.', label="T "+"("+tag[0]+")")
ax1.plot(x, Temperature_RCP, 'k--', label="T "+"("+tag[1]+")")
ax1.set_ylabel("T [°C]", fontname="Times New Roman", color='k', fontsize=text_size)
# Ajustar los parámetros de los ticks del eje y derecho
ax1.tick_params(axis='y', colors='k', labelsize=numbers_size)
for label in ax1.get_yticklabels():
    label.set_fontname("Times New Roman")

# plt.legend(prop={'family': 'Times New Roman', 'size': text_size}, loc=position_y2)
plt.tight_layout() # to fix the size