"""
@author: Daniel A. Triana-Camacho

Este algoritmo se creo para comparar los parametros obtenidos del modelo de
impedancia a diferentes frecuencias
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
curve_color_CNT1 ='blue'
curve_color_CNT1p5 ='darkblue'
labels = ['CNT 1.0 $\%$', 'CNT 1.5 $\%$']

file = "parameters_CNT_cube.xlsx"
path = "G:/My Drive/Papers/2024_Software_KDssZ/data/"
tag = 'CNTs'
dpi = 300

df = pd.read_excel(path+file, sheet_name=tag)
x_CNT1 = df['force (kN)'][0:11]
x_CNT1p5 = df['force (kN)'][15:26]

# Polarization resistance------------------------------------------------------ 
fig, ax = plt.subplots(figsize=(a, b))
# Datos CNT 1%
y_CNT1 = df['R1-Ri [Ω]'][0:11]
# Datos CNT 1.5%
y_CNT1p5 = df['R1-Ri [Ω]'][15:26]

ax.plot(x_CNT1, y_CNT1, '.-', color=curve_color_CNT1, label = labels[0])
# ax.plot(x_CNT1p5, y_CNT1p5, '.-', label = label[1])

# Configurar los ejes
ax.set_xlabel('Force [kN]', fontname="Times New Roman", color='k', fontsize=text_size)
ax.set_ylabel('$\Delta$R [$\Omega$]', fontname="Times New Roman", color='k', fontsize=text_size)
plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
# Create custom legend handles
plt.legend(prop={'family': 'Times New Roman', 'size': text_size}, loc=position_y1)
plt.xticks(fontsize=numbers_size, fontname="Times New Roman", rotation=0)
plt.yticks(fontsize=numbers_size, fontname="Times New Roman", color='k') # cambia color y fuente de los ticks

# Graficar la segunda curva en el eje derecho
ax1 = plt.twinx()
# ax1.plot(x_CNT1p5, y_CNT1p5, 'k-.', label=label[1])
ax1.plot(x_CNT1p5, y_CNT1p5, '.-.', color=curve_color_CNT1p5, label=labels[1])
ax1.set_ylabel("$\Delta$R [$\Omega$]", fontname="Times New Roman", color='k', fontsize=text_size)
# Ajustar los parámetros de los ticks del eje y derecho
ax1.tick_params(axis='y', colors='k', labelsize=numbers_size)
for label in ax1.get_yticklabels():
    label.set_fontname("Times New Roman")
plt.legend(prop={'family': 'Times New Roman', 'size': text_size}, loc=position_y2)

plt.tight_layout() # to fix the size

# # Admitance Q1 at midle frequency----------------------------------------------  
fig, ax = plt.subplots(figsize=(a, b))
# Datos CNT 1%
y_CNT1 = df['Q1 [T] | hight freq'][0:11]
# Datos CNT 1.5%
y_CNT1p5 = df['Q1 [T] | hight freq'][15:26]

ax.plot(x_CNT1, y_CNT1, '.-', color=curve_color_CNT1, label = labels[0])
# ax.plot(x_CNT1p5, y_CNT1p5, '.-', label = label[1])

# Configurar los ejes
ax.set_xlabel('Force [kN]', fontname="Times New Roman", color='k', fontsize=text_size)
ax.set_ylabel(r'Q$_1$ [$\Omega^{-1}s^{1-\alpha}$]', fontname="Times New Roman", color='k', fontsize=text_size)
plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
# Create custom legend handles
plt.legend(prop={'family': 'Times New Roman', 'size': text_size}, loc=position_y1)
plt.xticks(fontsize=numbers_size, fontname="Times New Roman", rotation=0)
plt.yticks(fontsize=numbers_size, fontname="Times New Roman", color='k') # cambia color y fuente de los ticks

# Graficar la segunda curva en el eje derecho
ax1 = plt.twinx()
# ax1.plot(x_CNT1p5, y_CNT1p5, 'k-.', label=label[1])
ax1.plot(x_CNT1p5, y_CNT1p5, '.-.', color=curve_color_CNT1p5, label=labels[1])
ax1.set_ylabel(r'Q$_1$ [$\Omega^{-1}s^{1-\alpha}$]', fontname="Times New Roman", color='k', fontsize=text_size)
# Ajustar los parámetros de los ticks del eje y derecho
ax1.tick_params(axis='y', colors='k', labelsize=numbers_size)
for label in ax1.get_yticklabels():
    label.set_fontname("Times New Roman")
plt.legend(prop={'family': 'Times New Roman', 'size': text_size}, loc=position_y2)

plt.tight_layout() # to fix the size

# # Admitance Q2 at midle frequency----------------------------------------------  
fig, ax = plt.subplots(figsize=(a, b))
# Datos CNT 1%
y_CNT1 = df['Q2 [T] | middle freq'][0:11]
# Datos CNT 1.5%
y_CNT1p5 = df['Q2 [T] | middle freq'][15:26]

ax.plot(x_CNT1, y_CNT1, '.-', color=curve_color_CNT1, label = labels[0])
# ax.plot(x_CNT1p5, y_CNT1p5, '.-', label = label[1])

# Configurar los ejes
ax.set_xlabel('Force [kN]', fontname="Times New Roman", color='k', fontsize=text_size)
ax.set_ylabel(r'Q$_2$ [$\Omega^{-1}s^{1-\alpha}$]', fontname="Times New Roman", color='k', fontsize=text_size)
plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
# Create custom legend handles
plt.legend(prop={'family': 'Times New Roman', 'size': text_size}, loc=position_y1)
plt.xticks(fontsize=numbers_size, fontname="Times New Roman", rotation=0)
plt.yticks(fontsize=numbers_size, fontname="Times New Roman", color='k') # cambia color y fuente de los ticks

# Graficar la segunda curva en el eje derecho
ax1 = plt.twinx()
# ax1.plot(x_CNT1p5, y_CNT1p5, 'k-.', label=label[1])
ax1.plot(x_CNT1p5, y_CNT1p5, '.-.', color=curve_color_CNT1p5, label=labels[1])
ax1.set_ylabel(r'Q$_2$ [$\Omega^{-1}s^{1-\alpha}$]', fontname="Times New Roman", color='k', fontsize=text_size)
# Ajustar los parámetros de los ticks del eje y derecho
ax1.tick_params(axis='y', colors='k', labelsize=numbers_size)
for label in ax1.get_yticklabels():
    label.set_fontname("Times New Roman")
plt.legend(prop={'family': 'Times New Roman', 'size': text_size}, loc=position_y2)

plt.tight_layout() # to fix the size

# # Capacitance C3 at low frequency----------------------------------------------  
fig, ax = plt.subplots(figsize=(a, b))
# Datos CNT 1%
y_CNT1 = df['C3 [F] | low freq'][0:11]
# Datos CNT 1.5%
y_CNT1p5 = df['C3 [F] | low freq'][15:26]

ax.plot(x_CNT1, y_CNT1, '.-', color=curve_color_CNT1, label = labels[0])
# ax.plot(x_CNT1p5, y_CNT1p5, '.-', label = label[1])

# Configurar los ejes
ax.set_xlabel('Force [kN]', fontname="Times New Roman", color='k', fontsize=text_size)
ax.set_ylabel('C$_3$ [F]', fontname="Times New Roman", color='k', fontsize=text_size)
plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
# Create custom legend handles
plt.legend(prop={'family': 'Times New Roman', 'size': text_size}, loc=position_y1)
plt.xticks(fontsize=numbers_size, fontname="Times New Roman", rotation=0)
plt.yticks(fontsize=numbers_size, fontname="Times New Roman", color='k') # cambia color y fuente de los ticks

# Graficar la segunda curva en el eje derecho
ax1 = plt.twinx()
# ax1.plot(x_CNT1p5, y_CNT1p5, 'k-.', label=label[1])
ax1.plot(x_CNT1p5, y_CNT1p5, '.-.', color=curve_color_CNT1p5, label=labels[1])
ax1.set_ylabel('C$_3$ [F]', fontname="Times New Roman", color='k', fontsize=text_size)
# Ajustar los parámetros de los ticks del eje y derecho
ax1.tick_params(axis='y', colors='k', labelsize=numbers_size)
for label in ax1.get_yticklabels():
    label.set_fontname("Times New Roman")
plt.legend(prop={'family': 'Times New Roman', 'size': text_size}, loc=position_y2)

plt.tight_layout() # to fix the size