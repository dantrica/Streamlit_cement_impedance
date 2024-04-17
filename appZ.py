# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 22:34:44 2024

@author: Daniel A- Triana-Camacho

The AppZ is a window for inputting electrical impedance data into 
the Key Data Source Since impedance (KDssZ) library to optimize a 
three-part lumped circuit model. The App provides Nyquist and Bode 
plots along with a table that contains the optimized parameters 
from the model.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from KDssZ import *

# Instantiating the class KDssZ()
KDss = KDssZ()

# Setting the app to occupy the entire width of the screen
st.set_page_config(layout="wide")

# Welcome message on the title
title = 'Welcome to the key data source since impedance (KDssZ)'
st.title(title)

# Software description in brief
st.write("This is a software tool for electrical impedance analysis of cement-based composites")
st.write('---')

# Adding selected_filename to the application sesion
if 'selected_filename' not in st.session_state:
    st.session_state.selected_filename = []

#------------------------------------------------------------------------------
# Building the left option sidebar
with st.sidebar:
    sidebar_title = "KDssZ Options"
    st.sidebar.title(sidebar_title)
    
    # file_uploader to get the .txt impedance data
    files_txt = st.file_uploader(
        label="Upload Z data", 
        accept_multiple_files=True,
        help='''Upload all .txt files that contains impedance measurements 
            from PGSTAT206 by AUTOLAB, and those were extracted with 
            the software NOVA 2.0'''
        )
    
    # Asking if is there elements in the list
    if len(files_txt) > 0:
        
        # adding elements from files_txt to the list file_names
        file_names = []
        for j in files_txt:
            file_names.append(j.name)
        
        # The box contains a list with the file names st.selectbox('text', list)
        file_box = st.selectbox(
            'Measurements',
            file_names,
            label_visibility="visible",
            help='''Select here the measure to add or remove in the 
                optimization zone'''
            )
    
        # Create two columns
        sidebar_col1, sidebar_col2 = st.columns(2)
    
        # Button to add Z data, column 1
        with sidebar_col1:
            with st.container(height=180):
                if st.button('Add data', help="Add chosen data file"):
                    # Warning, don't add the same file twice
                    if file_box in st.session_state.selected_filename:
                        st.write('''The file already was added, please 
                                 chose other''')
                    else:
                        st.session_state.selected_filename.append(file_box)
    
        # Button to remove Z data, column 2
        with sidebar_col2:
            with st.container(height=180):
                if st.button('Remove', help='''Remove data file previoulsy 
                             added'''):
                    # Warning, There are not more files to remove
                    if len(st.session_state.selected_filename) == 0:
                        st.write('The list is empty, please add a file')
                    else:
                        if file_box not in st.session_state.selected_filename:
                            st.write('''The file is not on the list, please 
                                     try removing other''')
                        else:
                            st.session_state.selected_filename.remove(file_box)

        # Updating the list of Z files
        with st.container(height=200):
            st.write('Optimization zone:', st.session_state.selected_filename)
    
        #----------------------------------------------------------------------
        # Buttons Run and Reset to start and delete 
        # the optimization, respectively
    
        # Create two columns
        sidebar_col3, sidebar_col4 = st.columns(2)
    
        # Button Run, Calling the method KDss.call_data to save the information
        # coming from .txt files in global variables of KDss
        with sidebar_col3:
            if st.button('Run', help="Run the optimization algoritm"):
                'Optimizing'
                fn_chose = st.session_state.selected_filename # fn: file name
                # KDss.call_data(fn_chose, df)
                KDss.call_data(fn_chose, files_txt)

        with sidebar_col4:
            if st.button('Reset', help="Clear the paramenters table and figures"):
                'we are working on'
                st.session_state.selected_filename.clear()
    
    #--------------------------------------------------------------------------
    # Presenting the figure of the lumped circuital model
    imagen = 'figures/circuital_model.png'
    if os.path.exists(imagen):
        st.image(
            imagen, caption='Model: three frequencies mode',
            use_column_width=True
            )
    else:
        st.write('The image is not in the filepath: '+imagen)

#------------------------------------------------------------------------------
fig1 = plt.figure()
fig2 = plt.figure()

# Container with the Nyquist and Bode diagrams
with st.container():    
       
    param_labels = [
    "Ri [Ω]",
    "R1-Ri [Ω]",
    "𝜏1 [s] | hight freq",
    "α1 | hight freq",
    "R3 [Ω] | low freq",
    "𝜏3 [s] | low freq",
    "α3 | low freq",
    "R2 [Ω] | middle freq",
    "𝜏2 [s] | middle freq",
    "α2 | middle freq"
    ]
    model_data = {
        'sample':[]
    }
    for p in param_labels:
        model_data[p] = []
        
    params = []
    fdata = []
    Zreal = []
    Zimag = []
    f_model = []
    Zreal_model = []
    Zimag_model = []
    for i, label in enumerate(KDss.labels):
        print(i)
        f = KDss.freqs[i] 
        Z = KDss.impedances[i]
        p = 10*[np.nan]
        if max(f) > 100e3:
            p, Z_model, fm = KDss.model(i, f, Z, f_hight=25e3, f_middle=10e3, f_low=1, label=label)
            
        params.append(p)
        model_data['sample'].append(KDss.labels[i])
        for key, val in enumerate(p):
            this_val = val
            if key == 6:
                this_val = 1 - val
            model_data[param_labels[key]].append(this_val)
        fdata.append(f)
        Zreal.append(np.real(Z)*1e-3)
        Zimag.append(-np.imag(Z)*1e-3)
        
        if 'fm' in globals():
            f_model.append(fm)
            Zreal_model.append(np.real(Z_model)*1e-3)
            Zimag_model.append(-np.imag(Z_model)*1e-3)
        else:
            st.write('This data file was not optimized')
            f_model.append([])
            Zreal_model.append([])
            Zimag_model.append([])
    
    new_fdata = list(zip(*fdata))
    new_Zreal = list(zip(*Zreal))
    new_Zimag = list(zip(*Zimag))
    new_f_model = list(zip(*f_model))
    new_Zreal_model = list(zip(*Zreal_model))
    new_Zimag_model = list(zip(*Zimag_model))
    
    # Column format to present the Nyquist and Bode diagrams
    sidebar_col5, sidebar_col6 = st.columns(2)
    
    # Nyquist plot
    with sidebar_col5:
        fig1 = plt.figure(1)
        
        if len(KDss.labels) == 0:
            sl = ''
        elif len(KDss.labels) == 1:
            sl = KDss.labels[0], 'Model Fitting'
        else:
            sl = []
            for sj in KDss.labels:
                sl.append(sj)
            sl.append('Model Fitting')
        
        plt.plot(new_Zreal, new_Zimag, linestyle = 'None', marker='o', markerfacecolor='None')
        plt.plot(new_Zreal_model, new_Zimag_model, 'k:')
        plt.gca().set_aspect('equal', 'box')
        plt.xlabel(r"real(Z) [k$\Omega$]")
        plt.ylabel(r"-imag(Z) [k$\Omega$]")
        xlim = []
        for sr in new_Zreal:
            xlim.append(sr.max())
        ylim = []
        for si in new_Zimag:
            ylim.append(si.max())
        plt.xlim(0, max(xlim))
        plt.ylim(0, max(ylim))
        plt.legend(labels=sl)
        
        st.pyplot(fig1)
    
    # Bode plot
    with sidebar_col6:
        
        fig2 = plt.figure(2)
        plt.subplot(2,1,1)
        plt.semilogx(new_fdata, new_Zreal, linestyle = 'None', marker='o', markerfacecolor='None')
        plt.semilogx(new_f_model, new_Zreal_model, 'k:')#, lw=2, markerfacecolor='None', alpha=0.7)
        plt.subplot(2,1,2)
        plt.semilogx(new_fdata, new_Zimag, linestyle = 'None', marker='o', markerfacecolor='None')
        plt.semilogx(new_f_model, new_Zimag_model, 'k:')#, lw=2, markerfacecolor='None', alpha=0.7)
        #----------------------------------------------------------------------
        plt.subplot(2,1,1)
        plt.xlim(0.1, 1e6)
        #plt.ylim(0, 30)
        plt.xlabel('Frequency [Hz]')
        plt.ylabel(r"real(Z) [k$\Omega$]")
        plt.gca().yaxis.set_label_position("right")
        plt.gca().yaxis.tick_right()
        plt.legend(labels=sl)
        #----------------------------------------------------------------------
        plt.subplot(2,1, 2)
        plt.xlim(0.1, 1e6)
        #plt.ylim(0, 10)
        plt.xlabel('Frequency [Hz]')
        plt.ylabel(r"-imag(Z)  [k$\Omega$]")
        plt.gca().yaxis.set_label_position("right")
        plt.gca().yaxis.tick_right()
        plt.legend(labels=sl)
        
        st.pyplot(fig2)
        
        # st.scatter_chart(pd.DataFrame(new_Zreal_model))
        
columns = [param_labels[i] for i in [0, 1, 2, 3, 7, 8, 9, 4, 5, 6]]
model_df = pd.DataFrame(model_data, columns=columns)


# DataFrame which contains the optimized parameters
with st.container():
    edited_model_df = st.data_editor(
        model_df,
        hide_index=True,
    )
