# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 22:03:30 2024

@authors: David Alejandro Miranda Mercado and Daniel Andrés Triana Camacho
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy import optimize
from os.path import join

class KDssZ:
    freqs = []
    impedances = []
    labels = []
    concent = []
    def __init__(self):
        self.instance_variable = []
        self.freqs = []
        self.impedances = []
        self.labels = []
        self.concent = []
     
    # fn: archivos seleccionados en la app, metadata: todos los archivos
    def call_data(self, fn, files_txt):
        for uf in files_txt:
            if uf.name in fn:
                df = pd.read_csv(uf, sep='\t')
                f = df["Frequency (Hz)"]
                Z = df["Z' (Ω)"].to_numpy() - 1j * df["-Z'' (Ω)"].to_numpy()
                self.freqs.append(f)
                self.impedances.append(Z)
                self.labels.append(uf.name)
                
    def cole_cole(self, f, Xi, DX, tau, alpha):
        return Xi + DX / ( 1 + (2j*np.pi*f*tau)**(1-alpha) )
    
    
    def optimize_tau(self, f, X, Xi, DX, tau, alpha, algorith='least_squares'):
        def err(T):
            if T < 0:
                return 100
            X1 = self.cole_cole(f, Xi, DX, T, alpha)
            return sum(np.abs(X1 - X)**2)/sum(np.abs(X)**2)
        opt = np.nan
        if 'least_squares' in algorith:
            opt = optimize.least_squares(err, tau, bounds=(0, np.inf))
        if 'basinhopping' in algorith:
            opt = optimize.basinhopping(err, tau, niter=1000)
        return opt.x[0]
    
    def get_cole_params(self, f, X, algorith='least_squares', verbose=True):
        def f_2b(c):
            """ calculate the algebraic distance between the 2D points and the mean circle centered at c=(-xc, -yc, C) """
            A, B, C = c
            Xi = 100
            if A**2 >= C:
                R = np.sqrt(A**2 + B**2 - C)
                Xi = A + np.sqrt( A**2 - C )
            x = np.real(X)
            y = np.imag(X)
            r = x**2 + y**2 + 2*x*A + 2*y*B + C
            if Xi < 0:
                return 10*r
            return r
        x_m = np.mean(np.real(X))
        R0  = ( max(np.real(X)) - min(np.real(X)) )/2.0
        y_m = max(np.imag(X)) - R0
        C0  =  x_m**2 + y_m**2 - R0**2
        ABC = -x_m, -y_m, C0
        ABC_opt, ier = optimize.leastsq(f_2b, ABC)
        A, B, C = ABC_opt
        R = np.sqrt(A**2 + B**2 - C) # (y + B)^2 + (x + A)^2 = R^2
        if R**2 < B**2:
            if verbose:
                print('Warning! The Cole-Cole inversion fails.')
            return np.nan, np.nan, np.nan, np.nan
        DX = 2 * np.sqrt( R**2 - B**2 )
        Xi1 = -A - DX/2
        Xi2 =  A + DX/2
        Xmax = R - B
        phi = np.arctan(-2*B/DX)
        alpha = 2 * phi / np.pi

        def get_tau(Xi):
            # Obtention of tau #
            k = min(range(len(f)), key=lambda i: abs(-np.imag(X)[i] - Xmax))
            fc = f[k]
            if type(fc) == type([]):
                fc = fc[0]
            if DX/(DX+Xi) < 0:
                return 1
            return np.sqrt(DX/(DX+Xi))/(2*np.pi*fc)
            
        X1 = self.cole_cole(f, Xi1, DX, get_tau(Xi1), alpha)
        X2 = self.cole_cole(f, Xi2, DX, get_tau(Xi2), alpha)
        Xi = Xi1
        tau = get_tau(Xi1)
        if sum(np.abs(X1 - X)**2) > sum(np.abs(X2 - X)**2):
            Xi = Xi2
            tau = get_tau(Xi2)
    
        tau_opt = self.optimize_tau(f, X, Xi, DX, tau, alpha, algorith=algorith)
        X1 = self.cole_cole(f, Xi1, DX, tau, alpha)
        X2 = self.cole_cole(f, Xi2, DX, tau_opt, alpha)
        if not np.isnan(tau_opt) and sum(np.abs(X1 - X)**2) > sum(np.abs(X2 - X)**2):
            tau = tau_opt

        return Xi, DX, tau, alpha
    
    def impedance_model(self, f, p):
        return self.impedance_model_opt(f, p[4:], p[:4])

    def impedance_model_opt(self, f, p, cole_cole_fixed_params):
        #Rq, T, a, Ro, To, a0 = p
        other_fixed_params = p[3:]
        return self.impedance_model_two(f, p[:3], cole_cole_fixed_params, other_fixed_params)

    def impedance_model_one(self, f, p, cole_cole_fixed_params):
        Ro, To, a0 = p
        Xi, DX, tau, alpha = cole_cole_fixed_params
        Z1 = self.cole_cole(f, Xi, DX, tau, alpha)
        ZZ = Ro/(2j * f * To)**(1 - a0)
        return  Z1 + ZZ

    def impedance_model_two(self, f, p, cole_cole_fixed_params, other_fixed_params):
        Rq, T, a = p
        Xi, DX, tau, alpha = cole_cole_fixed_params
        Ro, To, a0 = other_fixed_params
        #Z1 = impedance_model_one(f, other_fixed_params, cole_cole_fixed_params)
        Z1 = self.cole_cole(f, Xi, DX, tau, alpha)
        ZZ = Ro/(2j * f * To)**(1 - a0)
        ZC = Rq/(2j * f * T)**a
        return  Z1 + ZC + ZZ

    def optimize1_model(self, f, Z, x0, cole_cole_fixed_params):
        """ optimize_model determina los parámetros de un modelo de fase constante tipo capactivio
            con Z = Rq/(jwT)^a, donde T = Rq Cq. cole_Cole_fixed_params son parámetros del modelo de cole-cole
            obtenido con los datos de impedancia, se dejan fijos.
            x0 = [tau_c/(DC+Ci), tau_c, alpha_c], donde tau_c, DC, Ci y alpha_c son obtenidos del modelo de cole-cole
            aplicado a la capacitancia compleja C = 1/(jwZ)
        """
        if any(np.isnan(x0)):
            return 6*[np.nan]
        def err1(p):
            a = p[2]
            if any( p < 0 ) or a > 1:
                return 100 
            ZZ = self.impedance_model_one(f, p, cole_cole_fixed_params)
            return sum(np.abs(Z - ZZ)**2)/sum(np.abs(Z)**2)

        opt = optimize.least_squares(err1, np.array(x0)[:3], bounds=(0, np.inf))
        #Rq, T, a = opt.x
        def err2(p):
            a = p[2]
            if any( p < 0 ) or a > 1:
                return 100 
            ZZ = self.impedance_model_two(f, p, cole_cole_fixed_params, opt.x)
            return sum(np.abs(Z - ZZ)**2)/sum(np.abs(Z)**2)
        #Ro, To, a0 = x0[:3]
        opt2 = optimize.least_squares(err2, np.array(x0[3:]), bounds=(0, np.inf))
        return np.concatenate([opt2.x, opt.x])

    def optimize_model(self, f, Z, x0, cole_cole_fixed_params):
        if any(np.isnan(x0)):
            return 6*[np.nan]
        if any(np.isnan(cole_cole_fixed_params)):
            return 6*[np.nan]
        p0 = self.optimize1_model(f, Z, x0, cole_cole_fixed_params)
        Rq, T, a, Ro, To, a0 = p0
        def err(p):
            if any( p < 0 ):
                return 100
            this_p = [Rq, p[0], a, Ro, p[1], a0]
            ZZ = self.impedance_model_opt(f, this_p, cole_cole_fixed_params)
            return sum(np.abs(Z - ZZ)**2)/sum(np.abs(Z)**2)
    
        T_To = optimize.least_squares(err, [T, To], bounds=(0, np.inf))
        return [Rq, T_To.x[0], a, Ro, T_To.x[1], a0]

    def get_linear_slope(self, x, y):
        err = lambda slope: sum( (slope*x - y)**2 )
        opt = optimize.least_squares(err, 1, bounds=(0, np.inf))
        return opt.x[0]

    def model(self, i, f, Z, label, f_hight=25000, f_middle=10000, 
                       f_low=0.11, f_=np.logspace(-2, 8, 2000), f2=np.logspace(-5, 8, 2000)):
        
        Xi, DX, tau, alpha = self.get_cole_params(f[f>f_hight], Z[f>f_hight])
        Z_ = self.cole_cole(f_, Xi, DX, tau, alpha) # Xi + DX / ( 1 + (2j*np.pi*f_*tau)**(1-alpha) )#
        
        x = np.real(Z)[f<f_low]
        y = -np.imag(Z)[f<f_low]
        p = np.polyfit(x, y, 1)
        x_ = np.linspace(-p[1]/p[0], max(x), 10)
    
        x1 = np.real(Z)[(f=>f_low) & (f<=f_middle)]
        y1 = -np.imag(Z)[(f=>f_low) & (f<=f_middle)]
        slope1 = self.get_linear_slope(x1, y1)
        p1 = [slope1, 0]
        #p1 = np.polyfit(x1, y1, 1)
        print('Medium frequency phenomena (slope, intercept)', p1)

        C = 1 / (2j * np.pi * f * Z)

        # Params based on physical information
        Ci, DC, tau_c, alpha_c = self.get_cole_params(f, C)
        cole_cole_fixed_params = [Xi, DX, tau, alpha]
        CPE_params = [max(np.real(Z)), 1, 2 * np.arctan(1/slope1) / np.pi] # CPE start parameters: Zo, tau, alpha
        x0 = CPE_params + [tau_c/(DC+Ci), tau_c, alpha_c] # initial params to the model CPE_params + [ Ro, tau, alpha_c ]
    
        p_model = self.optimize_model(f, Z, x0, cole_cole_fixed_params)
        opt_params = np.concatenate([cole_cole_fixed_params, p_model])

        if any(np.isnan(opt_params)):
            return 10*[np.nan]
        
        Z2 = self.impedance_model(f_, opt_params) # data of the model

        Zmodel = self.impedance_model(f, opt_params) # data of the model
        Zdata = Z
        error_real = np.sum((np.real(Zdata)-np.real(Zmodel))**2) / np.sum(np.real(Zdata)**2)
        error_imag = np.sum((-np.imag(Zdata)-(-np.imag(Zmodel)))**2) / np.sum((-np.imag(Zdata))**2)
        error_Z = np.sqrt(np.sum(error_real**2 + error_imag**2))
    
        C_ = self.cole_cole(f2, Ci, DC, tau_c, alpha_c)
        C2 = 1 / (2j * np.pi * f_ * Z2)
        C1 = 1 / (2j * np.pi * f_ * Z_)

        return opt_params, Z2, f_, error_Z
