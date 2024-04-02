#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 20:05:48 2024

@author: janecohen
"""

#%% Imports

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

#%% ODE solvers

"Runge-Kutta ODE solver"
def RungeKutta(f,y,t,h):
    k1 = h*f(y,t)
    k2 = h*f(y+k1/2, t+h/2)
    k3 = h*f(y+k2/2, t+h/2)
    k4 = h*f(y+k3, t+h)
    y = y + (1/6)*(k1+2*k2+2*k3+k4)
    return y 

"Velocity-Verlet ODE solver"
def leapfrog(diffeqn, y, t, h):
    hh = h/2
    y[0] = y[0] + y[1]*hh
    y[1] = y[1] + diffeqn(y, t+hh)[1] * h
    y[0] = y[0] + y[1]*hh
    return y   
    
#%% Equations

"Simple Harmonic Oscillator"
def SHO(y, t):
    return np.array([y[1], -y[0]]) #[velocity, acceleration]


def energy(y):
    KE = 0.5*y[1]**2 # kinetic energy
    PE = 0.5*y[0]**2 # potential energy
    return KE+PE


#%% Run Solver Function

def solve(N, h, solver, tend):
    tlist = np.arange(0.0, tend, h) # list of time intervals
    npts = len(tlist) # number of points
    y = np.zeros((npts,2)) # 2D array for position and velocity
    E = np.zeros((npts,2)) # 2D array for energy values
    
    # initial conditions [position, velocity]
    y0 = [1,0]
    y[0,:] = y0
    
    # call solver
    for i in range(1,npts):   # loop over time
        y0 = solver(SHO, y0, tlist[i-1], h)
        y[i,:]= y0
        E[i,:] = energy(y0)
        
    # returns
    x = y[:,0] 
    v = y[:,1]
    
    return x, v, tlist, E

#%% Graphics

def plotStatic(xRK,vRK, xLF, vLF, t, ERK, ELF, colors):
    plt.rcParams.update({'font.size': 14})
    plt.rcParams['figure.dpi']= 120
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(7, 10))

    ax1.plot(xRK, vRK, color = colors[0], linewidth=2)
    ax1.set(xlabel='$x$(m)', ylabel='$v$(m/s)', title='Runge-Kutta')
    ax1.grid('True')

    
    ax2.plot(t[1:], ERK[1:], color = colors[1], linewidth=2)
    ax2.set(xlabel='$t/T_0$', ylabel='$E$(J)', title='Runge-Kutta')
    ax2.set_ylim(np.min(ERK[1:])-0.1, np.max(ERK[1:])+0.1)
    ax2.grid('True')
    
    ax3.plot(xLF, vLF, color = colors[0], linewidth=2)
    ax3.set(xlabel='$x$(m)', ylabel='$v$(m/s)', title='Leapfrog')
    ax3.grid('True')

    
    ax4.plot(t[1:], ELF[1:], color = colors[1], linewidth=2)
    ax4.set(xlabel='$t/T_0$', ylabel='$E$(J)', title='Leapfrog')
    ax4.set_ylim(np.min(ELF[1:])-0.1, np.max(ELF[1:])+0.1)
    ax4.grid('True')
    
    
    plt.tight_layout()
    plt.show()

    
#%% Animations

def run_animation(x1, x2, v1, v2, colors):
    # set up plot window with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    ax1.set_xlim(-2, 2)
    ax1.set_ylim(-2, 2)
    ax2.set_xlim(-2, 2)
    ax2.set_ylim(-2, 2)
    
    ax1.set_title('Runge-Kutta')
    ax1.set_xlabel('x')
    ax1.set_ylabel('v')
    ax2.set_title("Leapfrog")
    ax2.set_xlabel('x')
    ax2.set_ylabel('v')

    # create empty lines for each subplot
    line1, = ax1.plot([], [], color=colors[0], lw=2)
    line2, = ax2.plot([], [], color=colors[1], lw=2)
    
    # initialize circle markers at the head of the lines
    circle1, = ax1.plot([], [], marker='o', markersize=5, color=colors[0])
    circle2, = ax2.plot([], [], marker='o', markersize=5, color=colors[1])

    # animation function
    def update(i):
        
        line1.set_data(x1[:i], v1[:i])
        line2.set_data(x2[:i], v2[:i])
        
        # Update circle marker positions
        circle1.set_data([x1[i]], [v1[i]])
        circle2.set_data([x2[i]], [v2[i]])
        
        return line1, line2, circle1, circle2,

    # calling the animation function
    global anim
    anim = animation.FuncAnimation(fig, update, frames=len(x1), interval=50, blit=True)
    
    plt.show()

#%% USER CODE 

"Edit parameters and run cell to store results"

# period (in normalized units)
T0 = 2*np.pi 

# number of periods
N = 10

tmax = N*2*np.pi # N number of periods

# step size (in units of period)
h = 0.1*T0

static_graphs = True # turn on or off 2d static graphs
download_graphics = False # turn on or off to download static graphs
animate = True # turn on or off animation
download_movie = False # turn on or off movie

#Runge-Kutta
xRK, vRK, t, ERK = solve(N, h, RungeKutta, tmax)

#Leap frog
xLF, vLF, t, ELF= solve(N, h, leapfrog, tmax)

#%%

"Run cell to display plots, animations, etc."

colors = ['palevioletred', 'cornflowerblue']

if (static_graphs == True):
    plotStatic(xRK,vRK,xLF,vLF,t,ERK,ELF, colors)
    
if(download_graphics == True):
    plt.savefig('./Desktop/ENPH479/A2Q1.pdf', format='pdf', dpi=1200, bbox_inches = 'tight')

if (animate == True):
    movie = run_animation(xRK, xLF, vRK, vLF, colors)
    
if (animate == True and download_movie == True):
    f = "/Users/janecohen/Desktop/ENPH479/Assignment2/3bodyProblem.gif"
    movie.save(filename=f, writer="pillow")
    

