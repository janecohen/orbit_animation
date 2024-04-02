#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 09:30:25 2024

@author: janecohen
"""

#%% Imports

import numpy as np
import matplotlib.pyplot as plt
import math as m
from scipy.optimize import fsolve
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
    split = len(y)//2
    r0 = y[:split]
    v0 = y[split:]
    hh = h/2
    r1 = r0 + hh * v0
    v1 = v0 + h * diffeqn(y, t+hh)[split:]
    r1 = r1 + hh * v1
    yResult = np.concatenate((r1,v1))
    return yResult  

    
#%% Equations
    
"Euler's Quintic Equation"
EulerQuintic = lambda x: x**5*(m[1]+m[2]) + x**4*(2*m[1]+3*m[2]) + x**3*(m[1]+3*m[2]) + x**2*(-3*m[0]-m[1]) + x*(-3*m[0]-2*m[1]) + (-m[0]-m[1])

"Equations of Motion for Three Bodies"
def EoM(y, t):
    # y = [r1x, r1y, r2x, r2y, r3x, r3y, v1x, v1y, v2x, v2y, v3x, v3y]
    
    # seperate position and velocity
    split = len(y) // 2
    r = np.resize(y[:split], (3,2)) #r = [r1x, r1y, r2x, r2y, r3x, r3y]
    v = y[split:].astype(float) #v = [v1x, v1y, v2x, v2y, v3x, v3y]
    a = np.zeros([3,2]) # acceleration (dv/dt)
    
    for i in range(len(r)):
        for j in range(len(r)):
            if i != j:
                ri = r[i]
                rj = r[j]
                rij = ri-rj 
                a[i] = a[i] -G * m[j] * rij / np.linalg.norm(rij)**3 
    
    #[velocity, acceleration]
    yResult = np.concatenate((v, a.flatten()))
    return yResult


#%% Initial conditions calculator

def general_init_cond(m):
    "Find root of Euler's quintic equation" 
    lam = fsolve(EulerQuintic, 1)[0] # lambda = root of Eulers quintic equation

    "Find distance a = x3 − x2"
    a = ((1/w**2) * m[1] + m[2] - m[0]*(1+2*lam) / (lam**2*(1+lam)**2.)) **(1/3)

    "Initial conditions"
    r, v = np.zeros((3,2)), np.zeros((3,2))
    r[1,0] = (m[0]/lam**2-m[2])/(w**2 * a**2) # x2
    r[0,0] = r[1,0]-lam*a # x1
    r[2,0] = -(m[0]*r[0,0] + m[1]*r[1,0])/m[2] # x3
    v[0,1] =  w*r[0,0] # v1y 
    v[1,1] =  w*r[1,0] # v2y 
    v[2,1] =  w*r[2,0] # v3y 
    return r, v

# Fixed initial condition (1)
def init_cond1():
    r, v = np.zeros((3,2)), np.zeros((3,2))
    # initial r and v - set 1
    r[0,0] = -0.30805788; v[0,1] = -1.015378093 # x1, v1y
    r[1,0] = 0.15402894; r[1,1] = -0.09324743 # x2, y2
    v[1,0] = 0.963502817 ; v[1,1] = 0.507689046 # v2x, v2y
    r[2,0] = r[1,0]; r[2,1] = -r[1,1] # x3, y3
    v[2,0] = -v[1,0]; v[2,1] = v[1,1] # v3x, v3y
    return r, v

# Fixed initial condition (2)
def init_cond2():
    r, v = np.zeros((3,2)), np.zeros((3,2))
    # initial r and v - set 2
    r[0,0] = 0.97000436; r[0,1] = -0.24308753 # x1, y1
    v[2,0] = -0.93240737; v[2,1] = -0.86473146 # v3x, v3y
    v[0,0] = -v[2,0]/2.; v[0,1] = -v[2,1]/2. # v1x, v1y
    r[1,0] = -r[0,0]; r[1,1] = -r[0,1] # x2, y2
    v[1,0] = v[0,0]; v[1,1] = v[0,1] # v2x, v2y
    return r, v


#%% 3 body problem 

def run_solver(solver, h, m, r0, v0, tend):
    
    tlist = np.arange(0.0, tend, h) # list of time intervals
    npts = len(tlist) # number of points

    #y = [r1x, r1y, r2x, r2y, r3x, r3y, v1x, v1y, v2x, v2y, v3x, v3y]
    y0 =  np.concatenate((r0.flatten(),v0.flatten()))
    y = np.zeros((npts,len(y0)), dtype='object')
    y[0,:] = y0
 
    for i in range(1,npts):   # loop over time
        y0 = solver(EoM, y0, tlist[i-1], 0.001)
        y[i,:]= y0
        
    split = len(y[0]) // 2
    r = y[:, :split] #r = [r1x, r1y, r2x, r2y, r3x, r3y]
    v = y[:, split:].astype(float)
    
    return r, v, tlist

#%% Flip velocity

#flip the direction of the velocities half way through the simulation
def flip_velocity(v1, v2):
    split = len(v1)//2
    v1[split,:] *= (-1)
    v2[split,:] *= (-1)
    

#%% Graphics

def plotOrbits(rRK, vRK, rLF, vLF, t, colors, mass):  
    plt.rcParams.update({'font.size': 14})
    plt.rcParams['figure.dpi']= 120

    # create a figure 
    fig, (ax1,ax2) = plt.subplots(1, 2, figsize=(9,5))
    
    # plot initial positions
    ax1.plot(rRK[0,0],rRK[0,1], color=colors[0], marker='*')
    ax1.plot(rRK[0,2],rRK[0,3], color=colors[1], marker='*')
    ax1.plot(rRK[0,4],rRK[0,5], color=colors[2], marker='*')
    ax2.plot(rLF[0,0],rLF[0,1], color=colors[0], marker='*')
    ax2.plot(rLF[0,2],rLF[0,3], color=colors[1], marker='*')
    ax2.plot(rLF[0,4],rLF[0,5], color=colors[2], marker='*')
    
    # plot final positions
    ax1.plot(rRK[-1,0],rRK[-1,1], color=colors[0], marker='o', markersize=5*mass[0])
    ax1.plot(rRK[-1,2],rRK[-1,3], color=colors[1], marker='o', markersize=5*mass[1])
    ax1.plot(rRK[-1,4],rRK[-1,5], color=colors[2], marker='o', markersize=5*mass[2])
    ax2.plot(rLF[-1,0],rLF[-1,1], color=colors[0], marker='o', markersize=5*mass[0])
    ax2.plot(rLF[-1,2],rLF[-1,3], color=colors[1], marker='o', markersize=5*mass[1])
    ax2.plot(rLF[-1,4],rLF[-1,5], color=colors[2], marker='o', markersize=5*mass[2])
    
    ax1.plot(rRK[:,0],rRK[:,1], color=colors[0], label="Mass 1")
    ax1.plot(rRK[:,2],rRK[:,3], color=colors[1], label="Mass 2")
    ax1.plot(rRK[:,4],rRK[:,5], color=colors[2], label="Mass 3")
    ax1.set(xlabel='x', ylabel='y', title='Runge-Kutta')
    ax1.legend(loc='best')
    ax1.grid('True')
    
    ax2.plot(rLF[:,0],rLF[:,1], color=colors[0], label="Mass 1")
    ax2.plot(rLF[:,2],rLF[:,3], color=colors[1], label="Mass 2")
    ax2.plot(rLF[:,4],rLF[:,5], color=colors[2], label="Mass 3")
    ax2.set(xlabel='x', ylabel='y', title='Leapfrog')
    ax2.legend(loc='best')
    ax2.grid('True')

    plt.tight_layout()
    plt.show() 


#%% Animations- generalized -  WORKING

def run_animation(r1, v1, r2, v2, colors, mass, intervals):
    
    # set up plot window with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    ax1.set_xlim(-3, 3)
    ax1.set_ylim(-3, 3)
    ax2.set_xlim(-3, 3)
    ax2.set_ylim(-3, 3)
    
    ax1.set_title('Runge-Kutta')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax2.set_title("Leapfrog")
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')

    
    # create empty lines for each subplot
    line1, = ax1.plot([], [], color=colors[0], lw=2, label='Mass 1')
    line2, = ax1.plot([], [], color=colors[1], lw=2, label='Mass 2')
    line3, = ax1.plot([], [], color=colors[2], lw=2, label='Mass 3')
    
    
    line4, = ax2.plot([], [], color=colors[0], lw=2, label='Mass 1')
    line5, = ax2.plot([], [], color=colors[1], lw=2, label='Mass 2')
    line6, = ax2.plot([], [], color=colors[2], lw=2, label='Mass 3')
    #ax2.set(xlabel='$x$', ylabel='$y$', title='Leapfrog')
    
    # initialize circle markers at the head of the lines
    circle1, = ax1.plot([], [], marker='o', markersize=5*mass[0], color=colors[0])
    circle2, = ax1.plot([], [], marker='o', markersize=5*mass[1], color=colors[1])
    circle3, = ax1.plot([], [], marker='o', markersize=5*mass[2], color=colors[2])

    circle4, = ax2.plot([], [], marker='o', markersize=5*mass[0], color=colors[0])
    circle5, = ax2.plot([], [], marker='o', markersize=5*mass[1], color=colors[1])
    circle6, = ax2.plot([], [], marker='o', markersize=5*mass[2], color=colors[2])


    # animation function
    def update(i):
        
        # star markers for initial positions
        ax1.plot(r1[0,0], r1[0,1], marker='*', markersize=5, color=colors[0])
        ax1.plot(r1[0,2], r1[0,3], marker='*', markersize=5, color=colors[1])
        ax1.plot(r1[0,4], r1[0,5], marker='*', markersize=5, color=colors[2])
        
        ax2.plot(r2[0,0], r2[0,1], marker='*', markersize=5, color=colors[0])
        ax2.plot(r2[0,2], r2[0,3], marker='*', markersize=5, color=colors[1])
        ax2.plot(r2[0,4], r2[0,5], marker='*', markersize=5, color=colors[2])
        
        # update for Set 1
        xA1, yA1, xB1, yB1, xC1, yC1 = r1[i,0], r1[i,1], r1[i,2], r1[i,3], r1[i,4], r1[i,5]
        vxA1, vyA1, vxB1, vyB1, vxC1, vyC1 = v1[i,0], v1[i,1], v1[i,2], v1[i,3], v1[i,4], v1[i,5]
        
        line1.set_data(r1[:i,0], r1[:i,1])
        line2.set_data(r1[:i,2], r1[:i,3])
        line3.set_data(r1[:i,4], r1[:i,5])
        line1.set_alpha(0.5)
        line2.set_alpha(0.5)
        line3.set_alpha(0.5)
        
        # update circle marker positions for Set 1
        circle1.set_data([xA1], [yA1])
        circle2.set_data([xB1], [yB1])
        circle3.set_data([xC1], [yC1])

        # update for Set 2
        xA2, yA2, xB2, yB2, xC2, yC2 = r2[i,0], r2[i,1], r2[i,2], r2[i,3], r2[i,4], r2[i,5]
        vxA2, vyA2, vxB2, vyB2, vxC2, vyC2 = v2[i,0], v2[i,1], v2[i,2], v2[i,3], v2[i,4], v2[i,5]
        
        line4.set_data(r2[:i,0], r2[:i,1])
        line5.set_data(r2[:i,2], r2[:i,3])
        line6.set_data(r2[:i,4], r2[:i,5])
        line4.set_alpha(0.5)
        line5.set_alpha(0.5)
        line6.set_alpha(0.5)
        
        # update circle marker positions for Set 2
        circle4.set_data([xA2], [yA2])
        circle5.set_data([xB2], [yB2])
        circle6.set_data([xC2], [yC2])
        
        global quiver1  
        global quiver2
        global quiver3
        global quiver4
        global quiver5
        global quiver6

        # create new quiver plots for Set 1
        quiver1 = ax1.quiver(xA1, yA1, vxA1, vyA1, color='dodgerblue', scale=1, scale_units='xy', angles='xy', width=0.005, headwidth=5, headlength=5)
        quiver2 = ax1.quiver(xB1, yB1, vxB1, vyB1, color='darkorange', scale=1, scale_units='xy', angles='xy', width=0.005, headwidth=5, headlength=5)
        quiver3 = ax1.quiver(xC1, yC1, vxC1, vyC1, color='mediumorchid', scale=1, scale_units='xy', angles='xy', width=0.005, headwidth=5, headlength=5)
        
        # create new quiver plots for Set 2
        quiver4 = ax2.quiver(xA2, yA2, vxA2, vyA2, color='dodgerblue', scale=1, scale_units='xy', angles='xy', width=0.005, headwidth=5, headlength=5)
        quiver5 = ax2.quiver(xB2, yB2, vxB2, vyB2, color='darkorange', scale=1, scale_units='xy', angles='xy', width=0.005, headwidth=5, headlength=5)
        quiver6 = ax2.quiver(xC2, yC2, vxC2, vyC2, color='mediumorchid', scale=1, scale_units='xy', angles='xy', width=0.005, headwidth=5, headlength=5)
    
        
        return (line1, line2, line3, quiver1, quiver2, quiver3, circle1, circle2, circle3,
                line4, line5, line6, quiver4, quiver5, quiver6, circle4, circle5, circle6)

    # calling the animation function
    global anim
    anim = animation.FuncAnimation(fig, update, frames=range(0, len(r1[:,0]), 100), interval=intervals, blit=True)
    
    ax1.legend()
    ax2.legend()
    plt.show()
    return anim
    
#%% USER INSTRUCTION

"HI USER!"
"The three cells below will allow you to vary parameters and solve the three body problem"
"Make sure to run cell 2 each time the parameters in cell 1 are changed"

#%% USER: EDIT and RUN THIS CELL TO ADJUST PARAMETERS

"Parameters"

colors = ['dodgerblue', 'darkorange', 'mediumorchid']
G = 1
ep = 0
w0 = 1 + ep # angluar frequency
w = w0
T0 = 2.*np.pi/w0 # period (in normalized units)
h = 0.001 # step size

"User controls"
static_graphs = True # turn on or off 2d static graphs
animate = True # turn on or off animation
download = False # turn on or off to save movie

"Choose initial conditions - set ONLY ONE of the options to 1"
general = 1
fixed_1 = 0
fixed_2 = 0


#%% USER: RUN CELL TO RUN SOLVERS AND STORE RESULTS

"Initial Conditions"
if (general == 1):
    m = np.array([1, 2, 3]) # three body masses
    N = 4 # number of periods/cycles
    tmax = N*2*np.pi # maximum time value (number of cycles * 2pi)
    r_initial, v_initial = general_init_cond(m)
    intervals = 0.1
    
elif(fixed_1 == 1):
    m = np.array([1/3, 1/3, 1/3]) # three body masses
    N = 1/2
    tmax = N*2*np.pi
    r_initial, v_initial = init_cond1()
    intervals = 100
    print("For masses of 1/3, and a nominal period of 1/2:") 
    print("For the RK4 solver, two of the three planets orbit round the other while the third completes a small orbit. Each rotation, the planet in the small orbit changes.")
    print("For the leapfrog solver, this orbit pattern happens initially, but the then planets spin off, showing the instability of the system.")
    
elif(fixed_2 == 1):
    m = np.array([1, 1, 1]) # three body masses
    N = 10
    tmax = N*2*np.pi
    ep = 0.00001
    w0 = 1 + ep # angluar frequency
    r_initial, v_initial = init_cond2()
    intervals = 10
    print("For masses of 1, and a nominal period of 10:") 
    print("For both solvers, the planets move in an orbit shaped like an infinity sign.")
    print("For the number of periods being run, the system appears to be stable in the animation.")
    print("When we observe the static plot from the leapfrog solver, we can see the orbit is changing shape.")
    print("Changing the value of epsilon only effects the leapfrog solver. It makes the change in orbit shape occur more quickly.")
         
else:
    print("No initial condition selected")

"Run Solver"
rRK, vRK, t = run_solver(RungeKutta, h, m, r_initial, v_initial, tmax)
rLF, vLF, t = run_solver(leapfrog, h, m, r_initial, v_initial, tmax)


#%% USER: RUN CELL TO SHOW PLOTS, ANIMATIONS, ETC.

"Run cell to see plots, animations, etc."

if (static_graphs == True):
    plotOrbits(rRK, vRK, rLF, rRK, t, colors, m)
    
if (animate == True):
    intervals = 100
    movie = run_animation(rRK, vRK, rLF, vLF, colors, m, intervals)
    
if (animate == True and download == True):
    f = "/Users/janecohen/Desktop/ENPH479/Assignment2/3bodyProblem.gif"
    movie.save(filename=f, writer="pillow")


