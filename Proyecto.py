#!/usr/bin/env python
# coding: utf-8

# In[512]:


import numpy as np
import matplotlib.pyplot as plt
from sympy import *
import math as m


# # Ejercicio 1

# In[513]:


def fr(r):
    return (r**2)/((R**2+r**2)**(3/2))

def fz(r):
    return (r**3)/((R**2 + r**2)**(3/2))

mu=4*np.pi*10**-7
sigma=1
frec=1
const=((mu*sigma*frec)/2)
R=3


# ### Montecarlo

# In[514]:


min_r=0
max_r=R
min_y=0
#max_yfr=fr(max_r)
max_yfz=fz(max_r)
n=100000


# In[515]:


"""""""""""
##Función r
random_r=np.random.rand(n)*(max_r-min_r)-min_r
random_y=np.random.rand(n)*(max_yfr-min_y)-min_y

plt.scatter(random_r,random_y,alpha=0.2,color="gray")
plt.xlim(0, R)
plt.ylim(0, 0.6)

index=np.where(fr(random_r)-random_y>0)
plt.scatter(random_r[index], random_y[index],color="pink")
plt.ylim(0, 0.6)

rline=np.linspace(-2,8,100)
yline=fr(rline)
plt.plot(rline,yline)

interval_integral=(max_yfr)*(max_r)
integral=interval_integral*(np.size(index)/(np.size(random_y)))
print(integral)
integral_fr=integral*2*np.pi*R
print(integral_fr)
"""""""""""


# In[516]:


##Función z
random_r=np.random.rand(n)*(max_r-min_r)-min_r
random_y=np.random.rand(n)*(max_yfz-min_y)-min_y

print(monte_fz(100))
plt.scatter(random_r,random_y,alpha=0.2,color="gray")
plt.xlim(0, R)
plt.ylim(0, 0.6)

index=np.where(fz(random_r)-random_y>0)
plt.scatter(random_r[index], random_y[index],color="pink")
plt.ylim(0, 0.6)

rline=np.linspace(-2,8,100)
yline=fz(rline)
plt.plot(rline,yline)

interval_integral=(max_yfz)*(max_r)
integral_fz=interval_integral*(np.size(index)/(np.size(random_y)))
print(integral_fz)


# In[517]:


def monte_points(n):
    random_r=np.random.rand(n)*(max_r-min_r)-min_r
    random_y=np.random.rand(n)*(max_yfz-min_y)-min_y
    return random_r,random_y

lista=[]
for i in range(10000):
    points=monte_points(n)
    rr=points[0]
    ry=points[1]
    index=np.where(fz(rr)-ry>0)
    interv_integral=(max_yfz)*(max_r)
    integral_fz1=interv_integral*(np.size(index)/(np.size(ry)))
    lista.append(integral_fz1)

monte_promedio=np.mean(lista)
print(monte_promedio)


# In[518]:


##Valor final calculado montecarlo
const*monte_promedio


# ### Trapecio

# In[519]:


def trap_integrate(fz, min_r, max_r, n):
    x, h = np.linspace(min_r, max_r, num=n-1, retstep = True)
    return 0.5*h*(fz(x[0]) + fz(x[-1])) + h*np.sum(fz(x[1:-1]))

integral_tr=trap_integrate(fz, min_r, max_r, n)
print(integral_tr)


# In[520]:


ng=100


# In[521]:


rline=np.linspace(0,R,ng+1)
yline=fz(rline)
plt.xlim(0, R)
plt.ylim(0, 0.6)

plt.plot(rline,yline)

for i in range(ng):
    xs = [rline[i],rline[i],rline[i+1],rline[i+1]]
    ys = [0,fz(rline[i]),fz(rline[i+1]),0]
    plt.fill(xs,ys,'b',edgecolor='b',alpha=0.2)


# In[522]:


##Valor final calculado trapecio
const*integral_tr


# ### Valores esperados

# In[523]:


""""""""""
##Valor esperado r
r=Symbol("r")
integrate((r**2)/((R**2+r**2)**(Rational("3/2"))), (r,0,R))
"""""""""""


# In[524]:


##Valor esperado z
r=Symbol("r")
integrate((r**3)/((R**2+r**2)**(Rational("3/2"))), (r,0,R))


# In[525]:


#Valor esperado final
print(0.3639610307*const)


# # Ejercicio 2

# In[526]:


def f(xy):
    return (1/((z**2+xy**2)**(3/2)))

z=1
mu=1
I=1
const=(z*mu*I)/(4*np.pi)


# In[527]:


min_xy=0
max_xy=10
min_=0
max_=f(0)
n=100000


# ### Montecarlo

# In[538]:


random_xy=np.random.rand(n)*(max_xy-min_xy)-min_xy
random_=np.random.rand(n)*(max_-min_)-min_

plt.scatter(random_xy,random_,alpha=0.2,color="gray")
plt.xlim(0, 9)
plt.ylim(0, 1.25)

index=np.where(f(random_xy)-random_>0)
plt.scatter(random_xy[index], random_[index],color="pink")
plt.ylim(0, 1.25)

xyline=np.linspace(-8,8,100)
line=f(xyline)
plt.plot(xyline,line)

interval_integral=(max_)*(max_xy)
integral=interval_integral*(np.size(index)/(np.size(random_y)))
print(integral)


# In[529]:


def monte_points(n):
    random_xy=np.random.rand(n)*(max_xy-min_xy)-min_xy
    random_=np.random.rand(n)*(max_-min_)-min_
    return random_xy,random_

lista=[]
for i in range(10000):
    points=monte_points(n)
    rxy=points[0]
    r=points[1]
    index=np.where(f(rxy)-r>0)
    interv_integral=(max_)*(max_xy)
    integral_f1=interv_integral*(np.size(index)/(np.size(r)))
    lista.append(integral_f1)

monte_promedio=np.mean(lista)
print(monte_promedio)


# In[530]:


##Valor final calculado montecarlo
const*monte_promedio


# ### Trapecio 

# In[531]:


def trap_integrate(f, min_xy, max_xy, n):
    x, h = np.linspace(min_xy, max_xy, num=n-1, retstep = True)
    return 0.5*h*(f(x[0]) + f(x[-1])) + h*np.sum(f(x[1:-1]))

integral=trap_integrate(f, min_xy, max_xy, n)
print(integral)


# In[532]:


xyline=np.linspace(-2,max_xy,ng+1)
line=fz(rline)
plt.xlim(0, max_xy)
plt.ylim(0, 1.6)

line=f(xyline)
plt.plot(xyline,line)

for i in range(ng):
    xs = [xyline[i],xyline[i],xyline[i+1],xyline[i+1]]
    ys = [0,f(xyline[i]),f(xyline[i+1]),0]
    plt.fill(xs,ys,'b',edgecolor='b',alpha=0.2)


# In[533]:


##Valor final calculado trapecio
const*integral


# ### Valor esperado

# In[534]:


##Valor esperado de la integral
xy=Symbol("xy")
integrate(1/((z**2+xy**2)**(Rational("3/2"))), (xy,0,oo))


# In[535]:


##Valor esperado final
const*1


# In[ ]:




