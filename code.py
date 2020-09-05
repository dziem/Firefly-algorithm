import random
import numpy as np
import math

def fungsi1(z):
  x = z[0]
  y = z[1]
  a1 = 0
  a2 = 0
  for i in range(1,6):
    a1 += i * math.cos(math.radians((i + 1) * x + 1))
    a2 += i * math.cos(math.radians((i + 1) * y + 1))
  a1 = a1 * -1
  return a1 * a2

def fungsi2(z):
  x = z[0]
  y = z[1]
  a = -1 * math.cos(math.radians(x))
  b = math.cos(math.radians(y))
  c = math.exp((-1 * ((x - math.pi) ** 2)) - ((y - math.pi) ** 2))
  return a * b * c

def fitness1(x):
  return 2**(-fungsi1(x))

def fitness2(x):
  return 2**(-fungsi2(x))

def euclidean(a, b):
  sm = ((a[0] - b[0])**2) + ((a[1] - b[1])**2)
  return np.sqrt(sm)

def attractiveness(beta0, gamma, r):
  if r == 0:
    beta0 = 0
  return beta0*np.exp(-gamma*(r**2))

# FUNGSI 1
# dirun sebanyak 5 kali
for n in range(5):
  # parameter FA
  max_gen = 20
  gamma = 0.5/(200**2)
  beta0 = 1
  alpha = 0.01*200
  n_population = 40

  # inisialisasi populasi sebanyak n individu
  population = np.zeros(shape=(n_population, 3))

  for i in range(n_population):
    x1 = random.uniform(-100,100)
    x2 = random.uniform(-100,100)
    I = fitness1([x1,x2])
    population[i] = ([x1, x2, I])

  # inisialisasi best individu 
  best_ind_index = np.argmax(population[:,-1])
  best_ind = population[best_ind_index]

  for i in range(max_gen):
    for j in range(len(population)):
      b = True # tidak ada individu yang lebih baik dari j diset True
      for k in range(len(population)):
        # attractiveness
        r = euclidean(population[j,:-1], population[k,:-1])
        attr = attractiveness(beta0, gamma, r)

        if population[k,-1] > population[j,-1] :
          b = False
          population[j,0] += (attr*(population[k,0]-population[j,0])) + (alpha*(random.uniform(0,1) - 0.5))
          population[j,1] += (attr*(population[k,1]-population[j,1])) + (alpha*(random.uniform(0,1) - 0.5))
          population[j,-1] = fitness1(population[j,:-1])

      # jika tidak ada individu k yang lebih baik dari individu j
      if b == True:
        population[j,0] = random.uniform(-100,100)
        population[j,1] = random.uniform(-100,100)
        population[j,-1] = fitness1(population[j,:-1])
      
      # mengambil solusi terbaik sementara
      best_ind_index_temp = np.argmax(population[:,-1])
      best_ind_temp = population[best_ind_index_temp]
      if best_ind_temp[-1] > best_ind[-1]:
        best_ind = best_ind_temp
    
    alpha *= 0.97**i 

  # solusi FA pada kasus minimasi fungsi
  print("Solusi Fungsi 1, Percobaan ke-"+str(n+1))
  print(best_ind[:-1])
  print(fungsi1(best_ind))

# FUNGSI 1
# dirun sebanyak 5 kali
for n in range(5):
  # parameter FA
  max_gen = 20
  gamma = 0.5/(200**2)
  beta0 = 1
  alpha = 0.01*200
  n_population = 40
  
  # inisialisasi populasi sebanyak n individu
  population = np.zeros(shape=(n_population, 3))

  for i in range(n_population):
    x1 = random.uniform(-100,100)
    x2 = random.uniform(-100,100)
    I = fitness2([x1,x2])
    population[i] = ([x1, x2, I])

  # inisialisasi best individu 
  best_ind_index = np.argmax(population[:,-1])
  best_ind = population[best_ind_index]

  for i in range(max_gen):
    for j in range(len(population)):
      b = True # tidak ada individu yang lebih baik dari j diset True
      for k in range(len(population)):
        # attractiveness
        r = euclidean(population[j,:-1], population[k,:-1])
        attr = attractiveness(beta0, gamma, r)

        if population[k,-1] > population[j,-1] :
          b = False
          population[j,0] += (attr*(population[k,0]-population[j,0])) + (alpha*(random.uniform(0,1) - 0.5))
          population[j,1] += (attr*(population[k,1]-population[j,1])) + (alpha*(random.uniform(0,1) - 0.5))
          population[j,-1] = fitness2(population[j,:-1])
      # jika tidak ada individu k yang lebih baik dari individu j
      if b == True:
        population[j,0] = random.uniform(-100,100)
        population[j,1] = random.uniform(-100,100)
        population[j,-1] = fitness2(population[j,:-1])
      
      # mengambil solusi terbaik sementara
      best_ind_index_temp = np.argmax(population[:,-1])
      best_ind_temp = population[best_ind_index_temp]
      if best_ind_temp[-1] > best_ind[-1]:
        best_ind = best_ind_temp
    
    alpha *= 0.97**i 

  # solusi FA pada kasus minimasi fungsi
  print("Solusi Fungsi 2, Percobaan ke-"+str(n+1))
  print(best_ind[:-1])
  print(fungsi2(best_ind))