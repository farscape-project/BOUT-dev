import sys
import os

import numpy as np

def initFlow(npv):

  print(sys.version)
  print(sys.path)
  print("PYTHONPATH:", os.environ.get('PYTHONPATH'))

  SIZE = 256

  n    = np.zeros((SIZE, SIZE))
  phi  = np.zeros((SIZE, SIZE))
  vort = np.zeros((SIZE, SIZE))
  rnpv  = np.zeros(3*SIZE*SIZE, dtype="float64")

  cont=0
  for i in range(SIZE):
      for k in range(SIZE):
        rnpv[cont] = 0.001 #n[i,k]
        cont = cont+1
        rnpv[cont] = 0.001 #phi[i,k]
        cont = cont+1
        rnpv[cont] = 0.001 #vort[i,k]

  print("Called initFlow!")

  return rnpv


def findLESTerms(n):
  print(n)
  return n
