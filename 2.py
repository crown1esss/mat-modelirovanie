
import numpy as np

def mu(massiv):
  N = 3
  z = np.array(massiv , dtype = 'complex_')
  for i in range(N-1):
    for j in range(i,N):
      mu = z[i] + z[j] / 2 + (complex(0,1) * np.imag(z[i] * np.conj(z[j])) * (z[i] - z[j])) / (2 * (np.abs(z[i] * np.conj(z[j]))) + np.real(z[i] * np.conj(z[j])))
      R = np.sqrt((np.abs(z[i] - z[j]) * np.abs(z[i] - z[j]))) * np.abs(np.conj(z[i]) * z[j]) / (2 * (np.abs(np.conj(z[i]) * z[j])) + np.real(np.conj(z[i]) * z[j]))
      t1 = 0
      mu_res1 = 0
      for k in range(N):
        if np.any(((mu-z[k])>R)):
          t1 = 1
        if (t1==0):
          mu_res1 = mu
  for i in range(N-1):
    for j in range(N-i):
      for k in range(N-j):
        t2 = 0
        mu_res2 = 0
        if (2 * np.imag(z[i] * np.conj(z[j]) + z[j] * np.conj(z[k]) + z[k] * np.conj(z[i]))) != 0: 
          mu = (complex(0,1) * ((np.abs(z[i]) * np.abs(z[i]) * (z[j] - z[k]) + np.abs(z[j]) * np.abs(z[j]) * (z[k] - z[i]) +np.abs(z[k]) * np.abs(z[k]) * (z[i] - z[j])) / (2 * np.imag(z[i] * np.conj(z[j]) + z[j] * np.conj(z[k]) + z[k] * np.conj(z[i])))))
          R = np.sqrt(np.abs(z - mu) * np.abs(z - mu))
          print(R)
    
          return mu 



def iter(H ,f , mu ):
  eps = eps=10e-8
  n_iter = 10000
  U=np.random.normal(0,1,H.shape[0])
  for i in range(n_iter):
    U_mu = U - 1/mu * (H*U - f) 
    dif = U - U_mu
    if (np.sqrt(dif@dif))/(np.sqrt(f@f))<eps:
      break
    U=U_mu.copy()
  return U_mu
def test(massiv):
  return np.linalg.solve(np.diag(massiv), np.array([1,2,3]))
def main():
    H = np.array([5 + 0j + 2j, 10 + 5j + 1j, 10 - 5j + 1j])
    f = np.array([1,2,3])
    m = mu(H)
    iteration = iter(H,f , mu=m)
    testing = test(H)
    print(m)
    print('iteration')
    print(iteration)
    print('testing')
    print(testing)


if __name__ == '__main__':
    main()