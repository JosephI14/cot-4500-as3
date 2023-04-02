import numpy as np


np.set_printoptions(precision=7, suppress=True, linewidth=100)

def function(t: float, w: float):
  return t - (w**2)
  
# Question 1
def euler_method(initial_point, start_of_t, end_of_t, num_of_iterations):


  h = (end_of_t - start_of_t) / num_of_iterations
  t = start_of_t
  y = initial_point

  for cur_iteration in range(0, num_of_iterations):

    w0 = y
    t0 = t
    
    y = w0 + (h * function(t0,w0))
    t = t0 + h

  return y
  
# Question 2
def runge_kutta(start_of_t, end_of_t, x, h):
    
    n = (int)((x - start_of_t)/h)
    
    y = end_of_t
    for i in range(1, n + 1):
        
        k1 = h * function(start_of_t, y)
        k2 = h * function(start_of_t + 0.5 * h, y + 0.5 * k1)
        k3 = h * function(start_of_t + 0.5 * h, y + 0.5 * k2)
        k4 = h * function(start_of_t + h, y + k3)
 
        
        y = y + (1.0 / 6.0)*(k1 + 2 * k2 + 2 * k3 + k4)
 
        
        start_of_t = start_of_t + h
    return y
 

# Question 3
def gauss_jordan(mat_aug):
    n = len(mat_aug)
     
    for i in range(n):
        
        val = i
        for j in range(i+1, n):
            if abs(mat_aug[j][i]) > abs(mat_aug[val][i]):
                val = j
                
        
        mat_aug[i], mat_aug[val] = mat_aug[val], mat_aug[i]
        
        
        for j in range(i+1, n):
            ops = mat_aug[j][i] / mat_aug[i][i]
            for k in range(i, n+1):
                mat_aug[j][k] -= ops * mat_aug[i][k]
    
    
    x = [0] * n
    for i in range(n-1, -1, -1):
        x[i] = mat_aug[i][n] / mat_aug[i][i]
        for j in range(i-1, -1, -1):
            mat_aug[j][n] -= mat_aug[j][i] * x[i]
    
    return x
  
# Question 4
def LU_factorization(LU_mat):
    
    z = LU_mat.shape[0]
  
    L = np.eye(z)
    U = np.zeros((z, z))
    
    
    for i in range(z):
        
        U[i, i:] = LU_mat[i, i:] - L[i,:i] @ U[:i,i:]
        L[(i+1):,i] = (LU_mat[(i+1):,i] - L[(i+1):,:] @ U[:,i]) / U[i, i]
    deter = np.linalg.det(U)
    return L, U, deter
  
# Question 5
def diagonally_dominate(mat_dom):
  
  n = len(mat_dom)
  for i in range(n):
    row_sum = 0
    for j in range(n):
      row_sum = row_sum + abs(mat_dom[i][j])

    row_sum = row_sum - abs(mat_dom[i][j])

    if (abs(mat_dom[i][i]) < row_sum):
      return False
  return True

  return

# Question 6
def positive_definite(mat_def):
  return np.all(np.linalg.eigvals(mat_def) > 0)
  
    
  

if __name__ == "__main__":
  initial_point = 1
  start_of_t = 0
  end_of_t = 2
  num_of_iterations = 10
  print("%.5f" %euler_method(initial_point, start_of_t, end_of_t, num_of_iterations))
  print()

  
  start_of_t = 0
  y = 1
  x = 2
  h = 0.2
  print("%.5f" %runge_kutta(start_of_t, y, x, h), end="\n\n")

  
  mat_aug = ([[2,-1,1,6], [1,3,1,0], [-1,5,4,-3]])
  print(gauss_jordan(mat_aug), end="\n\n")
  

  LU_mat = np.array([[1, 1, 0, 3], [2, 1, -1, 1], [3, -1, -1, 2],   [-1, 2, 3, -1]])
  L, U, deter = LU_factorization(LU_mat)
  print('%.5f' %deter, end="\n\n")
  print(L, end="\n\n")
  print(U, end="\n\n")

  mat_dom = [[9, 0, 5, 2, 1], [3, 9, 1, 2, 1], [0, 1, 7, 2, 3], [4, 2, 3, 12, 2], [3, 2, 4, 0, 8]]
  print(diagonally_dominate(mat_dom), end="\n\n")

  mat_def = [[2, 2, 1], [2, 3, 0], [1, 0, 2]]
  print(positive_definite(mat_def), end="\n\n")
