
import numpy as np


# source number
m = 200

# Committee number
N = 7

# Matrix division

# Field size assumed to be prime for this implementation
F = 65537

# Input matrix size - M, X: s by r
s = 1000
r = 1000
# choose exponent of x_i
t = 4


def polynomial_encoding(cf, x):
    # coefficients is a list of matrices A_i0, A_i1, ..., A_i(t-1)
    # x is the variable

    # Initialize the result matrix
    results = np.zeros_like(cf[0])

    # Perform polynomial encoding %F
    for i in range(len(cf)):
        results += cf[i] * (x ** i)
    return results

def coefficients_Ax(s, r, t):
    # Generate random example matrices B_i0, B_i1, ..., B_i(t-1)
    # Initialize the coefficients list with the first matrix having all zeros in the first column
    coefficients = [np.zeros((s, r), dtype=int) for _ in range(t)]

    # Generate random matrices for the remaining coefficients
    for coeff_matrix in coefficients[0:]:
        coeff_matrix[:, 0] = 10  # Set the first column to secret value
        coeff_matrix[:, 1:] = np.random.randint(0, 256, size=(s, r - 1))
    return coefficients

def coefficients_Bx(s, r, t):
    # Generate random example matrices B_i0, B_i1, ..., B_i(t-1)

    # Initialize the coefficients list with the first matrix having all zeros in the first column
    coefficients = [np.zeros((s, r), dtype=int) for _ in range(t)]

    # Generate random matrices for the remaining coefficients
    for coeff_matrix in coefficients[0:]:
        # Set the first column to zero
        coeff_matrix[:, 0] = 0
        # random numbers
        coeff_matrix[:, 1:] = np.random.randint(0, 256, size=(s, r - 1))

    return coefficients

def offline(s, r, t, x_values):
    # Generate random example matrices B_i0=0, B_i1, ..., B_i(t-1)
    cf = coefficients_Bx(s, r, t)
    tuple = []
    for x in x_values:
        a = polynomial_encoding(cf, x)
        tuple.append(a)
    return tuple

def handoff(tuple, res):

    # Print the generated coefficients
    # for i, coeff_matrix in enumerate(cf):
    #    print(f"B_{i}:\n{coeff_matrix}")

    # offine
    # Generate random example matrices B_i0=0, B_i1, ..., B_i(t-1)
    #cf = generate_coefficients_with_zero_first_column(s, r, t)

    #tep = []
    #for x in x_values:
    #    a = polynomial_encoding(cf, x)
    #    tep.append(a)

    res = np.add(res, tuple)

    return res

def refresh(num, res, x_values):
    # Generate random example matrices B_i0=0, B_i1, ..., B_i(t-1)

    cf = coefficients_Bx(num, r, t)
    temp = []
    for x in x_values:
        a = polynomial_encoding(cf, x)
        temp.append(a)
    return temp

def lagrange(x, y, num_points, x_test):
    # Ensure the size of x and y is at least num_points
    if len(x) < num_points or len(y) < num_points:
        raise ValueError("Not enough data points for Lagrange interpolation")
    # All base function values
    l = np.zeros(shape=(num_points,))

    # the value of the kth base function
    for k in range(num_points):
        # init l =1
        l[k] = 1
        # the k_th term in the kth base function
        for k_ in range(num_points):
            if k != k_ and (x[k] - x[k_]) != 0:
                l[k] = l[k] * (x_test - x[k_]) / (x[k] - x[k_])
            else:
                pass

    # the y_test value corresponding to the current x_test to be predicted
    L = 0
    for i in range(num_points):
        L += y[i] * l[i]
    return L


def recover(x_values, results, num_points, x_test_values):
    # Get the number of dimensions
    if x_values.size == 0:
        raise ValueError("x_values is empty.")
    num_dimensions = results.shape[1]

    # Store the interpolated results in a matrix
    interpolated_results = np.zeros_like(results)

    # Perform interpolation for each dimension
    x_test = x_test_values
 #   if(x_test_values>xm.max() or x_test_values<xm.min()):
 #       raise ValueError("x_values is out of scope.")
    for dim in range(num_dimensions):
        y = results[:, dim]
        # Use Lagrange interpolation for prediction
        interpolated_results[:, dim] = lagrange(x_values, y, num_points, x_test)
    return interpolated_results[:, :, 0]

def recoverK(k, x_values, res, num_points, x_test_values):
    # Get the number of dimensions
    if x_values.size == 0:
        raise ValueError("x_values is empty.")

    # Store the interpolated results in a matrix
    interpolated_results = np.zeros_like(res)
    y = res[:, k]
    interpolated_results[:, 1] = lagrange(x_values, y, num_points, x_test_values)

    return interpolated_results[:, :, 0]

# coefficients
# Generate random matrices A_i0, A_i1, ..., A_i(t-1)
coefficients = coefficients_Ax(s, r, t)
print(coefficients)

# Values of x_i used by N workers
x_values = [1, 2] + list(range(3, N + 1))  # x1, x2, ..., x17
xm = np.array(x_values)

# share and encoding
results = []
for x in x_values:
    result = polynomial_encoding(coefficients, x)
    results.append(result)

res = np.array(results)
x_test_value = 0
interpolated_results = recover(xm, res, t + 1, x_test_value)
print(interpolated_results)

res2 = lagrange(xm, res, t + 1, x_test_value)
print(res2)

# handover
tuple = offline(s, r, t, x_values)
res = handoff(tuple, results)

# Recovery of secret value at 0 point
x_test_value = 0
interpolated_results = recover(xm, res, t + 1, x_test_value)
# Print the interpolated results
print(interpolated_results)


refresh(200, res, xm)

# Recover the secret of U_k, at point 0
k=2
rs = recoverK(2, xm, res, t + 1, x_test_value)
#print(rs)

# Recover the share of C_k at k point
x_test_value = 30
interpolated_results = recover(xm, res, t + 1, x_test_value)
