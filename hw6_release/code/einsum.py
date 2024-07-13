import numpy as np

np.random.seed(4444)

A = np.random.rand(5, 5)

trace_np = np.trace(A)
trance_einsum = np.einsum("ii", A)

print("Trace comparison, norm is: ", np.linalg.norm(trace_np - trance_einsum))

B = np.random.rand(5, 5)

mult_np = A.dot(B)
mult_einsum = np.einsum("ij,jk->ik", A, B)
print("Multiplication comparison, norm is: ", np.linalg.norm(mult_np - mult_einsum))


A_batch = np.random.rand(3,4,5)
B_batch = np.random.rand(3,5,6)

mult_batch_np = np.matmul(A_batch,B_batch)
mult_batch_einsum = np.einsum("ijk,ikm->ijm", A_batch, B_batch)

print("Batch Multiplication comparison, norm is: ", np.linalg.norm(mult_batch_np - mult_batch_einsum))





