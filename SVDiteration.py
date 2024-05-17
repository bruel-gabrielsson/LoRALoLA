import torch
import torch.nn.functional as F


# this expects lora to be W + BA
def full_lora_svd_wrapper(As,Bs,r, display=True):
    print("[!] fullLoRAPCAgetCombination", As[0].shape, Bs[0].shape, r)
    newAs = [A.t().to(torch.device("cuda")) for A in As]
    Bs = [B.to(torch.device("cuda")) for B in Bs]

    _As, _Bs = Bs, newAs # AB^T

    weights = compute_weights(_As, _Bs)
    U, V = loraSVDIteration(_As, _Bs, weights, r, printstatus=display) # expecting lora to be W + AB^T, A=arg1, B=arg2
    # sum_sigmas = torch.sum(torch.stack(sigmas), dim=0) / len(sigmas)
    return U, V

# normalizes by frobenius squared
def compute_weights(A, B):
    dataset_size = len(A)
    weights = torch.zeros(dataset_size, device=A[0].device)
    for i in range(dataset_size):
        weights[i] = 1.0 / torch.trace((B[i].t() @ B[i]) @ (A[i].t() @ A[i]))
    return weights

def fullSigmaObjective(A, B, w, U, V):
    obj = 0.0
    for i in range(len(A)):
        obj += w[i] * torch.norm((U.t() @ A[i]) @ (B[i].t() @ V), p='fro')**2
    return obj

def loraSVDIteration(A, B, weights, r, tol=0.001, printstatus=True):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    U, _ = torch.linalg.qr(torch.randn(A[0].size(0), r, device=device)) # out_dim x r
    V, _ = torch.linalg.qr(torch.randn(B[0].size(0), r, device=device)) # in_dim x r

    n = len(A)

    if printstatus:
        print('Iteration\tObjective\tU change\tV change')

    for i in range(1000):
        # U step
        stack = torch.zeros((V.size(1) * n, A[0].size(0)), device=device)
        for j in range(n):
            prod = torch.sqrt(weights[j]) * (V.t() @ B[j]) @ A[j].t()
            stack[j * V.size(1):(j + 1) * V.size(1), :] = prod
        oldU = U.clone()
        # print(stack.shape) # torch.Size([r * num_loras, out_dim])
        U = torch.svd_lowrank(stack, q=r+2, niter=100)[2][:, :r]
        #U = torch.svd(stack)[2][:, :r]
        # should be torch.Size([r * num_loras, r])

        # V step
        stack = torch.zeros((U.size(1) * n, B[0].size(0)), device=device)
        for j in range(n):
            prod = torch.sqrt(weights[j]) * (U.t() @ A[j]) @ B[j].t()
            stack[j * U.size(1):(j + 1) * U.size(1), :] = prod
        oldV = V.clone()
        V = torch.svd_lowrank(stack, q=r+2, niter=100)[2][:, :r]
        #V = torch.svd(stack)[2][:, :r]

        # Check convergence
        Uchange = torch.norm(U - oldU @ (oldU.t() @ U), p='fro') / torch.norm(U, p='fro')
        Vchange = torch.norm(V - oldV @ (oldV.t() @ V), p='fro') / torch.norm(V, p='fro')

        if printstatus:
            print(f'Iteration {i+1}: \t{fullSigmaObjective(A, B, weights, U, V)} \t{Uchange.item()} \t{Vchange.item()}')

        if max(Uchange, Vchange) < tol:
            print("Converged")
            return U, V
    return U, V

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    A = [torch.randn(100, 50).to(device) for _ in range(10)]
    B = [torch.randn(80, 50).to(device) for _ in range(10)]
    weights = compute_weights(A, B).to(device) # torch.ones(10).to(device)
    r = 16
    tol = 0.001

    U, V = loraSVDIteration(A, B, weights, r, tol)
