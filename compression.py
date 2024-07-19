import torch
import numpy as np

# this expects lora to be W + BA
def full_lora_pca_wrapper(As,Bs,r,niter=10, display=True):
    print("[!] fullLoRAPCAgetCombination", As[0].shape, Bs[0].shape, r)
    newAs = [A.t().to(torch.device("cuda")) for A in As]
    Bs = [B.to(torch.device("cuda")) for B in Bs]
    U, V, sigmas = full_lora_pca(Bs, newAs, r, niter, display) # expecting lora to be W + AB^T, A=arg1, B=arg2
    return U, V, sigmas

# this expects lora to be W + BA
def diagonal_lora_pca_sparse_wrapper(As,Bs,r,niter=100, display=True, sparse_reg=0):
    print("[!] SPARSE diagonal_lorapca_getCombination_sparse", sparse_reg, As[0].shape, Bs[0].shape, r)
    newAs = [A.t() for A in As]
    newAs = [A.t().to(torch.device("cuda")) for A in As]
    Bs = [B.to(torch.device("cuda")) for B in Bs]
    U, V, sigmas = diagonal_lora_pca_sparse(Bs, newAs, r, niter, display, sparse_reg=sparse_reg) # expecting lora to be W + AB^T, A=arg1, B=arg2
    return U, V, sigmas

# This expect lora to be W + AB^T
def full_lora_pca(A, B, r, niter=10, display=True):
    m = A[0].shape[0]
    n = B[0].shape[0]
    if display:
        print('Full LoRAPCA: m = {}, n = {}, r = {}, niter = {}'.format(m, n, r, niter))
        print('Dataset size: {}'.format(len(A)))
        print('A[0] shape: {}'.format(A[0].shape))
        print('B[0] shape: {}'.format(B[0].shape))

    dataset_size = len(A)

    # Random orthogonal initializers
    U, _ = torch.linalg.qr(torch.randn(m, r))
    V, _ = torch.linalg.qr(torch.randn(n, r))

    U, V = U.to(A[0].device), V.to(A[0].device)

    objectives = np.zeros(niter)
    objective = 0

    for iter in range(niter):
        if display:
            print('Iteration {}:'.format(iter + 1))

        if display:
            oldobjective = objective
            objective = 0
            for i in range(dataset_size):
                diff = A[i] @ B[i].t() - U @ U.t() @ A[i] @ B[i].t() @ V @ V.t()
                objective += torch.norm(diff, p='fro')**2
            print('\tObjective: {} (diff = {})'.format(objective, oldobjective - objective))
            objectives[iter] = objective

        # U iteration
        stack = torch.zeros((V.size(1) * dataset_size, A[0].size(0)), device=A[0].device)
        for j in range(dataset_size):
            prod = (V.t() @ B[j]) @ A[j].t()
            stack[j * V.size(1):(j + 1) * V.size(1), :] = prod

        oldU = U

        U = torch.svd_lowrank(stack.t(), q=r+2, niter=2)[0][:,:r]

        if display:
            print("U.shape, oldU.shape", U.shape, oldU.shape)
            print('\tU difference: {}'.format(torch.norm(U - oldU, p='fro')))

        # V iteration
        stack = torch.zeros((U.size(1) * dataset_size, B[0].size(0)), device=A[0].device)
        for j in range(dataset_size):
            prod = (U.t() @ A[j]) @ B[j].t()
            stack[j * U.size(1):(j + 1) * U.size(1), :] = prod

        oldV = V
        V = torch.svd_lowrank(stack.t(), q=r+2, niter=2)[0][:,:r]
        #V = torch.linalg.svd(stack, full_matrices=False)[2].t()[:, :r]

        if display:
            print('\tV difference: {}'.format(torch.norm(V - oldV, p='fro')))

    sigmas = []
    for i in range(dataset_size):
        sigma = U.t() @ A[i] @ B[i].t() @ V
        sigmas.append(sigma)

    return U, V, sigmas

# This expect lora to be W + AB^T
def diagonal_lora_pca_sparse(A, B, r, niter=100, display=True, sparse_reg = 0, tol=0.001):
    m = A[0].shape[0]
    n = B[0].shape[0]
    dataset_size = len(A)

    objectives = torch.zeros(niter)

    # Randomly initialize
    U = torch.randn(m, r)
    V = torch.randn(n, r)
    U, V = U.to(A[0].device), V.to(A[0].device)
    Sigmas = [torch.diag(torch.rand(r)).to(A[0].device) for _ in range(dataset_size)]
    oldSigmas = Sigmas
    
    objective = 0.0

    for iter in range(niter):
        if display:
            print(f'Iteration {iter + 1}:')

        if display:
            old_objective = objective
            objective = 0.0
            for i in range(dataset_size):
                objective += torch.norm(A[i] @ B[i].t() - U @ Sigmas[i] @ V.t())**2
            print(f'\tObjective: {objective} (diff = {old_objective - objective})')

            objectives[iter] = objective

        # Sigma optimization
        mtx = (U.t() @ U) * (V.t() @ V)
        R = torch.linalg.cholesky(mtx)
        diff = 0.0
        
        if sparse_reg <= 0: # NO SPARSE REG
            for i in range(dataset_size):
                oldSigma = Sigmas[i]
                Sigmas[i] = torch.diag(torch.linalg.solve(R.t(), torch.linalg.solve(R, torch.sum((U.t() @ A[i]) * (V.t() @ B[i]), dim=1))))
                diff += torch.norm(oldSigma - Sigmas[i])**2

        else: # SPARSE REG
            maxs = 0 * torch.ones(r).to(A[0].device)
            sigms = []
            for i in range(dataset_size):
                oldSigmas[i] = Sigmas[i]
                sigms.append( torch.linalg.solve(R.t(), torch.linalg.solve(R, torch.sum((U.t() @ A[i]) * (V.t() @ B[i]), dim=1))))
                maxs = torch.maximum(maxs,torch.abs(sigms[i]))
        
            for i in range(dataset_size):
                #Shrink
                shrink = min(sparse_reg, 0.25 * torch.min(maxs))
                abs_sigms = torch.maximum(torch.zeros_like(sigms[i], device=A[0].device),torch.abs(sigms[i]) - shrink)
                sigms_thrsh = abs_sigms * torch.sign(sigms[i])
                Sigmas[i] = torch.diag(sigms_thrsh)
                diff += torch.norm(oldSigmas[i] - Sigmas[i])**2

        diff = torch.sqrt(diff)
        if display:
            print(f'\tSigma difference: {diff}')

        # U optimization
        oldU = U.clone()  # Copy of U before updating
        lhs = torch.zeros(r, r).to(A[0].device)
        rhs = torch.zeros(m, r).to(A[0].device)
        for i in range(dataset_size):
            Vs = V @ Sigmas[i].t()
            lhs += Vs.t() @ Vs
            rhs += A[i] @ (B[i].t() @ (V @ Sigmas[i].t()))
        U = torch.linalg.solve(lhs.t(), rhs.t()).t()

        if display:
            print(f'\tU difference: {torch.norm(U - oldU)}')

        # V optimization
        oldV = V.clone()  # Copy of V before updating
        lhs = torch.zeros(r, r).to(A[0].device)
        rhs = torch.zeros(n, r).to(A[0].device)
        for i in range(dataset_size):
            Us = U @ Sigmas[i]
            lhs += Us.t() @ Us
            rhs += B[i] @ (A[i].t() @ (U @ Sigmas[i]))
        V = torch.linalg.solve(lhs.t(), rhs.t()).t()

        if display:
            print(f'\tV difference: {torch.norm(V - oldV)}')

        # Rescale
        sigma_norm = sum([torch.norm(Sigma)**2 for Sigma in Sigmas])**0.5
        for i in range(dataset_size):
            Sigmas[i] /= sigma_norm
        U *= sigma_norm

        c = (torch.norm(V) / torch.norm(U))**0.5
        V /= c
        U *= c

        # Check convergence
        # Uchange = torch.norm(U - oldU @ (oldU.t() @ U), p='fro') / torch.norm(U, p='fro')
        # Vchange = torch.norm(V - oldV @ (oldV.t() @ V), p='fro') / torch.norm(V, p='fro')

        # if max(Uchange, Vchange) < tol:
        #     print("Converged")
        #     return U, V, Sigmas

    return U, V, Sigmas