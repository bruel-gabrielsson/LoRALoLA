import torch

from .SVDiteration import compute_weights, fullSigmaObjective

# this expects lora to be W + BA
def full_lora_eigen_wrapper(As,Bs,r, display=True):
    print("[!] full lora eigen", As[0].shape, Bs[0].shape, r)
    newAs = [A.t().to(torch.device("cuda")) for A in As]
    Bs = [B.to(torch.device("cuda")) for B in Bs]

    _As, _Bs = Bs, newAs # AB^T

    weights = compute_weights(_As, _Bs)
    U, V = loraEigenvalueIteration(_As, _Bs, weights, r, printstatus=display) # expecting lora to be W + AB^T, A=arg1, B=arg2
    # sum_sigmas = torch.sum(torch.stack(sigmas), dim=0) / len(sigmas)
    return U, V

def loraEigenvalueIteration(A, B, weights, r, tol=0.001, printstatus=True):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    n = len(A)
    
    U, _ = torch.linalg.qr(torch.randn(A[0].size(0), r, device=device)) # out_dim x r
    V, _ = torch.linalg.qr(torch.randn(B[0].size(0), r, device=device)) # in_dim x r

    if printstatus:
        print('Iteration\tObjective\tU change\tV change')

    for i in range(1000):
        Uprod = torch.zeros_like(U)
        Vprod = torch.zeros_like(V)

        for j in range(n):
            BV = B[j].t() @ V
            AU = A[j].t() @ U
            P = BV.t() @ AU
            Uprod += weights[j] * A[j] @ (BV @ P)
            Vprod += weights[j] * B[j] @ (AU @ P.t())

        oldU = U.clone()
        oldV = V.clone()

        U = torch.linalg.qr(Uprod)[0]
        V = torch.linalg.qr(Vprod)[0]

        Uchange = torch.norm(U - oldU @ (oldU.t() @ U), p='fro') / torch.norm(U, p='fro')
        Vchange = torch.norm(V - oldV @ (oldV.t() @ V), p='fro') / torch.norm(V, p='fro')

        if printstatus:
            print(f'Iteration {i+1}: \t{fullSigmaObjective(A, B, weights, U, V)} \t{Uchange.item()} \t{Vchange.item()}')

        if max(Uchange, Vchange) < tol:
            print(f'Converged after {i+1} iterations')
            return U, V

    return U, V


def recon_errors(As, Bs, U, V):
    errors = []
    for i in range(len(As)):
        A, B = As[i], Bs[i]
        sigma = U.t() @ A @ B.t() @ V
        error = torch.pow(torch.norm(U @ sigma @ V.t() - A @ B.t(), p='fro') / torch.norm(A @ B.t(), p='fro'), 2).item()
        errors.append(error)

    return errors

# Example usage
if __name__ == "__main__":
    from SVDiteration import compute_weights, fullSigmaObjective
    from SVDiteration import loraSVDIteration
    import numpy as np

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # use pickle to load /Users/rbg/Downloads/tensors_python.pkl
    import pickle
    with open('/Users/rbg/Downloads/tensors_python.pkl', 'rb') as f:
        data = pickle.load(f)
    A = [torch.tensor(m).to(device) for m in data['Bs']]
    B = [torch.tensor(m).to(device).t() for m in data['As']]

    # A = [torch.randn(100, 50).to(device) for _ in range(10)]
    # B = [torch.randn(80, 50).to(device) for _ in range(10)]
    weights = compute_weights(A, B).to(device)
    r = 4
    tol = 0.001

    U, V = loraEigenvalueIteration(A, B, weights, r, tol)

    U1, V1 = loraSVDIteration(A, B, weights, r, tol)

    print(torch.norm(U - U1, p='fro'), torch.norm(V - V1, p='fro'))

    rec = np.array(recon_errors(A, B, U, V))
    print("eigen error", rec.mean(), rec.std())

    rec = np.array(recon_errors(A, B, U1, V1))
    print("svd error", rec.mean(), rec.std())