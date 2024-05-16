import torch
import numpy as np

from peft.utils.save_and_load import set_peft_model_state_dict, get_peft_model_state_dict

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

    # ABt_prods = torch.mean( torch.stack([A[i] @ B[i].t() for i in range(dataset_size)]), dim=0 )
    # U, _, V = torch.svd_lowrank(ABt_prods, q=r+2, niter=2)
    # U, V = U[:,:r], V[:,:r]

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
        stack = []
        for i in range(dataset_size):
            prod = (V.t() @ B[i]) @ A[i].t()
            stack.append(prod)
        stack = torch.cat(stack, dim=0)
        oldU = U

        # batch_size, d_out, d_in = stack.shape 
        # U, S, Vt for each batch item
        # [ (d_out, ) (r, r) (d_in r) ]
        # (32x34 and 4096x16)
        # full svd [ (d_out, d_in) (d_in, d_in) (d_in, d_in).T ] reduced form d_in < d_out
        # lowrank [ (d_out, q) (q) (q d_in).T ]
        U = torch.svd_lowrank(stack.t(), q=r+2, niter=2)[0][:,:r]
        # (U, S, Vh)
        #U = torch.linalg.svd(stack, full_matrices=False)[2].t()[:, :r]

        if display:
            print("U.shape, oldU.shape", U.shape, oldU.shape)
            print('\tU difference: {}'.format(torch.norm(U - oldU, p='fro')))

        # V iteration
        stack = []
        for i in range(dataset_size):
            prod = (U.t() @ A[i]) @ B[i].t()
            stack.append(prod)
        stack = torch.cat(stack, dim=0)
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


# this expects lora to be W + BA
def full_lora_pca_wrapper(As,Bs,r,niter=10, display=True):
    print("[!] fullLoRAPCAgetCombination", As[0].shape, Bs[0].shape, r)
    newAs = [A.t().to(torch.device("cuda")) for A in As]
    Bs = [B.to(torch.device("cuda")) for B in Bs]
    U, V, sigmas = full_lora_pca(Bs, newAs, r, niter, display) # expecting lora to be W + AB^T, A=arg1, B=arg2
    # sum_sigmas = torch.sum(torch.stack(sigmas), dim=0) / len(sigmas)
    return U, V, sigmas

# This expect lora to be W + AB^T
def diagonal_lora_pca(A, B, r, niter=100, display=True):
    m = A[0].shape[0]
    n = B[0].shape[0]
    dataset_size = len(A)

    objectives = torch.zeros(niter)

    # Randomly initialize
    U = torch.randn(m, r)
    V = torch.randn(n, r)
    U, V = U.to(A[0].device), V.to(A[0].device)

    # ABt_prods = torch.mean( torch.stack([A[i] @ B[i].t() for i in range(dataset_size)]), dim=0 )
    # U, _, V = torch.svd_lowrank(ABt_prods, q=r+2, niter=2)
    # U, V = U[:,:r], V[:,:r]
    # U, V = U.to(A[0].device), V.to(A[0].device)

    Sigmas = [torch.diag(torch.rand(r)).to(A[0].device) for _ in range(dataset_size)]

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
        for i in range(dataset_size):
            oldSigma = Sigmas[i]
            Sigmas[i] = torch.diag(torch.linalg.solve(R.t(), torch.linalg.solve(R, torch.sum((U.t() @ A[i]) * (V.t() @ B[i]), dim=1))))
            diff += torch.norm(oldSigma - Sigmas[i])**2
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

    # if display:
    #     plt.plot(objectives.numpy())
    #     plt.title('Objective values, diagonal')
    #     plt.show()

    return U, V, Sigmas

# this expects lora to be W + BA
def diagonal_lora_pca_wrapper(As,Bs,r,niter=100, display=True):
    newAs = [A.t().to(torch.device("cuda")) for A in As]
    Bs = [B.to(torch.device("cuda")) for B in Bs]
    U, V, sigmas = diagonal_lora_pca(Bs, newAs, r, niter, display) # expecting lora to be W + AB^T, A=arg1, B=arg2
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
def diagonal_lora_pca_sparse(A, B, r, niter=100, display=True, sparse_reg = 0):
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

    # if display:
    #     plt.plot(objectives.numpy())
    #     plt.title('Objective values, diagonal')
    #     plt.show()

    return U, V, Sigmas

def set_leaf_module(model, key_to_change, new_weight):
    # Split the key to navigate through the model's hierarchy
    key_parts = key_to_change.split('.')

    # Navigate through the model hierarchy to reach the parameter
    parts = key_parts # param_key.split('.')
    submodule = model
    for part in parts[:-1]: # Exclude the last part ('weight')
        if part.isdigit():
            # Access by index if part is a digit, indicating a list or tuple
            submodule = submodule[int(part)]
        else:
            # Access by attribute or key
            try:
                submodule = getattr(submodule, part)
            except AttributeError:
                # If the attribute access fails, try accessing as a dictionary key, which is relevant for ModuleDict
                submodule = submodule[part]

    # Replace the old weight parameter with the new one
    if hasattr(submodule, 'weight'):
        setattr(submodule, 'weight', torch.nn.Parameter(new_weight))
    else:
        # If the submodule is a ModuleDict and you are targeting a specific module within it
        target_module = submodule['default']  # 'default' is the key in ModuleDict
        target_module.weight = torch.nn.Parameter(new_weight)

# cache is expected to be a dictionary of state_dicts, where the keys are the model_ids
def lola_loras(lora_module_list, cache, r=8, type="diagonal", sparse_reg=0, transform_lora="none"):

    print("[!] lola_loras", "rank", r, "sparse_reg", sparse_reg)

    '''
    'base_model.model.decoder.block.23.layer.1.EncDecAttention.q.lora_A.weight', 'base_model.model.decoder.block.23.layer.1.EncDecAttention.q.lora_B.weight', 
    'base_model.model.decoder.block.23.layer.1.EncDecAttention.v.lora_A.weight', 'base_model.model.decoder.block.23.layer.1.EncDecAttention.v.lora_B.weight'])
    '''

    lola_dict = {}
    keys = cache[lora_module_list[0]].keys() # this is for a single model

    # each key is a lora_A AND lora_B, so we need to group them together.
    # before_lora_dict is a dictionary of lists of tuples of (key, weight)
    before_lora_dict = {}

    for i, peft_model_id in enumerate(lora_module_list): # across models, i is the model number
        lora_state_dict = cache[peft_model_id]
        for key in keys:
            # this makes sure that lora A and B are kept together
            # also makes sure that same lora across models are being merged
            # pre_key maps to same module across models and A B 
            pre_key = key.split("lora_")[0] # 
            if pre_key not in before_lora_dict.keys():
                before_lora_dict[ pre_key ] = [(key, lora_state_dict[key], peft_model_id)]
            else:
                before_lora_dict[ pre_key ].append( (key, lora_state_dict[key], peft_model_id) )

    # We want to iterate through models_ids
    for key in before_lora_dict.keys(): # Will be iterated same order as inserted. Should be same order as models
        As, Bs = [], []
        A_key, B_key = None, None
        norms_A, norms_B = [], []
        
        assert(len(before_lora_dict[key])//2 == len(lora_module_list)) # one per model and lora, vs one per model

        # takes the A and B
        for i in range(len(before_lora_dict[key])): # assume in same order as models
            long_key, weight, peft_model_id = before_lora_dict[key][i]

            if "lora_A" in long_key:
                # This is a hack for random A, B
                #randm = torch.randn_like(weight)
                #weight = randm / torch.norm(randm, p='fro') * torch.norm(weight, p='fro')
                
                if transform_lora == "normalize":
                    norm_factor = torch.norm(weight, p='fro')
                    weight = weight / norm_factor
                elif transform_lora == "none":
                    norm_factor = 1.0
                else:
                    raise ValueError("Invalid transform_lora")
                As.append(weight)
                norms_A.append(norm_factor)
                A_key = long_key
            elif "lora_B" in long_key:
                # This is a hack for random A, B
                #randm = torch.randn_like(weight)
                #weight = randm / torch.norm(randm, p='fro') * torch.norm(weight, p='fro')
                if transform_lora == "normalize":
                    norm_factor = torch.norm(weight, p='fro')
                    weight = weight / norm_factor
                else:
                    norm_factor = 1.0
                Bs.append(weight)
                norms_B.append(norm_factor)
                B_key = long_key
            else:
                # throw error
                assert(False, "lora not in key")


        #print(len(As),len(Bs))
        if type == "diagonal":
            U, V, sigmas = diagonal_lora_pca_sparse_wrapper(As,Bs,r,niter=10, display=False, sparse_reg=sparse_reg)    
        elif type == "full":
            U, V, sigmas = full_lora_pca_wrapper(As,Bs,r,niter=10, display=False) 

            # for i in range(10):
            #     device = torch.device("cuda")
            #     U, V, sigmas = full_lora_pca_wrapper(As,Bs,r,niter=100, display=False) 
            #     reconstruction_error = torch.pow( torch.norm(Bs[i].to(device) @ As[i].to(device) - U @ sigmas[i].to(device) @ V.t(), p='fro') / torch.norm(Bs[i].to(device) @ As[i].to(device), p='fro'), 2)
            #     print("reconstruction_error", reconstruction_error)
            # assert(False)
        elif type == "SVD":
            Us, Vs, Sigmas = [], [], []
            for i in range(len(As)):
                # A = Udiag(S)V.t()
                U, S, V = torch.svd_lowrank(Bs[i].to(torch.device("cuda")) @ As[i].to(torch.device("cuda")), q=r+2, niter=2)
                # U [d_out, r] [r, r] [d_in, r]
                Us.append(U[:,:r])
                Vs.append(V[:,:r])
                Sigmas.append(S[:r])
            
            U, V, sigmas = Us, Vs, Sigmas
        else:
            raise ValueError("Invalid type")

        lola_dict[(A_key, B_key)] = (U, sigmas, V, As, Bs, norms_A, norms_B) # including As and Bs too for reconstruction error, etc
    
    return lola_dict 

# lora_module_list should be exact same list as used to create the lola_dict
# [!] what if model is lora peft model, the uncompressed one?
# if project=True it assumes that the model is the lora model, and will project the lora to the compressed version
def set_lora_from_dict(model, lolas_dict, lora_module_list, return_only_lora, type="diagonal", project=False):
    final_state_dict = {}
    return_only_lora_index = None
    for i, peft_model_id in enumerate(lora_module_list): # across models, i is the model number
        if return_only_lora == peft_model_id:
            return_only_lora_index = i 

    org_state_dict = get_peft_model_state_dict(model) # model.state_dict()
    if return_only_lora_index is None or project:
        return_only_lora_index = None
        print("[!] Obs, we'll project LoRA to compress, assume LoRA model passed")

    # lolas_dict is from the one involved in compression, not targets
    for (A_key, B_key), values in lolas_dict.items():
        U, sigmas, V, As, Bs, norm_A, norm_B = values
        

        if return_only_lora_index is None:
            #raise NotImplementedError("Not implemented")
            #print(org_state_dict) # 'base_model.model.model.layers.31.self_attn.k_proj.lora_B.default.weight'
            # KeyError: 'base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight'
            A, B = org_state_dict[A_key], org_state_dict[B_key] # unnormalized
            A, B = A.to(U.device), B.to(U.device)

            

            # what if U, V aren't orthogonal? Then U.t() @ U != I, V.t() @ V != I. Do I need to do a linear solve?
            # BA = U @ sigma @ V.t()
            # sigma = U.t() @ B @ A @ V
            # X = pinv(U) *(BA) *pinv(V).t() ### K
            # # X = pinv(U) *(BA) *pinv(V.t()) ### K
            if type == "full":
                A_m = V.t() # The V.t() part
                # orthogonal
                sigma = U.t() @ B @ A @ V
                B_m = U @ sigma

                # print(U.t() @ U)
                # print(V.t() @ V)
                # assert(False)

            elif type == "diagonal":
                b = U.t() @ A * V.t() @ torch.ones((V.t().shape[0], 1), device=V.device)
                M = (U.t() @ U) * (V.t() @ V)
                sigma = torch.linalg.solve(M, b)
                
                A_m = V.t() # The V.t() part
                B_m = U @ torch.diag(sigma)

            elif type == "SVD": # could just do svd with the correct rank
                r = len(sigmas[0])
                assert(U[0].shape[1] == r)
                _U, _S, _V = torch.svd_lowrank(B.to(torch.device("cuda")) @ A.to(torch.device("cuda")), q=r+2, niter=2)
                A_m = _V[:,:r].t()
                B_m = _U[:,:r] @ torch.diag(_S[:r])
            else:
                raise ValueError("Invalid type")
            
        else:

            if type=="full" or type=="diagonal":
                A_m = V.t() # The V.t() part
                B_m = U @ sigmas[return_only_lora_index].reshape(sigmas[return_only_lora_index].shape) * norm_A[return_only_lora_index] * norm_B[return_only_lora_index] # The (U @ sigma) part. De normalized
            elif type=="SVD":
                this_U, this_V, sigma = U[return_only_lora_index], V[return_only_lora_index], sigmas[return_only_lora_index] 
                A_m = this_V.t() # The V.t() part
                B_m = this_U @ torch.diag(sigma) * norm_A[return_only_lora_index] * norm_B[return_only_lora_index] # The (U @ sigma) part. De normalized
            else:
                raise ValueError("Invalid type")

        final_state_dict[A_key] = A_m 
        final_state_dict[B_key] = B_m 

        set_leaf_module(model, A_key, A_m) # If we're changing the shape of the weight, we need to set leaf module
        set_leaf_module(model, B_key, B_m) # If we're changing the shape of the weight, we need to set leaf module

    return final_state_dict

# return recon_matrix rows of models, columns of layers
def get_reconstruction_error(lolas_dict, type="full"):
    reconstruction_errors = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # we want to have a list of list (matrix), first list is across models, other layers
    recon_matrix = np.zeros((len(list(lolas_dict.values())[0][1]), len(lolas_dict)))
    j = -1
    for (A_key, B_key), values in lolas_dict.items():
        j += 1
        Us, sigmas, Vs, As, Bs, norm_A, norm_B = values # These A and B are potentaily normalized to 1
        
        for i in range(len(sigmas)):
            if type=="full" or type=="diagonal":
                U, V = Us.to(device), Vs.to(device)
            elif type=="SVD":
                U, V = Us[i].to(device), Vs[i].to(device)
            recon = U @ sigmas[i].to(device) @ V.t() * norm_A[i] * norm_B[i]
            renorm_A = As[i] * norm_A[i]
            renorm_B = Bs[i] * norm_B[i]
            # Since normalized, this should not matter, both the As Bs and the U,V,sigma are normalized. Cancel each other out
            reconstruction_error = torch.pow( torch.norm(renorm_B.to(device) @ renorm_A.to(device) - recon, p='fro') / torch.norm(renorm_B.to(device) @ renorm_A.to(device), p='fro'), 2)
            recon_matrix[i,j] = reconstruction_error.item()
            #print(reconstruction_error)
        #reconstruction_errors.append(reconstruction_error.item())

    #reconstruction_errors = np.array(reconstruction_errors)
    # print mean and std
    #print("Reconstruction error mean: ", np.mean(reconstruction_errors), "std: ", np.std(reconstruction_errors))
    return recon_matrix