import torch
import numpy as np

from peft.utils.save_and_load import set_peft_model_state_dict, get_peft_model_state_dict
from .compression import diagonal_lora_pca_sparse_wrapper, full_lora_pca_wrapper
from .utils import set_leaf_module
from .merging import merge_from_string

# cache is expected to be a dictionary of state_dicts, where the keys are the model_ids
def lola_loras(lora_module_list, cache, r=8, type="diagonal", sparse_reg=0, transform_lora="none"):

    print("[!] lola_loras", "rank", r, "sparse_reg", sparse_reg)

    if type=="full":
        assert(transform_lora=="none", "transform_lora should be none for full")

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
                # randm = torch.randn_like(weight)
                # weight = randm / torch.norm(randm, p='fro') * torch.norm(weight, p='fro')
                
                if transform_lora == "normalize":
                    norm_factor = torch.norm(weight, p='fro')
                    weight = weight / norm_factor
                elif transform_lora == "normalize_BA":
                    norm_factor = 1.0
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

        if transform_lora == "normalize_BA": # hack for now
            for lora_index in range(len(As)):
                A, B = As[lora_index], Bs[lora_index]
                norm_factor = torch.sqrt( torch.norm(B @ A, p='fro') ) # np.linalg.norm(a @ b.T)
                A = A / norm_factor
                B = B / norm_factor
                As[lora_index], Bs[lora_index] = A, B
                norms_A[lora_index], norms_B[lora_index] = norm_factor.item(), norm_factor.item()

        if type == "diagonal":
            U, V, sigmas = diagonal_lora_pca_sparse_wrapper(As,Bs,r, display=False, sparse_reg=sparse_reg)    
        elif type == "full":
            U, V, sigmas = full_lora_pca_wrapper(As,Bs,r,niter=10, display=False)
        elif type == "SVD":
            Us, Vs, Sigmas = [], [], []
            for i in range(len(As)):
                # A = Udiag(S)V.t()
                U, S, V = torch.svd_lowrank(Bs[i].to(torch.device("cuda")) @ As[i].to(torch.device("cuda")), q=r+2, niter=2)
                # U [d_out, r] [r, r] [d_in, r]
                Us.append(U[:,:r])
                Vs.append(V[:,:r])
                Sigmas.append(torch.diag(S[:r]))
            
            U, V, sigmas = Us, Vs, Sigmas
        elif type == "TIES":
            """
            Use merge_from_string("string_command", torch.stack({list of models}).reshape(len(sigmas), xx) )
            """

            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

            # First, merge all the As
            merge_string_input = "topk20_mass_dis-mean_none"

            prods = [(B.to(device) @ A.to(device)).to(torch.device("cpu")) for A, B in zip(As, Bs)] # cannot run on GPU, out of memory for 10

            combination_BA = merge_from_string(merge_string_input, 
                                               torch.stack(prods).reshape(len(prods), prods[0].shape[0] * prods[0].shape[1]) 
                                               )
            combination_BA_reshaped = combination_BA.reshape(prods[0].shape)

            # Might use U @ sigma @ V.t() later
            # 4096 x 1024
            V = torch.eye(combination_BA_reshaped.shape[1]).to(combination_BA_reshaped.device).t() # combination_A_reshaped.t() # single
            U = combination_BA_reshaped # combination_B_reshaped # single
            sigmas = torch.eye(combination_BA_reshaped.shape[1]).to(V.device) # [torch.eye(r) for _ in range(len(As))]
        elif type == "none":
            U, V, sigmas = None, None, None
        else:
            raise ValueError("Invalid type")

        lola_dict[(A_key, B_key)] = (U, sigmas, V, As, Bs, norms_A, norms_B) # including As and Bs too for reconstruction error, etc
    
    return lola_dict 

def project_from_AB_UV(A, B, U, V, type="diagonal"):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    A, B = A.to(device), B.to(device)

    if type == "diagonal":
        U, V = U.to(device), V.to(device)
        b = U.t() @ B @ A * V.t() @ torch.ones((V.t().shape[1], 1), device=V.device)
        M = (U.t() @ U) * (V.t() @ V)
        sigma = torch.linalg.solve(M, b)
        sigma = torch.diag(sigma.reshape(-1)).to(A.device)
        recon = U @ sigma @ V.t()
    elif type == "full":
        U, V = U.to(device), V.to(device)
        sigma = U.t() @ B @ A @ V
        recon = U @ sigma @ V.t()
    elif type == "SVD": # U, V are lists
        if isinstance(U, list):
            U, V = U[0].to(device), V[0].to(device)
        else:
            U, V = U.to(device), V.to(device)
        r = U.shape[1]
        assert(U.shape[1] == V.shape[1] == r)
        _U, _S, _V = torch.svd_lowrank(B @ A, q=r+2, niter=2)
        sigma = torch.diag(_S[:r])
        U, V = _U[:,:r], _V[:,:r]
        recon = U @ sigma @ V.t()
    elif type == "TIES":
        U, V = U.to(device), V.to(device)
        recon = U @ V.t() # sigma isn't needed, it was idenity
        sigma = torch.eye(U.shape[1]).to(U.device)
    else:
        raise ValueError("Invalid type")
    

    return sigma, U, V, recon

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
            # These are unnormalized A, B, straight from the model
            A, B = org_state_dict[A_key], org_state_dict[B_key] # unnormalized

            sigma, U, V, recon = project_from_AB_UV(A, B, U, V, type=type)

            A_m = V.t() 
            B_m = U @ sigma
            
        else:

            if type=="full" or type=="diagonal":
                A_m = V.t()
                B_m = U @ sigmas[return_only_lora_index].reshape(sigmas[return_only_lora_index].shape) * norm_A[return_only_lora_index] * norm_B[return_only_lora_index] # The (U @ sigma) part. De normalized
            elif type=="SVD":
                this_U, this_V, sigma = U[return_only_lora_index], V[return_only_lora_index], sigmas[return_only_lora_index] 
                A_m = this_V.t()
                B_m = this_U @ sigma * norm_A[return_only_lora_index] * norm_B[return_only_lora_index] # The (U @ sigma) part. De-normalized
            elif type=="TIES":
                A_m = V.t()
                B_m = U @ sigmas * norm_A[return_only_lora_index] * norm_B[return_only_lora_index] # wouldn't recommend normalzing
            else:
                raise ValueError("Invalid type")

        final_state_dict[A_key] = A_m 
        final_state_dict[B_key] = B_m 

        set_leaf_module(model, A_key, A_m) # If we're changing the shape of the weight, we need to set leaf module
        set_leaf_module(model, B_key, B_m) # If we're changing the shape of the weight, we need to set leaf module

    return final_state_dict

def reconstruction_error(A,B,recon,device=torch.device('cpu')):
    return torch.pow( torch.norm(B.to(device) @ A.to(device) - recon, p='fro') / torch.norm(B.to(device) @ A.to(device), p='fro'), 2).item()

# return recon_matrix rows of models, columns of layers
def get_reconstruction_error(lolas_dict, type="full", project=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # we want to have a list of list (matrix), first list is across models, other layers
    recon_matrix = np.zeros((len(list(lolas_dict.values())[0][4]), len(lolas_dict))) # should be As
    j = -1
    for (A_key, B_key), values in lolas_dict.items():
        j += 1
        Us, sigmas, Vs, As, Bs, norm_A, norm_B = values # These A and B are potentaily normalized to 1
        
        for i in range(len(As)):


            if type=="full" or type=="diagonal" or type=="TIES":
                U, V = Us.to(device), Vs.to(device)
            elif type=="SVD":
                U, V = Us[i].to(device), Vs[i].to(device) # torch.Size([4096, 16]) torch.Size([4096, 16])

            # PROJECTION
            if type=="diagonal" and (not project):
                sigma = sigmas[i].to(device)
                recon = U @ sigma @ V.t()
            else:
                sigma, U, V, recon = project_from_AB_UV(As[i], Bs[i], U, V, type=type)
            
            recon = recon * norm_A[i] * norm_B[i]

            renorm_A = As[i] * norm_A[i]
            renorm_B = Bs[i] * norm_B[i]
            # Since normalized, this should not matter, both the As Bs and the U,V,sigma are normalized. Cancel each other out
            reconstruction_error = torch.pow( torch.norm(renorm_B.to(device) @ renorm_A.to(device) - recon, p='fro') / torch.norm(renorm_B.to(device) @ renorm_A.to(device), p='fro'), 2)
            recon_matrix[i,j] = reconstruction_error.item()

    return recon_matrix