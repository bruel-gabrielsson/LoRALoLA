import torch
import math
#logger = logging.getLogger("root")

def TSNE(tv_flat_checks):
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    import numpy as np
    # Your data matrix X
    # X = np.array([...])  # Replace this with your data matrix
    X = tv_flat_checks.detach().cpu().numpy()

    # Check and set perplexity
    n_samples = X.shape[0]
    perplexity = min(30, n_samples / 3)  # Adjust perplexity based on your data size

    # Run t-SNE
    tsne = TSNE(n_components=2, random_state=0, perplexity=perplexity)  # 2D t-SNE
    X_2d = tsne.fit_transform(X)


    # Plot the results with color coding and annotations
    plt.figure(figsize=(10, 8))
    colors = plt.cm.rainbow(np.linspace(0, 1, n_samples))  # Creates a color spectrum

    for i, (x, y) in enumerate(X_2d):
        plt.scatter(x, y, color=colors[i])  # Color based on order
        plt.text(x, y, str(i), fontsize=9)  # Annotate the point with its index

    plt.title('t-SNE visualization with Order Coloring')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')

    # Save the figure
    #plt.savefig('tsne_visualization_order.png')  # Saves the figure as a PNG file
    plt.show()  # Show the plot as well

    assert(False)

def arg_special_max(tv_flat_checks):
     # Assuming tv_flat_checks is your tensor of shape (num_samples, weight_dim)
    # Compute the absolute values
    abs_tv_flat_checks = torch.abs(tv_flat_checks)

    # Find the indices of the maximum values along the num_samples dimension
    _, max_indices = torch.max(abs_tv_flat_checks, dim=0)

    # Gather the values from the original tv_flat_checks using these indices
    merged_tensor = torch.gather(tv_flat_checks, 0, max_indices.unsqueeze(0)).squeeze(0)
    return merged_tensor

# default_string = topk20_mass_dis-mean_none
def merge_from_string(merge_function, tv_flat_checks):
    if merge_function == "arg_special_max":
        return arg_special_max(tv_flat_checks)
    if merge_function == "varimax":
        raise ValueError("Varimax is not implemented") # Removed this because did not show promise
        # return factor_analysis(tv_flat_checks)
    if merge_function == "TSNE":
        return TSNE(tv_flat_checks)
    if merge_function == "rotateforsparsity":
        raise ValueError("Varimax is not implemented") # Removed this because did not show promise
        # return sparsity_rotations(tv_flat_checks)


    reset, resolve, merge, lambda_code = merge_function.split("_")
    if "topk" in reset:
        reset_type = "topk"
        reset_thresh = eval(reset[len(reset_type) :])
    elif "std" in reset:
        reset_type = "std"
        reset_thresh = eval(reset[len(reset_type) :])
    elif "nf" in reset:
        reset_type = "nf"
        reset_thresh = eval(reset[len(reset_type) :])
    else:
        reset_type = ""
        reset_thresh = "none"

    merged_tv = merge_methods(
        reset_type,
        tv_flat_checks,
        reset_thresh=reset_thresh,
        resolve_method=resolve,
        merge_func=merge,
    )
    return merged_tv

def merge_methods(
    reset_type, # reset?
    flat_task_checks, # checkpoints? What is the expected shape? I think it's [-1, num_paramters]
    reset_thresh=None, # threshold
    resolve_method=None, # resolve
    merge_func="", # merge
):
    all_checks = flat_task_checks.clone()

    if "nf" in reset_type and reset_thresh != "none":
        #logger.info(f"Pruning: {reset_type} {reset_thresh}")
        updated_checks, *_ = topk_mask_preserve_normfrac(
            all_checks, reset_thresh, return_mask=False
        )
    elif "topk" in reset_type and reset_thresh != "none":
        #logger.info(f"Pruning: {reset_type} {reset_thresh}")
        updated_checks, *_ = topk_values_mask(
            all_checks, K=reset_thresh, return_mask=False
        )
    elif "std" in reset_type and reset_thresh != "none":
        #logger.info(f"Pruning: {reset_type} {reset_thresh}")
        updated_checks, *_ = greater_than_std_mask(
            all_checks, reset_thresh, return_mask=False
        )
    else:
        #logger.info("Not removing NOISE")
        updated_checks = all_checks

    if resolve_method != "none":
        #logger.info(f"RESOLVING SIGN: {resolve_method}")
        final_signs = resolve_sign(updated_checks, resolve_method)
        assert final_signs is not None
    else:
        #logger.info("Not RESOLVING SIGN")
        final_signs = None

    if "dis" in merge_func:
        #logger.info(f"Disjoint AGGREGATION: {merge_func}")
        merged_tv = disjoint_merge(updated_checks, merge_func, final_signs)
    else:
        #logger.info(f"Basic AGGREGATION: {merge_func}")
        merged_tv = aggregate(updated_checks, merge_func, final_signs)

    return merged_tv

def topk_values_mask(M, K=0.7, return_mask=False):
    if K > 1:
        K /= 100

    original_shape = M.shape
    if M.dim() == 1:
        M = M.unsqueeze(0)

    n, d = M.shape
    k = int(d * K)
    k = d - k  # Keep top k elements instead of bottom k elements

    # Find the k-th smallest element by magnitude for each row
    kth_values, _ = M.abs().kthvalue(k, dim=1, keepdim=True)
    # Create a mask tensor with True for the top k elements in each row
    mask = M.abs() >= kth_values
    final_mask = mask.squeeze() if original_shape == M.squeeze().shape else mask

    if return_mask:
        return M * final_mask, final_mask.float().mean(dim=1), final_mask
    return M * final_mask, final_mask.float().mean(dim=1)

def topk_mask_preserve_normfrac(T, normfrac=0.9, return_mask=False):
    row_norms = torch.norm(T, p=2, dim=1, keepdim=True)

    # Calculate the proportion of each element's contribution to its row's norm
    proportion = T.abs() ** 2 / row_norms ** 2

    # Sort the proportions and their indices in descending order
    sorted_proportions, sorted_indices = torch.sort(proportion, dim=1, descending=True)

    # Calculate the cumulative sum of proportions
    cumsum_proportions = torch.cumsum(sorted_proportions, dim=1)

    # Find the indices where cumulative sum >= normfrac
    normfrac_mask = cumsum_proportions >= normfrac
    normfrac_indices = torch.argmax(normfrac_mask.float(), dim=1)

    # Create a range tensor to compare with normfrac_indices
    range_tensor = torch.arange(T.size(1)).unsqueeze(0).expand(T.size(0), -1)

    # Create a mask based on the normfrac_indices
    mask = range_tensor <= normfrac_indices.unsqueeze(1)

    # Initialize final_indices with a value that is out of bounds
    final_indices = torch.full_like(sorted_indices, T.size(1) - 1)

    # Use the mask to get the final indices
    final_indices[mask] = sorted_indices[mask]

    # Initialize the mask with zeros
    M = torch.zeros_like(T, dtype=torch.bool)

    # Use the final indices to update the final mask M
    M.scatter_(1, final_indices, True)

    if return_mask:
        return (T * M), M.float().mean(dim=1), M
    else:
        return (T * M), M.float().mean(dim=1)

def greater_than_std_mask(tensor, factor, return_mask=False):
    mask = (tensor - tensor.mean(dim=1).unsqueeze(1)).abs() > factor * tensor.std(
        dim=1
    ).unsqueeze(1)
    if return_mask:
        return tensor * mask, mask.float().mean(dim=1), mask
    return tensor * mask, mask.float().mean(dim=1)

def resolve_sign(Tensor, resolve_method):
    if resolve_method == "mass":
        sign_to_mult = torch.sign(Tensor.sum(dim=0))
    elif resolve_method == "normfrac":
        sign_to_mult = normfrac_based_sign(Tensor)
    elif resolve_method == "normmass":
        sign_to_mult = normmass_based_sign(Tensor)
    else:
        raise ValueError(f"Sign resolve method {resolve_method} is not defined.")
    sign_to_mult = resolve_zero_signs(sign_to_mult, "majority")
    return sign_to_mult

def normfrac_based_sign(Tensor):
    row_norms = torch.norm(Tensor, dim=1, keepdim=True)
    norm_fracs = (Tensor ** 2) / row_norms ** 2
    return torch.sign(Tensor[norm_fracs.argmax(dim=0), torch.arange(Tensor.shape[1])])

def normmass_based_sign(Tensor):
    row_norms = torch.norm(Tensor, dim=1, keepdim=True)
    norm_fracs = (Tensor ** 2) / row_norms ** 2
    return (Tensor.sign() * norm_fracs.abs()).sum(dim=0).sign()

def resolve_zero_signs(sign_to_mult, method="majority"):
    majority_sign = torch.sign(sign_to_mult.sum())

    if method == "majority":
        sign_to_mult[sign_to_mult == 0] = majority_sign
    elif method == "minority":
        sign_to_mult[sign_to_mult == 0] = -1 * majority_sign
    return sign_to_mult

def disjoint_merge(Tensor, merge_func, sign_to_mult):

    merge_func = merge_func.split("-")[-1]

    # If sign is provided then we select the corresponding entries and aggregate.
    if sign_to_mult is not None:
        rows_to_keep = torch.where(
            sign_to_mult.unsqueeze(0) > 0, Tensor > 0, Tensor < 0
        )
        selected_entries = Tensor * rows_to_keep
    # Else we select all non-zero entries and aggregate.
    else:
        rows_to_keep = Tensor != 0
        selected_entries = Tensor * rows_to_keep

    if merge_func == "mean":
        non_zero_counts = (selected_entries != 0).sum(dim=0).float()
        disjoint_aggs = torch.sum(selected_entries, dim=0) / torch.clamp(
            non_zero_counts, min=1
        )
    elif merge_func == "sum":
        disjoint_aggs = torch.sum(selected_entries, dim=0)
    elif merge_func == "max":
        disjoint_aggs = selected_entries.abs().max(dim=0)[0]
        disjoint_aggs *= sign_to_mult
    else:
        raise ValueError(f"Merge method {merge_func} is not defined.")

    return disjoint_aggs

def aggregate(T, agg_type, final_signs, dim=0):
    if agg_type == "mean":
        result = torch.mean(T, dim=dim)
    elif agg_type == "sum":
        result = torch.sum(T, dim=dim)
    elif agg_type == "median":
        result = torch.median(T, dim=dim)[0]
    elif agg_type == "magnitude":
        max_indices = T.abs().argmax(dim=0)
        result = T[max_indices, torch.arange(T.shape[1])]
    else:
        raise ValueError("Invalid agg_type: %s" % agg_type)

    if final_signs is not None:
        # print(final_signs)
        result = result.abs() * final_signs

    return result