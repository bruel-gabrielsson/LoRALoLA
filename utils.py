# If you want to change the complete weight tensor, you can use the following function. This is the case when the new shape of the weight tensor is different from the old one.
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