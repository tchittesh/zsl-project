def normalizeFeaturesL2(x):
    '''
    normalize tensor x by its L2 norm in the feature dimension
    expects a tensor x of shape [N, d] as input, where N is the number of
    instances and d is the feature dimension. outputs normalized_x of the same shape
    '''
    x = x + 1e-10
    feature_norm = (x**2).sum(dim=1)**0.5 # l2-norm
    feat = x / feature_norm.unsqueeze(1)

    return feat