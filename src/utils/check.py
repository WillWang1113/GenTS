def _condition_shape_check(n_sample, condition):
    if condition.shape[0] == 1:
        condition = condition.repeat(
            n_sample, *[1 for _ in range(len(condition.shape) - 1)]
        )
    elif condition.shape[0] == n_sample:
        pass
    else:
        raise ValueError(
            "The batch size of the given condition should be the same as n_sample or just 1."
        )
    return condition
            
