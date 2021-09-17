def round_filters(filters, multiplier, divisor, min_depth=None):
    """Round number of filters based on depth multiplier."""

    if multiplier == 1.0:
        return filters

    filters *= multiplier
    min_depth = min_depth or divisor
    new_filters = max(min_depth, int(filters + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_filters < 0.9 * filters:
        new_filters += divisor
    return new_filters
