import smores


def sample_uninteracted_items(item_ids, probabilities, interacted_items, n_items):
    """Sample n_items using rejection sampling to avoid interacted items.

    Uses rejection sampling: keep sampling until we have enough unique items
    that aren't in the interacted items set.

    Args:
        item_ids: List of all item IDs to sample from
        probabilities: Probability distribution over item_ids (must sum to 1.0)
        interacted_items: Set of item IDs to exclude from sampling
        n_items: Number of items to sample

    Returns:
        List of sampled item IDs (length n_items or less if max_attempts reached)
    """
    sampled_items = []
    max_attempts = len(item_ids) * 10  # Safety limit to prevent infinite loops
    attempts = 0

    while len(sampled_items) < n_items and attempts < max_attempts:
        # Sample one item using the provided probabilities
        item = smores.Smores.state.rand.choice(item_ids, p=probabilities)

        # Accept if not interacted and not already sampled
        if item not in interacted_items and item not in sampled_items:
            sampled_items.append(item)

        attempts += 1

    return sampled_items
