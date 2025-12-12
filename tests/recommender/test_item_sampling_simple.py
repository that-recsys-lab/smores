"""
Simple manual test for item sampling - run this in your development environment
where all dependencies are installed.

Usage:
    python tests/recommender/test_item_sampling_simple.py
"""

def test_imports():
    """Test that all imports work."""
    print("Testing imports...")
    try:
        from smores.samplers.item_sampler import ItemSampler
        print("  ✓ ItemSampler imported")
    except Exception as e:
        print(f"  ✗ ItemSampler import failed: {e}")
        return False

    try:
        from smores.samplers.rejection_sampler import RejectionSampler
        print("  ✓ RejectionSampler imported")
    except Exception as e:
        print(f"  ✗ RejectionSampler import failed: {e}")
        return False

    print("All imports successful!\n")
    return True


def test_rejection_sampler():
    """Test RejectionSampler basic functionality."""
    print("Testing RejectionSampler...")

    from pathlib import Path
    from smores.samplers.rejection_sampler import RejectionSampler
    import smores
    from smores.utils import SmoresConfig
    import yaml

    # Setup smores state (needed for logging and random)
    test_config_path = Path('tests/test_data/test_config.yaml')
    config = SmoresConfig.model_validate(yaml.safe_load(test_config_path.read_text()))
    smores_instance = smores.Smores(config)
    smores_instance.setup()

    # Create sampler
    sampler = RejectionSampler()
    file_path = Path('tests/test_data/item_popularity.csv')

    print(f"  Loading items from {file_path}...")
    sampler.load_from_file(file_path)

    # Check items loaded
    print(f"  ✓ Loaded {len(sampler.item_ids)} items")
    print(f"  Items: {sampler.item_ids}")

    # Check probabilities
    prob_sum = sum(sampler.base_probabilities)
    print(f"  ✓ Probabilities sum to {prob_sum:.6f}")
    assert abs(prob_sum - 1.0) < 0.0001, "Probabilities should sum to 1.0"

    # Test sampling
    print("\n  Testing sampling...")
    sampled = sampler.sample(3, exclude_items=None)
    print(f"  ✓ Sampled {len(sampled)} items: {sampled}")

    # Test sampling with exclusions
    print("\n  Testing sampling with exclusions...")
    exclude = {200, 201}
    sampled_excl = sampler.sample(3, exclude_items=exclude)
    print(f"  ✓ Sampled {len(sampled_excl)} items excluding {exclude}: {sampled_excl}")

    # Verify no excluded items
    for item in sampled_excl:
        assert item not in exclude, f"Item {item} should not be in sampled results"
    print("  ✓ No excluded items in results")

    print("\nRejectionSampler test passed!\n")
    return True


def test_recommender_integration():
    """Test recommender with item sampling."""
    print("Testing Recommender integration...")

    from pathlib import Path
    from smores.utils import SmoresConfig, PythonClassConfig
    from smores.recommender import RecommenderFactory
    from smores.samplers.rejection_sampler import RejectionSampler
    import smores
    import yaml
    import csv

    # Setup smores
    test_config_path = Path('tests/test_data/test_config.yaml')
    config = SmoresConfig.model_validate(yaml.safe_load(test_config_path.read_text()))
    smores_instance = smores.Smores(config)
    smores_instance.setup()

    # Create recommender with item_sampler config
    config_dict = {
        'name': 'TestRec',
        'class_name': 'popular',
        'params': {
            'min_user_count': 1,
            'min_interaction_count': 1,
            'file_name': 'item_popularity.csv',
            'item_sampler': {
                'class_name': 'rejection_sampler',
                'params': {
                    'file_name': 'item_popularity.csv',
                    'sampled_item_count': 2
                }
            }
        }
    }

    rec_config = PythonClassConfig(**config_dict)
    rec = RecommenderFactory.create('popular')
    rec.setup(rec_config)
    rec.setup_dataset()

    print("  ✓ Recommender created")
    print(f"  ✓ item_sampler type: {type(rec.item_sampler).__name__}")
    print(f"  ✓ sampled_item_count: {rec.sampled_item_count}")

    assert rec.item_sampler is not None, "item_sampler should be created"
    assert isinstance(rec.item_sampler, RejectionSampler), "Should be RejectionSampler"
    assert rec.sampled_item_count == 2, "Should be 2 sampled items"

    # Add interactions
    interactions_path = Path('tests/test_data/interactions.csv')
    with open(interactions_path) as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        reader.__next__()  # Skip header
        interactions = []
        for row in reader:
            row_int = [int(entry) for entry in row]
            interactions.append(row_int)

    time_step1 = [row for row in interactions if row[3] == 1]
    time_step2 = [row for row in interactions if row[3] == 2]
    rec.update_dataset(time_step1)
    rec.update_dataset(time_step2)
    print(f"  ✓ Added {len(time_step1) + len(time_step2)} interactions")

    # Train
    rec.train()
    print("  ✓ Recommender trained")

    # Get recommendations
    user_id = 102  # Use user with less interaction history
    smores.Smores.state.slate_size = 5  # Only 5 items exist in test data
    recommendations = rec.get_recommendations(user_id)

    rec_ids = list(recommendations.ids())
    scores = list(recommendations.scores())

    print(f"\n  Recommendations for user {user_id}:")
    print(f"  Total items: {len(rec_ids)}")
    print(f"  Item IDs: {rec_ids}")
    print(f"  Scores: {[f'{s:.2f}' for s in scores]}")

    # Verify - should get some items
    assert len(rec_ids) > 0, f"Should have at least 1 item, got {len(rec_ids)}"
    print(f"  ✓ Got {len(rec_ids)} items")

    # Check that sampled items exist (score 0.0)
    zero_score_items = [rec_ids[i] for i, s in enumerate(scores) if s == 0.0]
    if len(zero_score_items) > 0:
        print(f"  ✓ Found {len(zero_score_items)} sampled items (score=0.0): {zero_score_items}")
    else:
        print(f"  ⚠ No sampled items (might not have enough items left after exclusions)")

    # Check scores
    sorted_scores = sorted(enumerate(scores), key=lambda x: x[1])
    print(f"  Lowest scores: {[(rec_ids[i], f'{s:.2f}') for i, s in sorted_scores[:min(3, len(sorted_scores))]]}")

    print("\nRecommender integration test passed!\n")
    return True


if __name__ == '__main__':
    print("=" * 60)
    print("Item Sampling Manual Test Suite")
    print("=" * 60)
    print()

    try:
        # Test 1: Imports
        if not test_imports():
            print("Import test failed - stopping")
            exit(1)

        # Test 2: RejectionSampler
        if not test_rejection_sampler():
            print("RejectionSampler test failed - stopping")
            exit(1)

        # Test 3: Integration
        if not test_recommender_integration():
            print("Integration test failed - stopping")
            exit(1)

        print("=" * 60)
        print("✓ ALL TESTS PASSED!")
        print("=" * 60)

    except Exception as e:
        print(f"\n✗ TEST FAILED WITH ERROR:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
