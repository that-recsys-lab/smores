from pathlib import Path


TESTS_DIR = Path(__file__).resolve().parent
FIXTURES_DIR = TESTS_DIR / "fixtures"
FIXTURE_DATA_DIR = FIXTURES_DIR / "data"
FIXTURE_CONFIG_DIR = FIXTURES_DIR / "config"
TEST_CONFIG_PATH = FIXTURE_CONFIG_DIR / "test_config.yaml"
