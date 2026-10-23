import pytest

from he_snowflake.keys import load_all


@pytest.fixture(scope="session")
def keys(tmp_path_factory):
    """Fresh keys for the whole test session (packed keys sized for up to 128 values)."""
    return load_all(tmp_path_factory.mktemp("keys"), max_packed_values=128)
