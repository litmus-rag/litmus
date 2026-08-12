import pytest

from litmus.version import bump_version


@pytest.mark.parametrize(
    "version,level,expected",
    [
        ("1.0.0", "patch", "1.0.1"),
        ("1.0.0", "minor", "1.1.0"),
        ("1.0.0", "major", "2.0.0"),
        ("1.2.3", "patch", "1.2.4"),
        ("1.2.3", "minor", "1.3.0"),
        ("1.2.3", "major", "2.0.0"),
    ],
)
def test_bump_version(version, level, expected):
    assert bump_version(version, level) == expected


def test_bump_version_unknown_level_raises():
    with pytest.raises(ValueError):
        bump_version("1.0.0", "not_a_level")


def test_bump_version_malformed_input_defaults():
    # Malformed version strings fall back to 1.0.0 base rather than crashing.
    assert bump_version("not.a.version", "patch") == "1.0.1"
