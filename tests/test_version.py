from methscan import __version__

with open("pyproject.toml", "r") as toml:
    for line in toml:
        if line.startswith("version = "):
            toml_version = line.strip().split()[-1].strip('"')


def test_version():
    assert __version__ == toml_version
