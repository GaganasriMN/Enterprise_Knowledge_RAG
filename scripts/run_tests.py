import os
import subprocess
import sys


def main() -> int:
    env = os.environ.copy()
    env.setdefault("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
    return subprocess.call([sys.executable, "-m", "pytest", "-q"], env=env)


if __name__ == "__main__":
    raise SystemExit(main())
