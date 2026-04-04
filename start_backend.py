from pathlib import Path
import sys


ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from quarry.startup import main


if __name__ == "__main__":
    argv = sys.argv[1:]
    interactive_flags = {"-h", "--help", "--skip-corpus"}
    if any(flag in argv for flag in interactive_flags):
        main(argv)
    else:
        answer = input("Run corpus rebuild from data/sources before starting backend? [y/N]: ").strip().lower()
        if answer != "y":
            argv = [*argv, "--skip-corpus"]
        main(argv)
