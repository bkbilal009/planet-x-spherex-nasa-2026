"""SPHEREx change detection - local command-line entry point.

    python run_analysis.py --help
    python run_analysis.py --target r-dra --band D3 --match
"""
import sys

from image_analysis.cli import main

if __name__ == "__main__":
    sys.exit(main())
