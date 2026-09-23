"""``python -m unichart ...`` — the same as the ``unichart`` command."""

import sys

from .cli import main

sys.exit(main())
