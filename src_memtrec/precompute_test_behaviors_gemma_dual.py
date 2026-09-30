#!/usr/bin/env python3
"""Compatibility entry point; use precompute_memory.py test for new commands."""
if __package__:
    from .precompute_memory import test_main as main
else:
    from precompute_memory import test_main as main


if __name__ == "__main__":
    main()
