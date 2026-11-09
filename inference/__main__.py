"""`python -m inference` - works straight from a checkout, nothing installed."""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
