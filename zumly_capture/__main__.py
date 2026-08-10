"""Allow ``python -m zumly_capture`` during development."""

from .cli import main


if __name__ == "__main__":
    raise SystemExit(main())
