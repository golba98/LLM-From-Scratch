"""Inspect and validate prepared binary token data."""

import argparse
import json
from pathlib import Path
import sys



from llm_data.token_data import inspect_token_data


def main() -> None:
    """Inspect token data from command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Validate binary token data and document indexes."
    )
    parser.add_argument("manifest", type=Path)
    arguments = parser.parse_args()
    try:
        summary = inspect_token_data(arguments.manifest)
    except (OSError, UnicodeError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
