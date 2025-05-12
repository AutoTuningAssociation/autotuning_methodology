"""Interface for reading and writing input and output formats."""

import json
from pathlib import Path
from autotuning_methodology.validators import validate_T4

def load_T4_format(filepath: Path, validate: True) -> dict:
    """Load and optionally validate a T4 format file."""
    with open(filepath, "r", encoding="utf-8") as fh:
        # get the cache from the .json file
        orig_contents = fh.read()
        try:
            data: dict = json.loads(orig_contents)
        except json.decoder.JSONDecodeError:
            contents = orig_contents[:-1] + "}\n}"
            try:
                data = json.loads(contents)
            except json.decoder.JSONDecodeError:
                contents = orig_contents[:-2] + "}\n}"
                data = json.loads(contents)

        if validate:
            # validate it is in T4 format
            validate_T4(data)

        # return the T4 data
        return data