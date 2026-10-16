"""Interface for reading and writing input and output formats."""

import json
from pathlib import Path
from autotuning_methodology.validators import validate_T4


def get_T4_timeunit(metadata: dict) -> str:
    """Get the time unit the runtime measurements of a T4 file are expressed in.

    Reads ``metadata.session.timeunit``, falling back to the flat ``metadata.timeunit`` key used
    before the metadata reorganization, and to seconds when the file names no unit at all.

    Args:
        metadata: the ``metadata`` object of a T4 document.

    Returns:
        the name of the time unit.
    """
    session_timeunit = metadata.get("session", {}).get("timeunit")
    if session_timeunit is not None:
        return session_timeunit
    return metadata.get("timeunit", "seconds")


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

        # validate the data
        if validate:
            # validate it is in T4 format
            validate_T4(data)
        else:
            # if not validating, we still want to do a basic check of the format
            assert isinstance(data, dict), "T4 format file should be a dictionary."
            assert "metadata" in data, "T4 format file should contain metadata."
            assert "schema_version" in data, "T4 format file should contain schema_version."
            assert "results" in data, "T4 format file should contain results."

        # return the T4 data
        return data
