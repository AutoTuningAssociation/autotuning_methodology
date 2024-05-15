Contributing
------------
If you're looking to contribute to this package: welcome!
Please check the open `issues <https://github.com/AutoTuningAssociation/autotuning_methodology/issues>`_ and `pull requests <https://github.com/AutoTuningAssociation/autotuning_methodology/pulls>`_ to avoid duplicate work.

Setup
^^^^^
Start out by installing with ``pip install -e .[dev]`` (this installs the package in editable mode alongside the development dependencies). 
During development, unit and integration tests can be ran with ``pytest``. 
`Black <https://pypi.org/project/black/>`_ is used as a formatter, and `Ruff <https://pypi.org/project/ruff/>`_ is used as a linter to check the formatting, import sorting et cetera. 
When using Visual Studio Code, use the ``settings.json`` found in ``.vscode`` to automatically have the correct linting, formatting and sorting during development. 
In addition, install the extensions recommended by us by searching for ``@recommended:workspace`` in the extensions tab for a better development experience. 

Documentation
^^^^^^^^^^^^^
To build the documentation locally, run ``make clean html`` from the ``docs`` folder. Note that the package must have been installed in editable mode with ``pip install -e .``. 
Upon pushing to main or publishing a version, the documentation will be built and published to the GitHub Pages. 
The Docstring format used is Google. 
Type hints are to be included in the function signature and therefor omitted from the docstring. 
In Visual Studio Code, the ``autoDocstring`` extension can be used to automatically infer docstrings. 
When referrring to functions and parameters in the docstring outside of their definition, use double backquotes to be compatible with both MarkDown and ReStructuredText, e.g.: *"skip_draws_check: skips checking that each value in ``draws`` is in the ``dist``."*.

Testing
^^^^^^^
Before contributing a pull request, please run ``nox`` and ensure it has no errors. 
This will test against all Python versions explicitely supported by this package, and will check whether the correct formatting has been applied.
Upon submitting a pull request or pushing to main, these same checks will be ran remotely via GitHub Actions. 