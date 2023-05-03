import nox


# @nox.session  # uncomment this line to only run on the current python interpreter
@nox.session(python=["3.8", "3.9", "3.10", "3.11"])  # missing versions can be installed with `pyenv install ...`
# do not forget check / set the versions with `pyenv global`, or `pyenv local` in case of virtual environment
def tests(session: nox.Session) -> None:
    session.install(".[test]")
    session.run("pytest")


@nox.session
def lint(session: nox.Session) -> None:
    session.install("ruff")
    session.run("ruff", "--format=github", "--config=pyproject.toml", ".")
