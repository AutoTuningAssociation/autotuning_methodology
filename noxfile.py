import nox


@nox.session
def tests(session: nox.Session) -> None:
    session.install("-e", ".[test]")
    session.run("pytest")


@nox.session
def lint(session: nox.Session) -> None:
    session.install("ruff")
    session.run("ruff", "--format=github", "--config=pyproject.toml", ".")
