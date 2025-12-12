import os
from contextlib import contextmanager
from collections.abc import Iterator
from pathlib import Path

from ibm_watsonx_orchestrate.agent_builder.agents import Agent, AssistantAgent, ExternalAgent
from ibm_watsonx_orchestrate.cli.commands.agents.agents_controller import AgentsController
from ibm_watsonx_orchestrate.utils.exceptions import BadRequest


AgentSpec = Agent | ExternalAgent | AssistantAgent


class AgentImportError(RuntimeError):
    """Raised when importing an agent spec fails."""


@contextmanager
def _disable_badrequest_sys_exit() -> Iterator[None]:
    """Prevent ``BadRequest`` from calling ``sys.exit``.

    :yields: Nothing.
    """
    had_pytest_marker: bool = "PYTEST_CURRENT_TEST" in os.environ
    previous_value: str | None = os.environ.get("PYTEST_CURRENT_TEST")
    if had_pytest_marker is False:
        os.environ["PYTEST_CURRENT_TEST"] = "1"
    try:
        yield
    finally:
        if had_pytest_marker is False and "PYTEST_CURRENT_TEST" in os.environ:
            del os.environ["PYTEST_CURRENT_TEST"]
        if had_pytest_marker is True and previous_value is not None:
            os.environ["PYTEST_CURRENT_TEST"] = previous_value


def import_agents_from_file(
    file: str | Path,
    *,
    app_id: str | None = None,
) -> list[AgentSpec]:
    """Import one or more agents from a spec file into the active environment.

    This is equivalent to running ``orchestrate agents import -f <file>`` (and optionally
    ``--app-id <app_id>``) but is implemented as a direct call into the import logic rather
    than invoking the CLI.

    :param file: Path to a ``.yaml``, ``.yml``, ``.json`` or ``.py`` agent definition file.
    :param app_id: App id of the connection to associate with the imported external agent.
    :returns: The parsed agent spec objects that were published/updated.
    :raises FileNotFoundError: If ``file`` does not exist.
    :raises AgentImportError: If importing or publishing fails.
    """
    file_path: Path = file if isinstance(file, Path) else Path(file)
    if file_path.exists() is False:
        raise FileNotFoundError(str(file_path))

    controller: AgentsController = AgentsController()
    with _disable_badrequest_sys_exit():
        try:
            agent_specs: list[AgentSpec] = controller.import_agent(
                file=str(file_path),
                app_id=app_id,
            )
            controller.publish_or_update_agents(agent_specs)
            return agent_specs
        except BadRequest as exc:
            raise AgentImportError(str(exc)) from exc
        except SystemExit as exc:
            raise AgentImportError(
                "Agent import failed (underlying code attempted to exit the process)."
            ) from exc
