import os
from contextlib import contextmanager
from collections.abc import Iterator
from pathlib import Path

from ibm_watsonx_orchestrate.agent_builder.agents import Agent, AssistantAgent, ExternalAgent
from ibm_watsonx_orchestrate.cli.commands.agents.agents_controller import AgentsController
from ibm_watsonx_orchestrate.cli.commands.environment.environment_controller import activate
from ibm_watsonx_orchestrate.cli.config import (
    Config,
    CONTEXT_SECTION_HEADER,
    CONTEXT_ACTIVE_ENV_OPT,
    ENVIRONMENTS_SECTION_HEADER,
)
from ibm_watsonx_orchestrate.utils.exceptions import BadRequest


AgentSpec = Agent | ExternalAgent | AssistantAgent


class AgentImportError(RuntimeError):
    """Raised when importing an agent spec fails."""


class EnvironmentNotFoundError(RuntimeError):
    """Raised when the specified environment does not exist."""


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


def set_active_environment(
    environment_name: str,
    *,
    apikey: str | None = None,
    username: str | None = None,
    password: str | None = None,
) -> str:
    """Set the active environment for agent imports and other operations.

    This function changes the globally configured active environment and handles authentication,
    similar to running `orchestrate env activate <environment_name>`.

    :param environment_name: Name of the environment to activate (e.g., "local", "dev", "prod").
    :param apikey: API key for authentication (for SaaS/IBM Cloud environments or CPD with API key).
    :param username: Username for authentication (for CPD environments).
    :param password: Password for authentication (for CPD environments).
    :returns: The name of the previously active environment (or None if no environment was active).
    :raises EnvironmentNotFoundError: If the specified environment does not exist in the configuration.
    :raises AgentImportError: If authentication fails or credentials are invalid.
    
    Example:
        >>> # Switch to dev environment with API key
        >>> previous_env = set_active_environment("dev", apikey="your-api-key")
        >>> import_agents_from_file("agent.yaml")
        >>> # Restore previous environment
        >>> if previous_env:
        ...     set_active_environment(previous_env)
        
        >>> # Switch to CPD environment with username/password
        >>> set_active_environment("cpd-prod", username="admin", password="secret")
        
        >>> # Switch to local environment (no credentials needed)
        >>> set_active_environment("local")
    """
    
    cfg = Config()
    
    # Get the current active environment (to return it)
    previous_env = cfg.read(CONTEXT_SECTION_HEADER, CONTEXT_ACTIVE_ENV_OPT)
    
    # Check if the target environment exists
    environments = cfg.get(ENVIRONMENTS_SECTION_HEADER)
    if environments is None or environment_name not in environments:
        available_envs = list(environments.keys()) if environments else []
        raise EnvironmentNotFoundError(
            f"Environment '{environment_name}' not found. "
            f"Available environments: {available_envs}. "
            f"Use 'orchestrate env add' to create a new environment."
        )
    
    # Use the existing activate function which handles authentication
    try:
        activate(
            name=environment_name,
            apikey=apikey,
            username=username,
            password=password,
        )
    except Exception as exc:
        raise AgentImportError(
            f"Failed to activate environment '{environment_name}': {str(exc)}"
        ) from exc
    
    return previous_env


def get_active_environment() -> str | None:
    """Get the name of the currently active environment.

    :returns: The name of the active environment, or None if no environment is active.
    
    Example:
        >>> env = get_active_environment()
        >>> print(f"Current environment: {env}")
    """
    cfg = Config()
    return cfg.read(CONTEXT_SECTION_HEADER, CONTEXT_ACTIVE_ENV_OPT)


def list_environments() -> list[str]:
    """List all configured environments.

    :returns: A list of environment names that are configured in the system.
    
    Example:
        >>> envs = list_environments()
        >>> print(f"Available environments: {envs}")
    """
    cfg = Config()
    environments = cfg.get(ENVIRONMENTS_SECTION_HEADER)
    return list(environments.keys()) if environments else []


def get_environment_by_url(instance_url: str) -> str | None:
    """Find the environment name that matches the given instance URL.

    :param instance_url: The Watsonx Orchestrate instance URL to search for.
    :returns: The environment name if found, None otherwise.
    
    Example:
        >>> env_name = get_environment_by_url("https://orchestrate.ibm.com")
        >>> print(f"Environment: {env_name}")
    """
    from ibm_watsonx_orchestrate.cli.config import ENV_WXO_URL_OPT
    
    cfg = Config()
    environments = cfg.get(ENVIRONMENTS_SECTION_HEADER)
    
    if not environments:
        return None
    
    # Normalize the input URL (remove trailing slash)
    normalized_input = instance_url.rstrip('/')
    
    # Search for matching environment
    for env_name, env_config in environments.items():
        if isinstance(env_config, dict):
            env_url = env_config.get(ENV_WXO_URL_OPT, '')
            # Normalize the stored URL
            normalized_stored = env_url.rstrip('/')
            if normalized_stored == normalized_input:
                return env_name
    
    return None


def import_agents_from_file(
    file: str | Path,
    *,
    app_id: str | None = None,
    instance_url: str | None = None,
    apikey: str | None = None,
    username: str | None = None,
    password: str | None = None,
) -> list[AgentSpec]:
    """Import one or more agents from a spec file into the specified or active environment.

    This is equivalent to running ``orchestrate agents import -f <file>`` (and optionally
    ``--app-id <app_id>``) but is implemented as a direct call into the import logic rather
    than invoking the CLI.
    
    **Note**: If you provide an `instance_url`, the active environment will be changed and will
    NOT be automatically restored. This is optimal for batch operations where you're importing
    multiple agents to the same environment.

    :param file: Path to a ``.yaml``, ``.yml``, ``.json`` or ``.py`` agent definition file.
    :param app_id: App id of the connection to associate with the imported external agent.
    :param instance_url: Optional Watsonx Orchestrate instance URL to import to. The function will
                        find the matching environment from your config and switch to it. If not
                        provided, uses the active environment. The environment will remain active
                        after the import completes.
    :param apikey: API key for authentication when switching environments. Required for non-local
                  environments if valid credentials don't already exist.
    :param username: Username for authentication (CPD environments only).
    :param password: Password for authentication (CPD environments only).
    :returns: The parsed agent spec objects that were published/updated.
    :raises FileNotFoundError: If ``file`` does not exist.
    :raises AgentImportError: If importing, publishing, or authentication fails.
    :raises EnvironmentNotFoundError: If the instance URL doesn't match any configured environment.
    
    Example:
        >>> # Import multiple agents to the same environment (efficient)
        >>> agents1 = import_agents_from_file(
        ...     "agent1.yaml",
        ...     instance_url="https://api.us-south.watson-orchestrate.cloud.ibm.com/instances/abc123",
        ...     apikey="your-api-key"
        ... )
        >>> # Environment stays active, no re-authentication needed
        >>> agents2 = import_agents_from_file("agent2.yaml")
        >>> agents3 = import_agents_from_file("agent3.yaml")
        
        >>> # Import to CPD instance with username/password
        >>> agents = import_agents_from_file(
        ...     "agent.yaml",
        ...     instance_url="https://cpd.example.com/zen/instance-id",
        ...     username="admin",
        ...     password="secret"
        ... )
    """
    file_path: Path = file if isinstance(file, Path) else Path(file)
    if file_path.exists() is False:
        raise FileNotFoundError(str(file_path))

    # Resolve instance_url to environment name if provided
    if instance_url is not None:
        target_environment = get_environment_by_url(instance_url)
        if target_environment is None:
            raise EnvironmentNotFoundError(
                f"No environment found with URL '{instance_url}'. "
                f"Available environments: {list_environments()}. "
                f"Use 'orchestrate env add' to create a new environment with this URL."
            )
        
        # Only switch if we're not already on the target environment
        current_environment = get_active_environment()
        if current_environment != target_environment:
            set_active_environment(
                target_environment,
                apikey=apikey,
                username=username,
                password=password,
            )
    
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
