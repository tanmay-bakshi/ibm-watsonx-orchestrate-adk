from pathlib import Path
from unittest.mock import patch

import pytest

from ibm_watsonx_orchestrate.agents.importer import AgentImportError, import_agents_from_file


class TestImportAgentsFromFile:
    """Tests for :func:`ibm_watsonx_orchestrate.agents.importer.import_agents_from_file`."""

    def test_imports_and_publishes(self, tmp_path: Path) -> None:
        """Calls the underlying import and publish logic.

        :param tmp_path: Temporary directory path.
        """
        agent_file: Path = tmp_path / "agent.yaml"
        agent_file.write_text("name: test\nkind: native\n", encoding="utf-8")

        with patch(
            "ibm_watsonx_orchestrate.cli.commands.agents.agents_controller.AgentsController.import_agent"
        ) as import_mock, patch(
            "ibm_watsonx_orchestrate.cli.commands.agents.agents_controller.AgentsController.publish_or_update_agents"
        ) as publish_mock:
            import_mock.return_value = []

            result = import_agents_from_file(agent_file, app_id=None)

            import_mock.assert_called_once_with(file=str(agent_file), app_id=None)
            publish_mock.assert_called_once_with([])
            assert result == []

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        """Raises ``FileNotFoundError`` for missing files.

        :param tmp_path: Temporary directory path.
        """
        missing_file: Path = tmp_path / "missing.yaml"

        with pytest.raises(FileNotFoundError):
            import_agents_from_file(missing_file, app_id=None)

    def test_system_exit_is_wrapped(self, tmp_path: Path) -> None:
        """Wraps ``SystemExit`` into ``AgentImportError`` for library safety.

        :param tmp_path: Temporary directory path.
        """
        agent_file: Path = tmp_path / "agent.yaml"
        agent_file.write_text("name: test\nkind: native\n", encoding="utf-8")

        with patch(
            "ibm_watsonx_orchestrate.cli.commands.agents.agents_controller.AgentsController.import_agent",
            side_effect=SystemExit(1),
        ):
            with pytest.raises(AgentImportError):
                import_agents_from_file(agent_file, app_id=None)
