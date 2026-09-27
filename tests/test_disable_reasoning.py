"""Tests for the --disable-reasoning option.

Covers:
  - _build_model_settings: extra_body injection only for openai-compatible
  - _disable_reasoning_extra_body: merge semantics
  - call_llm: extra_body passed to the OpenAI client when enabled
  - CLI parsing: config set / generate flags and runtime override precedence
  - MCP legacy path: disable_reasoning propagated to BackendConfig
  - Backward compat: old config.json without disable_reasoning defaults to False
"""

from __future__ import annotations

import asyncio
import json
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest
from click.testing import CliRunner

import codewiki.cli.config_manager as cm_mod
from codewiki.cli.commands.config import config_set
from codewiki.cli.commands.generate import generate_command as generate
from codewiki.cli.models.config import Configuration
from codewiki.src.be.llm_services import (
    _build_model_settings,
    _disable_reasoning_extra_body,
    call_llm,
)
from codewiki.src.config import Config


UNSUPPORTED_PROVIDERS = [
    "atlas-cloud",
    "anthropic",
    "bedrock",
    "azure-openai",
    "claude-code",
    "codex",
]


def _make_config(
    *,
    disable_reasoning: bool = False,
    provider: str = "openai-compatible",
) -> Config:
    """Build a minimal backend Config for testing."""
    return Config(
        repo_path="/tmp/repo",
        output_dir="/tmp/out",
        dependency_graph_dir="/tmp/out/dep",
        docs_dir="/tmp/out/docs",
        max_depth=2,
        llm_base_url="http://localhost:1/v1",
        llm_api_key="test-key",
        main_model="test-model",
        cluster_model="test-model",
        fallback_model="test-model",
        provider=provider,
        disable_reasoning=disable_reasoning,
    )


def _setup_config_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point ConfigManager at a temp dir with file-based keyring."""
    monkeypatch.setenv("CODEWIKI_NO_KEYRING", "1")
    monkeypatch.setattr(cm_mod, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(cm_mod, "CONFIG_FILE", tmp_path / "config.json")
    monkeypatch.setattr(cm_mod, "CREDENTIALS_FILE", tmp_path / "credentials.json")


def _make_fake_openai_client() -> tuple[SimpleNamespace, mock.Mock]:
    completions = mock.Mock()
    completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))],
        usage=None,
    )
    client = SimpleNamespace(
        chat=SimpleNamespace(completions=completions),
    )
    return client, completions


def test_build_model_settings_no_extra_body_when_disabled():
    settings = _build_model_settings(
        _make_config(disable_reasoning=False),
        "test-model",
    )

    assert "extra_body" not in settings


def test_build_model_settings_adds_extra_body_for_openai_compatible():
    settings = _build_model_settings(
        _make_config(disable_reasoning=True),
        "test-model",
    )

    assert settings["extra_body"] == {
        "chat_template_kwargs": {"enable_thinking": False}
    }


@pytest.mark.parametrize("provider", UNSUPPORTED_PROVIDERS)
def test_build_model_settings_excludes_unsupported_providers(provider):
    settings = _build_model_settings(
        _make_config(disable_reasoning=True, provider=provider),
        "test-model",
    )

    assert "extra_body" not in settings


def test_disable_reasoning_extra_body_merges_without_mutating_input():
    existing = {
        "top_k": 5,
        "chat_template_kwargs": {
            "some_flag": True,
            "enable_thinking": True,
        },
    }

    result = _disable_reasoning_extra_body(existing)

    assert result == {
        "top_k": 5,
        "chat_template_kwargs": {
            "some_flag": True,
            "enable_thinking": False,
        },
    }
    assert existing["chat_template_kwargs"]["enable_thinking"] is True


def test_disable_reasoning_extra_body_from_none():
    assert _disable_reasoning_extra_body(None) == {
        "chat_template_kwargs": {"enable_thinking": False}
    }


@pytest.mark.parametrize("disable_reasoning", [True, False])
def test_call_llm_extra_body(disable_reasoning):
    config = _make_config(disable_reasoning=disable_reasoning)
    client, completions = _make_fake_openai_client()

    with mock.patch(
        "codewiki.src.be.llm_services.create_openai_client",
        return_value=client,
    ):
        assert call_llm("hello", config) == "ok"

    _, kwargs = completions.create.call_args

    if disable_reasoning:
        assert kwargs["extra_body"] == {
            "chat_template_kwargs": {"enable_thinking": False}
        }
    else:
        assert "extra_body" not in kwargs


def test_configuration_defaults_and_round_trips_disable_reasoning():
    config = Configuration.from_dict(
        {
            "base_url": "https://example.com",
            "main_model": "main",
            "cluster_model": "cluster",
        }
    )

    assert config.disable_reasoning is False

    config.disable_reasoning = True
    serialized = config.to_dict()

    assert serialized["disable_reasoning"] is True
    assert Configuration.from_dict(serialized).disable_reasoning is True


@pytest.mark.parametrize(
    ("flag", "expected"),
    [
        ("--disable-reasoning", True),
        ("--enable-reasoning", False),
    ],
)
def test_config_set_persists_disable_reasoning(
    tmp_path,
    monkeypatch,
    flag,
    expected,
):
    _setup_config_dir(tmp_path, monkeypatch)

    result = CliRunner().invoke(
        config_set,
        [
            "--api-key",
            "sk-test-key-12345",
            "--base-url",
            "http://localhost:1/v1",
            "--main-model",
            "m",
            "--cluster-model",
            "m",
            "--fallback-model",
            "m",
            flag,
        ],
    )

    assert result.exit_code == 0, result.output
    assert json.loads(
        (tmp_path / "config.json").read_text()
    )["disable_reasoning"] is expected


@pytest.mark.parametrize(
    ("persisted", "flag", "expected"),
    [
        (True, "--enable-reasoning", False),
        (False, "--disable-reasoning", True),
    ],
)
def test_generate_runtime_override(
    tmp_path,
    monkeypatch,
    persisted,
    flag,
    expected,
):
    _setup_config_dir(tmp_path, monkeypatch)
    cm_mod.ConfigManager().save(
        api_key="sk-test-key-12345",
        base_url="http://localhost:1/v1",
        main_model="m",
        cluster_model="m",
        fallback_model="m",
        disable_reasoning=persisted,
    )

    captured = {}

    class FakeGenerator:
        def __init__(self, *args, **kwargs):
            captured["config"] = kwargs.get("config", {})

        def generate(self):
            from codewiki.cli.models.job import DocumentationJob

            return DocumentationJob()

    with ExitStack() as stack:
        stack.enter_context(
            mock.patch(
                "codewiki.cli.commands.generate.CLIDocumentationGenerator",
                FakeGenerator,
            )
        )
        stack.enter_context(
            mock.patch(
                "codewiki.cli.commands.generate.is_git_repository",
                return_value=False,
            )
        )
        stack.enter_context(
            mock.patch(
                "codewiki.cli.commands.generate.validate_repository",
                return_value=(Path("/tmp/repo"), []),
            )
        )
        stack.enter_context(
            mock.patch("codewiki.cli.commands.generate.check_writable_output")
        )
        stack.enter_context(
            mock.patch(
                "codewiki.cli.commands.generate.get_git_commit_hash",
                return_value=None,
            )
        )

        result = CliRunner().invoke(
            generate,
            ["--no-artifacts", "--output", str(tmp_path / "out"), flag],
        )

    assert result.exit_code == 0, result.output
    assert captured["config"]["disable_reasoning"] is expected


def test_mcp_legacy_generate_propagates_disable_reasoning(
    tmp_path,
    monkeypatch,
):
    _setup_config_dir(tmp_path, monkeypatch)
    cm_mod.ConfigManager().save(
        api_key="sk-test-key-12345",
        base_url="http://localhost:1/v1",
        main_model="m",
        cluster_model="m",
        fallback_model="m",
        disable_reasoning=True,
    )

    captured = {}
    real_from_cli = Config.from_cli

    @classmethod
    def spy_from_cli(cls, **kwargs):  # noqa: N805 - mimics classmethod
        captured.update(kwargs)
        return real_from_cli(**kwargs)

    monkeypatch.setattr(Config, "from_cli", spy_from_cli)

    repo = tmp_path / "repo"
    repo.mkdir()
    output = tmp_path / "out"
    output.mkdir()

    with mock.patch(
        "codewiki.src.be.documentation_generator.DocumentationGenerator"
    ) as doc_generator:
        doc_generator.return_value.run = mock.AsyncMock(return_value=None)

        from codewiki.mcp.server import _legacy_generate_docs

        asyncio.run(
            _legacy_generate_docs(
                {
                    "repo_path": str(repo),
                    "output_dir": str(output),
                }
            )
        )

    assert captured["disable_reasoning"] is True
