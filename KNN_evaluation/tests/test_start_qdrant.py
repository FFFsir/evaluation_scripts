"""Tests for webui._start_qdrant idempotent three-state behavior (Task 3.5).

与 cli._start_qdrant（Task 3.3，ledger Minor #5）保持一致的幂等三态：
`docker ps -a` 全量列出（含停止容器）→ 运行中复用 / 存在但停止则 docker start /
不存在则 docker run（-v qdrant_data:/qdrant/storage）。任何失败返回 False。
"""
import subprocess
from types import SimpleNamespace

from KNN_evaluation import webui


def _make_fake_run(docker_ps_a_out: str = "", docker_ps_out: str = "",
                   rc: dict | None = None):
    """构造 fake subprocess.run：按命令分派 docker ps -a / ps / start / run.

    docker ps -a 返回含停止容器的全量列表；docker ps 返回仅运行中容器。
    rc 用 {"start": rc, "run": rc} 指定 start/run 的 returncode（默认 0）。
    """
    rc = rc or {}
    calls: list[list[str]] = []

    def fake_run(cmd, **kw):
        calls.append(cmd)
        if cmd[1] == "ps":
            stdout = docker_ps_a_out if "-a" in cmd else docker_ps_out
            return SimpleNamespace(stdout=stdout, returncode=0)
        if cmd[1] == "start":
            return SimpleNamespace(stdout="", returncode=rc.get("start", 0))
        if cmd[1] == "run":
            return SimpleNamespace(stdout="", returncode=rc.get("run", 0))
        return SimpleNamespace(stdout="", returncode=0)

    return fake_run, calls


def _start(monkeypatch, fake_run):
    monkeypatch.setattr(subprocess, "run", fake_run)
    return webui._start_qdrant()


def test_running_container_reused(monkeypatch):
    """三态-运行中：docker ps -a 与 docker ps 都含 qdrant → 复用，无多余命令."""
    fake_run, calls = _make_fake_run(
        docker_ps_a_out="qdrant\n", docker_ps_out="qdrant\n",
    )
    assert _start(monkeypatch, fake_run) is True
    assert calls == [
        ["docker", "ps", "-a", "--format", "{{.Names}}"],
        ["docker", "ps", "--format", "{{.Names}}"],
    ]


def test_running_container_reused_with_other_containers(monkeypatch):
    """三态-运行中（含其他容器）：docker ps 输出含 qdrant + 其他名称 → 仍复用."""
    fake_run, calls = _make_fake_run(
        docker_ps_a_out="qdrant\nsome-other-container\n",
        docker_ps_out="qdrant\nsome-other-container\n",
    )
    assert _start(monkeypatch, fake_run) is True
    assert calls == [
        ["docker", "ps", "-a", "--format", "{{.Names}}"],
        ["docker", "ps", "--format", "{{.Names}}"],
    ]


def test_stopped_container_started(monkeypatch):
    """三态-停止：docker ps -a 含 qdrant 但运行列表不含 → docker start."""
    fake_run, calls = _make_fake_run(
        docker_ps_a_out="qdrant\n", docker_ps_out="",
    )
    assert _start(monkeypatch, fake_run) is True
    assert calls == [
        ["docker", "ps", "-a", "--format", "{{.Names}}"],
        ["docker", "ps", "--format", "{{.Names}}"],
        ["docker", "start", "qdrant"],
    ]


def test_missing_container_run_with_volume(monkeypatch):
    """三态-不存在：docker ps -a 为空 → docker run 创建并挂 qdrant_data 卷."""
    fake_run, calls = _make_fake_run(docker_ps_a_out="")
    assert _start(monkeypatch, fake_run) is True
    assert calls[0] == ["docker", "ps", "-a", "--format", "{{.Names}}"]
    assert len(calls) == 2
    run_cmd = calls[-1]
    assert run_cmd[:2] == ["docker", "run"]
    assert run_cmd[run_cmd.index("--name") + 1] == "qdrant"
    assert "-v" in run_cmd
    assert run_cmd[run_cmd.index("-v") + 1] == "qdrant_data:/qdrant/storage"


def test_exception_returns_false(monkeypatch):
    """异常路径：subprocess.run 抛异常 → 返回 False，不抛给调用方."""
    def boom(cmd, **kw):
        raise subprocess.SubprocessError("boom")

    assert _start(monkeypatch, boom) is False


def test_start_failure_returns_false_no_run_fallback(monkeypatch):
    """docker start 失败（returncode != 0）→ 直接返回 False，不执行 docker run.

    容器名已存在时回退 `docker run` 必然失败（docker 不允许同名容器），
    直接 False 更正确——与 cli._start_qdrant 一致。
    """
    fake_run, calls = _make_fake_run(
        docker_ps_a_out="qdrant\n", docker_ps_out="", rc={"start": 1},
    )
    assert _start(monkeypatch, fake_run) is False
    assert calls == [
        ["docker", "ps", "-a", "--format", "{{.Names}}"],
        ["docker", "ps", "--format", "{{.Names}}"],
        ["docker", "start", "qdrant"],
    ]
    assert not any(c[1] == "run" for c in calls)


def test_run_failure_returns_false(monkeypatch):
    """docker run 失败（returncode != 0）→ 返回 False."""
    fake_run, _ = _make_fake_run(docker_ps_a_out="", rc={"run": 1})
    assert _start(monkeypatch, fake_run) is False
