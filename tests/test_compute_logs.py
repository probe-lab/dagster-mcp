"""get_compute_logs: finding a step's log files and reading the end of long logs.

A fake Dagster answers over httpx.post. It keeps each log in memory and reads
it the way Dagster's compute log managers do: from the byte offsets in the
cursor 'stdout_offset:stderr_offset', up to 'limit' bytes, returning the new
offsets as the next cursor.
"""

import re
from unittest.mock import MagicMock

import httpx
import pytest

from dagster_mcp import server
from dagster_mcp.server import get_compute_logs

RUN = "run-1"


def _capture(file_key, step_keys, **extra):
    return {"__typename": "LogsCapturedEvent", "fileKey": file_key, "stepKeys": step_keys, **extra}


class FakeDagster:
    def __init__(self, logs, events, *, page_size=None, cursor_format=True, location="/logs"):
        self.logs = logs  # file_key -> {"stdout": bytes, "stderr": bytes}
        self.location = location  # None: a cloud log that is not complete
        self.events = events
        self.page_size = page_size
        self.cursor_format = cursor_format
        self.requests = []
        self.sent = []

    def _read(self, file_key, cursor, limit):
        data = self.logs.get(file_key, {})
        offsets = [int(part) for part in cursor.split(":")] if cursor else [0, 0]
        result = {}
        for index, stream in enumerate(("stdout", "stderr")):
            chunk = data.get(stream, b"")[offsets[index] :]
            chunk = chunk if limit is None else chunk[:limit]
            offsets[index] += len(chunk)
            result[stream] = chunk.decode("utf-8", errors="replace") if chunk else None
        result["cursor"] = f"{offsets[0]}:{offsets[1]}" if self.cursor_format else "opaque"
        return result

    def __call__(self, url, **kwargs):
        query, variables = kwargs["json"]["query"], kwargs["json"]["variables"]
        name = re.search(r"\b(?:query|mutation)\s+(\w+)", query).group(1)
        self.requests.append(name)
        self.sent.append((name, variables))
        if name == "CapturedLogEvents":
            # The server picks the page size; Dagster 1.13.6 rejects a larger limit.
            assert "limit" not in variables
            start = int(variables["afterCursor"] or 0)
            size = self.page_size or len(self.events)
            page = self.events[start : start + size]
            data = {
                "logsForRun": {
                    "__typename": "EventConnection",
                    "cursor": str(start + size),
                    "hasMore": start + size < len(self.events),
                    "events": page,
                }
            }
        elif name == "CapturedLogs":
            file_key = variables["logKey"][2]
            data = {
                alias: self._read(file_key, variables[f"c{alias[1:]}"], variables[f"l{alias[1:]}"])
                for alias in re.findall(r"\b(r\d+): capturedLogs", query)
            }
            if "meta: capturedLogsMetadata" in query:
                base = self.location and f"{self.location}/{file_key}"
                data["meta"] = {
                    "stdoutLocation": base and f"{base}.out",
                    "stderrLocation": base and f"{base}.err",
                    "stdoutDownloadUrl": None,
                    "stderrDownloadUrl": None,
                }
        else:
            raise AssertionError(f"unexpected operation {name}")
        response = MagicMock(status_code=200)
        response.json.return_value = {"data": data}
        response.text = "{}"
        return response


@pytest.fixture
def fake(monkeypatch):
    def install(logs, events, **options):
        dagster = FakeDagster(logs, events, **options)
        monkeypatch.setattr(httpx, "post", dagster)
        return dagster

    return install


def _lines(count, prefix):
    return "".join(f"{prefix} line {i:07d}\n" for i in range(count)).encode()


class TestSelection:
    def test_single_capture_is_read_without_step_key(self, fake):
        fake({"aaa": {"stdout": b"hello\n", "stderr": b"oops\n"}}, [_capture("aaa", ["s1", "s2"])])
        result = get_compute_logs(RUN)
        [capture] = result["captures"]
        assert capture["step_keys"] == ["s1", "s2"]
        assert capture["stdout"] == {"text": "hello\n", "size_bytes": 6, "truncated": False}
        assert capture["stderr"] == {"text": "oops\n", "size_bytes": 5, "truncated": False}
        assert capture["stdout_location"] == "/logs/aaa.out"

    def test_several_captures_without_step_key_lists_them(self, fake):
        dagster = fake({}, [_capture("aaa", ["s1"]), _capture("bbb", ["s2"])])
        result = get_compute_logs(RUN)
        assert result["captures"] == [
            {"file_key": "aaa", "step_keys": ["s1"]},
            {"file_key": "bbb", "step_keys": ["s2"]},
        ]
        assert "Pass step_key" in result["message"]
        assert "CapturedLogs" not in dagster.requests

    def test_step_key_picks_its_captures_including_retries(self, fake):
        logs = {key: {"stderr": f"attempt {key}\n".encode()} for key in ("a1", "b1", "a2")}
        fake(logs, [_capture("a1", ["s1"]), _capture("b1", ["s2"]), _capture("a2", ["s1"])])
        result = get_compute_logs(RUN, step_key="s1", stream="stderr")
        assert [c["file_key"] for c in result["captures"]] == ["a1", "a2"]
        assert result["captures"][1]["stderr"]["text"] == "attempt a2\n"
        assert "stdout" not in result["captures"][0]

    def test_only_the_last_attempts_are_read(self, fake):
        keys = [f"k{i}" for i in range(8)]
        fake({}, [_capture(key, ["s1"]) for key in keys])
        result = get_compute_logs(RUN, step_key="s1")
        assert [c["file_key"] for c in result["captures"]] == keys[-5:]

    def test_unknown_step_names_the_steps_with_logs(self, fake):
        fake({}, [_capture("aaa", ["s2", "s1"])])
        result = get_compute_logs(RUN, step_key="nope")
        assert result["captures"] == []
        assert result["message"] == "No captured logs for step 'nope'. Steps with logs: s1, s2."

    def test_run_without_captures(self, fake):
        fake({}, [{"__typename": "RunStartEvent"}])
        result = get_compute_logs(RUN)
        assert result["captures"] == []
        assert "no captured logs" in result["message"]

    def test_events_are_paged_and_duplicates_dropped(self, fake):
        events = [{"__typename": "RunStartEvent"}] * 3 + [_capture("aaa", ["s1"])] * 2
        dagster = fake({"aaa": {"stdout": b"x"}}, events, page_size=2)
        result = get_compute_logs(RUN)
        assert [c["file_key"] for c in result["captures"]] == ["aaa"]
        assert dagster.requests.count("CapturedLogEvents") == 3

    def test_scan_limit_is_reported(self, fake, monkeypatch):
        monkeypatch.setattr(server, "_CAPTURE_EVENTS_MAX_PAGES", 2)
        events = [_capture("aaa", ["s1"])] + [{"__typename": "RunStartEvent"}] * 5
        fake({"aaa": {"stdout": b"x"}}, events, page_size=1)
        result = get_compute_logs(RUN)
        assert result["note"].startswith("Only the first 2 events")
        assert result["captures"][0]["file_key"] == "aaa"

    def test_external_urls_are_returned(self, fake):
        fake({}, [_capture("aaa", ["s1"], externalStdoutUrl="https://logs/out",
                           externalStderrUrl=None)])
        [capture] = get_compute_logs(RUN)["captures"]
        assert capture["external_stdout_url"] == "https://logs/out"
        assert "external_stderr_url" not in capture

    def test_run_not_found_raises(self, monkeypatch):
        monkeypatch.setattr(server, "gql", lambda *a, **k: {
            "logsForRun": {"__typename": "RunNotFoundError", "message": "Run x not found"}})
        with pytest.raises(RuntimeError, match="Run x not found"):
            get_compute_logs("x")

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [({"stream": "logs"}, "stream must be"), ({"max_bytes": 10}, "max_bytes must be")],
    )
    def test_invalid_arguments(self, kwargs, message):
        with pytest.raises(ValueError, match=message):
            get_compute_logs(RUN, **kwargs)


class TestTail:
    @pytest.mark.parametrize("size", [5_000, 999, 1_000, 1_001, 2_000, 123_457, 6_000_000])
    def test_the_end_of_the_log_is_returned(self, fake, size):
        stdout = (b"0123456789" * (size // 10 + 1))[:size]
        dagster = fake({"aaa": {"stdout": stdout, "stderr": b"short\n"}}, [_capture("aaa", ["s"])])
        [capture] = get_compute_logs(RUN, max_bytes=1_000)["captures"]
        out = capture["stdout"]
        assert out["size_bytes"] == size
        assert out["truncated"] == (size > 1_000)
        assert out["text"].encode() == stdout[-1_000:]
        assert capture["stderr"] == {"text": "short\n", "size_bytes": 6, "truncated": False}
        assert dagster.requests.count("CapturedLogs") <= 8

    def test_tail_starts_at_a_full_line(self, fake):
        stdout = _lines(5_000, "out")
        fake({"aaa": {"stdout": stdout}}, [_capture("aaa", ["s"])])
        text = get_compute_logs(RUN, stream="stdout", max_bytes=1_000)["captures"][0]["stdout"]["text"]
        assert text.endswith("out line 0004999\n")
        assert text.startswith("out line ")
        assert len(text.encode()) <= 1_000

    def test_both_streams_long(self, fake):
        logs = {"stdout": _lines(20_000, "out"), "stderr": _lines(300, "err")}
        fake({"aaa": logs}, [_capture("aaa", ["s"])])
        capture = get_compute_logs(RUN, max_bytes=500)["captures"][0]
        assert capture["stdout"]["text"].endswith("out line 0019999\n")
        assert capture["stderr"]["text"].endswith("err line 0000299\n")
        assert capture["stdout"]["size_bytes"] == len(logs["stdout"])
        assert capture["stderr"]["size_bytes"] == len(logs["stderr"])

    def test_multibyte_characters_are_not_broken(self, fake):
        stdout = ("äöü€" * 2_000).encode()
        fake({"aaa": {"stdout": stdout}}, [_capture("aaa", ["s"])])
        text = get_compute_logs(RUN, stream="stdout", max_bytes=1_001)["captures"][0]["stdout"]["text"]
        assert "�" not in text
        assert text.endswith("äöü€")

    def test_log_past_the_probe_limit_shows_the_start(self, fake, monkeypatch):
        monkeypatch.setattr(server, "_LOG_PROBE_LIMIT", 4_000)
        stdout = _lines(1_000, "out")
        fake({"aaa": {"stdout": stdout}}, [_capture("aaa", ["s"])])
        out = get_compute_logs(RUN, stream="stdout", max_bytes=1_000)["captures"][0]["stdout"]
        assert out["text"].startswith("out line 0000000")
        assert "only its start" in out["note"]

    def test_unknown_cursor_format_shows_the_start(self, fake):
        fake({"aaa": {"stdout": b"y" * 5_000, "stderr": b"e"}}, [_capture("aaa", ["s"])],
             cursor_format=False)
        capture = get_compute_logs(RUN, max_bytes=1_000)["captures"][0]
        assert capture["stdout"]["text"] == "y" * 1_000
        assert capture["stdout"]["truncated"] is None
        assert "only the start" in capture["stdout"]["note"]
        assert "note" not in capture["stderr"]

    def test_reads_use_one_byte_probes(self, fake):
        dagster = fake({"aaa": {"stdout": b"z" * 3_000_000}}, [_capture("aaa", ["s"])])
        get_compute_logs(RUN, stream="stdout", max_bytes=10_000)
        reads = [variables for name, variables in dagster.sent if name == "CapturedLogs"]
        limits = [v[k] for v in reads for k in sorted(v) if re.fullmatch(r"l\d+", k)]
        # One first read, one-byte probes, then one final read of at most 2 * max_bytes.
        assert limits[0] == 10_000
        assert set(limits[1:-1]) == {1}
        assert limits[-1] <= 20_000

    def test_log_without_a_location_is_not_probed(self, fake):
        dagster = fake({"aaa": {"stdout": b"p" * 50_000}}, [_capture("aaa", ["s"])], location=None)
        out = get_compute_logs(RUN, stream="stdout", max_bytes=1_000)["captures"][0]["stdout"]
        assert out["text"] == "p" * 1_000
        assert out["truncated"] is True
        assert "no storage location" in out["note"]
        assert dagster.requests.count("CapturedLogs") == 1

    def test_log_cut_at_upload_is_flagged(self, fake, monkeypatch):
        monkeypatch.setattr(server, "_LOG_UPLOAD_CUT_BYTES", 5_000)
        fake({"aaa": {"stdout": b"c" * 5_000}}, [_capture("aaa", ["s"])], location="s3://bucket")
        out = get_compute_logs(RUN, stream="stdout", max_bytes=1_000)["captures"][0]["stdout"]
        assert out["size_bytes"] == 5_000
        assert "not the real end" in out["note"]

    def test_local_log_of_the_cut_size_is_not_flagged(self, fake, monkeypatch):
        monkeypatch.setattr(server, "_LOG_UPLOAD_CUT_BYTES", 5_000)
        fake({"aaa": {"stdout": b"c" * 5_000}}, [_capture("aaa", ["s"])])
        out = get_compute_logs(RUN, stream="stdout", max_bytes=1_000)["captures"][0]["stdout"]
        assert "note" not in out
