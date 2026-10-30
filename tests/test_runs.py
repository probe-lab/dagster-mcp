import json
from unittest.mock import MagicMock

import httpx
import pytest

from dagster_mcp.server import (
    get_run_failure_summary,
    get_run_logs,
    get_run_stats,
    get_run_status,
    get_runs,
)


def _mock_response(data, status_code=200):
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = data
    resp.text = json.dumps(data)
    return resp


class TestGetRuns:
    def test_no_filters(self, mock_gql):
        mock_gql({"data": {"runsOrError": {"results": [
            {"runId": "r1", "status": "SUCCESS", "jobName": "job_a",
             "startTime": 1000, "endTime": 1100, "tags": []},
        ]}}})
        result = get_runs()
        assert len(result) == 1
        assert result[0]["runId"] == "r1"

    def test_with_job_name(self, mock_gql):
        mock_post = mock_gql({"data": {"runsOrError": {"results": []}}})
        get_runs(job_name="my_job")
        mock_post.assert_called_once()
        payload = mock_post.call_args.kwargs["json"]
        assert payload["variables"]["filter"] == {"pipelineName": "my_job"}

    def test_tags_run_ids_and_cursor(self, mock_gql):
        mock_post = mock_gql({"data": {"runsOrError": {"results": []}}})
        get_runs(
            tags={"dagster/partition": "2026-01-03", "team": "data"},
            run_ids=["r1", "r2"],
            cursor="r0",
        )
        variables = mock_post.call_args.kwargs["json"]["variables"]
        assert variables["filter"] == {
            "tags": [
                {"key": "dagster/partition", "value": "2026-01-03"},
                {"key": "team", "value": "data"},
            ],
            "runIds": ["r1", "r2"],
        }
        assert variables["cursor"] == "r0"

    def test_time_filters_accept_iso_strings_and_unix_seconds(self, mock_gql):
        mock_post = mock_gql({"data": {"runsOrError": {"results": []}}})
        get_runs(
            created_after="2026-01-31T00:00:00Z",
            # No time zone means UTC.
            created_before="2026-02-01T00:00:00",
            updated_after=1767225600,
        )
        run_filter = mock_post.call_args.kwargs["json"]["variables"]["filter"]
        assert run_filter == {
            "createdAfter": 1769817600.0,
            "createdBefore": 1769904000.0,
            "updatedAfter": 1767225600.0,
        }

    def test_bad_timestamp_raises_before_any_request(self, mock_gql):
        mock_post = mock_gql({"data": {"runsOrError": {"results": []}}})
        with pytest.raises(ValueError, match="created_after"):
            get_runs(created_after="yesterday")
        mock_post.assert_not_called()

    def test_statuses_are_upper_cased(self, mock_gql):
        mock_post = mock_gql({"data": {"runsOrError": {"results": []}}})
        get_runs(statuses=["failure"])
        assert mock_post.call_args.kwargs["json"]["variables"]["filter"] == {
            "statuses": ["FAILURE"]
        }

    def test_invalid_filter_error_raises(self, mock_gql):
        mock_gql({"data": {"runsOrError": {
            "__typename": "InvalidPipelineRunsFilterError",
            "message": "bad filter",
        }}})
        with pytest.raises(RuntimeError, match="bad filter"):
            get_runs()

    def test_with_statuses(self, mock_gql):
        mock_post = mock_gql({"data": {"runsOrError": {"results": []}}})
        get_runs(statuses=["FAILURE", "CANCELED"])
        payload = mock_post.call_args.kwargs["json"]
        assert payload["variables"]["filter"]["statuses"] == ["FAILURE", "CANCELED"]

    def test_empty_results(self, mock_gql):
        mock_gql({"data": {"runsOrError": {"results": []}}})
        assert get_runs() == []

    def test_custom_limit(self, mock_gql):
        mock_post = mock_gql({"data": {"runsOrError": {"results": []}}})
        get_runs(limit=5)
        payload = mock_post.call_args.kwargs["json"]
        assert payload["variables"]["limit"] == 5


class TestGetRunStatus:
    def test_found(self, mock_gql):
        mock_gql({"data": {"runOrError": {
            "runId": "r1", "status": "SUCCESS", "startTime": 1000,
            "endTime": 1100, "jobName": "job_a", "tags": [],
            "runConfigYaml": "{}",
            "rootRunId": "r1", "parentRunId": None,
            "resolvedOpSelection": None,
        }}})
        result = get_run_status("r1")
        assert result["status"] == "SUCCESS"
        assert result["runId"] == "r1"
        assert result["rootRunId"] == "r1"
        assert result["parentRunId"] is None

    def test_reexecuted_run(self, mock_gql):
        mock_gql({"data": {"runOrError": {
            "runId": "r2", "status": "SUCCESS", "startTime": 2000,
            "endTime": 2100, "jobName": "job_a", "tags": [],
            "runConfigYaml": "{}",
            "rootRunId": "r1", "parentRunId": "r1",
            "resolvedOpSelection": ["step_a"],
        }}})
        result = get_run_status("r2")
        assert result["parentRunId"] == "r1"
        assert result["rootRunId"] == "r1"
        assert result["resolvedOpSelection"] == ["step_a"]

    def test_not_found(self, mock_gql):
        mock_gql({"data": {"runOrError": {"message": "Run r999 not found"}}})
        result = get_run_status("r999")
        assert "message" in result

    def test_python_error(self, mock_gql):
        mock_gql({"data": {"runOrError": {"message": "Internal error"}}})
        result = get_run_status("bad")
        assert result["message"] == "Internal error"


class TestGetRunLogs:
    def test_events_returned(self, mock_gql):
        mock_gql({"data": {"logsForRun": {
            "cursor": "c1", "hasMore": False,
            "events": [
                {"__typename": "RunStartEvent", "timestamp": "1000",
                 "message": "Started", "level": "INFO"},
                {"__typename": "ExecutionStepFailureEvent", "timestamp": "1100",
                 "message": "Step failed", "level": "ERROR", "stepKey": "step_a",
                 "error": {"message": "boom", "causes": []}},
            ],
        }}})
        result = get_run_logs("r1")
        assert result["hasMore"] is False
        assert len(result["events"]) == 2

    def test_engine_event_metadata_flattened(self, mock_gql):
        mock_gql({"data": {"logsForRun": {
            "cursor": "c1", "hasMore": False,
            "events": [
                {"__typename": "EngineEvent", "timestamp": "1000",
                 "message": "Launching run in k8s", "level": "INFO",
                 "stepKey": None, "error": None,
                 "metadataEntries": [
                     {"__typename": "TextMetadataEntry", "label": "pod",
                      "description": None, "text": "run-abc"},
                     {"__typename": "UrlMetadataEntry", "label": "logs",
                      "description": "dashboard", "url": "http://x/logs"},
                     {"__typename": "IntMetadataEntry", "label": "retries",
                      "description": None, "intValue": 3},
                     {"__typename": "PythonArtifactMetadataEntry", "label": "image",
                      "description": None, "module": "acme", "name": "worker"},
                 ]},
            ],
        }}})
        result = get_run_logs("r1")
        md = result["events"][0]["metadataEntries"]
        assert md[0] == {"label": "pod", "description": None, "value": "run-abc"}
        assert md[1]["value"] == "http://x/logs"
        assert md[2]["value"] == 3
        assert md[3]["value"] == "acme.worker"

    def test_pagination(self, mock_gql):
        mock_post = mock_gql({"data": {"logsForRun": {
            "cursor": "c2", "hasMore": True, "events": [],
        }}})
        result = get_run_logs("r1", cursor="c1", limit=50)
        payload = mock_post.call_args.kwargs["json"]
        assert payload["variables"]["afterCursor"] == "c1"
        assert payload["variables"]["limit"] == 50
        assert result["hasMore"] is True

    def test_not_found(self, mock_gql):
        mock_gql({"data": {"logsForRun": {"message": "Run not found"}}})
        result = get_run_logs("r999")
        assert "message" in result

    def test_level_filter_error(self, mock_gql):
        mock_gql({"data": {"logsForRun": {
            "cursor": "c1", "hasMore": False,
            "events": [
                {"__typename": "RunStartEvent", "timestamp": "1000",
                 "message": "Started", "level": "INFO"},
                {"__typename": "EngineEvent", "timestamp": "1050",
                 "message": "Warning msg", "level": "WARNING", "stepKey": None,
                 "error": None},
                {"__typename": "ExecutionStepFailureEvent", "timestamp": "1100",
                 "message": "Step failed", "level": "ERROR", "stepKey": "step_a",
                 "error": {"message": "boom", "causes": []}},
                {"__typename": "RunFailureEvent", "timestamp": "1200",
                 "message": "Run failed", "level": "CRITICAL",
                 "error": {"message": "steps failed", "causes": []}},
            ],
        }}})
        result = get_run_logs("r1", level_filter="ERROR")
        # Should include the ERROR-level event AND the RunFailureEvent (even though CRITICAL)
        assert len(result["events"]) == 2
        types = {e["__typename"] for e in result["events"]}
        assert "ExecutionStepFailureEvent" in types
        assert "RunFailureEvent" in types

    def test_level_filter_none_returns_all(self, mock_gql):
        mock_gql({"data": {"logsForRun": {
            "cursor": "c1", "hasMore": False,
            "events": [
                {"__typename": "RunStartEvent", "timestamp": "1000",
                 "message": "Started", "level": "INFO"},
                {"__typename": "ExecutionStepFailureEvent", "timestamp": "1100",
                 "message": "Failed", "level": "ERROR", "stepKey": "s",
                 "error": {"message": "x", "causes": []}},
            ],
        }}})
        result = get_run_logs("r1", level_filter=None)
        assert len(result["events"]) == 2

    def test_level_filter_warning(self, mock_gql):
        mock_gql({"data": {"logsForRun": {
            "cursor": "c1", "hasMore": False,
            "events": [
                {"__typename": "EngineEvent", "timestamp": "1000",
                 "message": "warn", "level": "WARNING", "stepKey": None,
                 "error": None},
                {"__typename": "RunStartEvent", "timestamp": "900",
                 "message": "start", "level": "INFO"},
            ],
        }}})
        result = get_run_logs("r1", level_filter="WARNING")
        assert len(result["events"]) == 1
        assert result["events"][0]["level"] == "WARNING"

    def test_level_filter_includes_higher_levels(self, mock_gql):
        # Filtering at WARNING must not drop ERROR/CRITICAL: the docstring
        # promises "this level or above", and an agent triaging an incident
        # would otherwise be told there were no errors.
        mock_gql({"data": {"logsForRun": {
            "cursor": "c1", "hasMore": False,
            "events": [
                {"__typename": "RunStartEvent", "timestamp": "800",
                 "message": "debug", "level": "DEBUG"},
                {"__typename": "RunStartEvent", "timestamp": "900",
                 "message": "start", "level": "INFO"},
                {"__typename": "EngineEvent", "timestamp": "1000",
                 "message": "warn", "level": "WARNING", "stepKey": None,
                 "error": None},
                {"__typename": "EngineEvent", "timestamp": "1100",
                 "message": "boom", "level": "ERROR", "stepKey": None,
                 "error": None},
                {"__typename": "EngineEvent", "timestamp": "1200",
                 "message": "fatal", "level": "CRITICAL", "stepKey": None,
                 "error": None},
            ],
        }}})
        result = get_run_logs("r1", level_filter="WARNING")
        assert [e["level"] for e in result["events"]] == [
            "WARNING", "ERROR", "CRITICAL",
        ]

    def test_level_filter_keeps_failure_events_below_error(self, mock_gql):
        # A failure event whose own level is INFO must still surface when
        # filtering at WARNING, not only at ERROR.
        mock_gql({"data": {"logsForRun": {
            "cursor": "c1", "hasMore": False,
            "events": [
                {"__typename": "ExecutionStepFailureEvent", "timestamp": "1000",
                 "message": "step failed", "level": "INFO", "stepKey": "s",
                 "error": {"message": "boom", "causes": []}},
                {"__typename": "RunStartEvent", "timestamp": "900",
                 "message": "start", "level": "INFO"},
            ],
        }}})
        result = get_run_logs("r1", level_filter="WARNING")
        assert [e["__typename"] for e in result["events"]] == [
            "ExecutionStepFailureEvent",
        ]

    def test_level_filter_rejects_unknown_level(self, mock_gql):
        mock_gql({"data": {"logsForRun": {"cursor": "c", "hasMore": False,
                                          "events": []}}})
        with pytest.raises(ValueError, match="level_filter must be one of"):
            get_run_logs("r1", level_filter="LOUD")


class TestGetRunStats:
    def test_step_stats(self, mock_gql):
        mock_gql({"data": {"runOrError": {
            "runId": "r1", "status": "SUCCESS",
            "stepStats": [
                {"stepKey": "step_a", "status": "SUCCESS",
                 "startTime": 1000, "endTime": 1050,
                 "materializations": [{"label": "mat1"}],
                 "expectationResults": [{"success": True, "label": "exp1"}]},
            ],
        }}})
        result = get_run_stats("r1")
        assert result["runId"] == "r1"
        assert len(result["stepStats"]) == 1
        assert result["stepStats"][0]["stepKey"] == "step_a"

    def test_not_found(self, mock_gql):
        mock_gql({"data": {"runOrError": {"message": "Run not found"}}})
        result = get_run_stats("r999")
        assert result["message"] == "Run not found"


def _make_mock_post(responses):
    """Create a mock httpx.post that returns different responses on successive calls."""
    call_count = 0

    def _post(*args, **kwargs):
        nonlocal call_count
        resp_data = responses[min(call_count, len(responses) - 1)]
        call_count += 1
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = resp_data
        mock_resp.text = json.dumps(resp_data)
        return mock_resp

    return _post


class TestGetRunFailureSummary:
    def test_failed_run_two_steps(self, monkeypatch):
        responses = [
            # 1st call: status + stats
            {"data": {"runOrError": {
                "runId": "r1", "status": "FAILURE", "jobName": "job_a",
                "startTime": 1000, "endTime": 1200,
                "stepStats": [
                    {"stepKey": "extract", "status": "SUCCESS",
                     "startTime": 1000, "endTime": 1050},
                    {"stepKey": "transform", "status": "FAILURE",
                     "startTime": 1050, "endTime": 1100},
                    {"stepKey": "load", "status": "FAILURE",
                     "startTime": 1100, "endTime": 1150},
                ],
            }}},
            # 2nd call: logs
            {"data": {"logsForRun": {
                "cursor": "c1", "hasMore": False,
                "events": [
                    {"__typename": "ExecutionStepFailureEvent",
                     "timestamp": "1100", "stepKey": "transform",
                     "error": {"message": "NullPointerError", "causes": [{"message": "col X is null"}]}},
                    {"__typename": "ExecutionStepFailureEvent",
                     "timestamp": "1150", "stepKey": "load",
                     "error": {"message": "Upstream failed", "causes": []}},
                    {"__typename": "RunFailureEvent",
                     "timestamp": "1200",
                     "error": {"message": "Steps failed", "causes": []}},
                ],
            }}},
        ]
        monkeypatch.setattr(httpx, "post", _make_mock_post(responses))
        result = get_run_failure_summary("r1")

        assert result["status"] == "FAILURE"
        assert result["job_name"] == "job_a"
        assert result["duration_seconds"] == 200.0
        assert len(result["failed_steps"]) == 2
        assert result["failed_steps"][0]["step_key"] == "transform"
        assert result["failed_steps"][0]["error"]["message"] == "NullPointerError"
        assert result["root_cause_error"]["message"] == "Steps failed"
        assert len(result["all_step_durations"]) == 3
        assert any("Multiple steps failed" in s for s in result["suggestions"])

    def test_errors_keep_class_and_the_end_of_the_stack(self, monkeypatch):
        frames = [f'  File "f{i}.py", line {i}\n' for i in range(12)]
        responses = [
            {"data": {"runOrError": {
                "runId": "r1", "status": "FAILURE", "jobName": "job_a",
                "stepStats": [{"stepKey": "s", "status": "FAILURE"}],
            }}},
            {"data": {"logsForRun": {"hasMore": False, "events": [
                {"__typename": "ExecutionStepFailureEvent", "stepKey": "s", "error": {
                    "message": "Error occurred while executing op s",
                    "className": "DagsterExecutionStepExecutionError",
                    "stack": [],
                    "causes": [{
                        "message": "KeyError: 'price'",
                        "className": "KeyError",
                        "stack": frames,
                    }],
                }},
            ]}}},
        ]
        monkeypatch.setattr(httpx, "post", _make_mock_post(responses))

        error = get_run_failure_summary("r1")["failed_steps"][0]["error"]

        assert error == {
            "message": "Error occurred while executing op s",
            "className": "DagsterExecutionStepExecutionError",
            "causes": [{
                "message": "KeyError: 'price'",
                "className": "KeyError",
                "stack": "".join(frames[-8:]),
                "stack_frames_omitted": 4,
            }],
        }

    def test_successful_run(self, mock_gql):
        mock_gql({"data": {"runOrError": {
            "runId": "r2", "status": "SUCCESS", "jobName": "j",
            "startTime": 1000, "endTime": 1100,
            "stepStats": [],
        }}})
        result = get_run_failure_summary("r2")
        assert result["message"] == "Run did not fail."

    def test_not_found(self, mock_gql):
        mock_gql({"data": {"runOrError": {"message": "Run not found"}}})
        result = get_run_failure_summary("r999")
        assert result["message"] == "Run not found"

    def test_canceled_run(self, monkeypatch):
        responses = [
            {"data": {"runOrError": {
                "runId": "r3", "status": "CANCELED", "jobName": "j",
                "startTime": 1000, "endTime": 1100,
                "stepStats": [],
            }}},
            {"data": {"logsForRun": {
                "cursor": None, "hasMore": False, "events": [],
            }}},
        ]
        monkeypatch.setattr(httpx, "post", _make_mock_post(responses))
        result = get_run_failure_summary("r3")
        assert result["status"] == "CANCELED"
        assert any("canceled" in s.lower() for s in result["suggestions"])

    def test_retry_suggestion(self, monkeypatch):
        responses = [
            {"data": {"runOrError": {
                "runId": "r4", "status": "FAILURE", "jobName": "j",
                "startTime": 1000, "endTime": 1200,
                "stepStats": [
                    {"stepKey": "flaky", "status": "FAILURE",
                     "startTime": 1000, "endTime": 1200},
                ],
            }}},
            {"data": {"logsForRun": {
                "cursor": None, "hasMore": False,
                "events": [
                    {"__typename": "ExecutionStepUpForRetryEvent",
                     "timestamp": "1100", "stepKey": "flaky",
                     "secondsToWait": 30,
                     "error": {"message": "timeout", "causes": []}},
                    {"__typename": "ExecutionStepFailureEvent",
                     "timestamp": "1200", "stepKey": "flaky",
                     "error": {"message": "timeout again", "causes": []}},
                ],
            }}},
        ]
        monkeypatch.setattr(httpx, "post", _make_mock_post(responses))
        result = get_run_failure_summary("r4")
        assert any("retried" in s.lower() for s in result["suggestions"])

    def test_step_error_prefers_the_failure_over_earlier_retries(self, monkeypatch):
        responses = [
            {"data": {"runOrError": {
                "runId": "r4", "status": "FAILURE", "jobName": "j",
                "tags": [{"key": "dagster/failure_reason", "value": "STEP_FAILURE"}],
                "stepStats": [{"stepKey": "flaky", "status": "FAILURE"}],
            }}},
            {"data": {"logsForRun": {"hasMore": False, "events": [
                {"__typename": "ExecutionStepUpForRetryEvent", "stepKey": "flaky",
                 "error": {"message": "timeout", "causes": []}},
                {"__typename": "ExecutionStepFailureEvent", "stepKey": "flaky",
                 "error": {"message": "timeout again", "causes": []}},
                # A run that failed because steps failed has no run-level error.
                {"__typename": "RunFailureEvent", "error": None},
            ]}}},
        ]
        monkeypatch.setattr(httpx, "post", _make_mock_post(responses))
        result = get_run_failure_summary("r4")
        assert result["failure_reason"] == "STEP_FAILURE"
        assert result["failed_steps"][0]["error"] == {"message": "timeout again"}
        assert result["root_cause_error"] == {"message": "timeout again"}
        assert result["retried_steps"] == [{"step_key": "flaky", "retries": 1}]

    def test_interrupted_run_with_a_retried_step(self, monkeypatch):
        retry_error = {"message": "RetryRequestedFromPolicy", "className": "RetryRequestedFromPolicy",
                       "causes": [{"message": "OSError: checksum mismatch", "className": "OSError"}]}
        responses = [
            {"data": {"runOrError": {
                "runId": "r5", "status": "FAILURE", "jobName": "j",
                "tags": [{"key": "dagster/failure_reason", "value": "UNEXPECTED_TERMINATION"}],
                "stepStats": [{"stepKey": "upload", "status": "IN_PROGRESS"}],
            }}},
            {"data": {"logsForRun": {"hasMore": True, "cursor": "c1", "events": [
                {"__typename": "ExecutionStepUpForRetryEvent", "stepKey": "upload",
                 "error": {"message": "first", "causes": []}},
            ]}}},
            {"data": {"logsForRun": {"hasMore": False, "cursor": "c2", "events": [
                {"__typename": "ExecutionStepUpForRetryEvent", "stepKey": "upload",
                 "error": retry_error},
                {"__typename": "RunFailureEvent", "error": {
                    "message": "DagsterExecutionInterruptedError",
                    "className": "DagsterExecutionInterruptedError", "causes": []}},
            ]}}},
        ]
        monkeypatch.setattr(httpx, "post", _make_mock_post(responses))
        result = get_run_failure_summary("r5")
        assert result["failure_reason"] == "UNEXPECTED_TERMINATION"
        assert result["failed_steps"] == []
        assert result["root_cause_error"]["className"] == "DagsterExecutionInterruptedError"
        assert result["retried_steps"] == [{
            "step_key": "upload", "retries": 2,
            "last_error": {"message": "RetryRequestedFromPolicy",
                           "className": "RetryRequestedFromPolicy",
                           "causes": [{"message": "OSError: checksum mismatch",
                                       "className": "OSError"}]},
        }]
        assert any("still running when the run failed: upload" in s for s in result["suggestions"])

    def test_events_are_read_past_the_first_pages(self, monkeypatch):
        from dagster_mcp import server

        monkeypatch.setattr(server, "_FAILURE_EVENTS_MAX_PAGES", 3)
        page = {"data": {"logsForRun": {"hasMore": True, "cursor": "c",
                                        "events": [{"__typename": "LogMessageEvent"}] * 2}}}
        responses = [
            {"data": {"runOrError": {"runId": "r6", "status": "FAILURE", "jobName": "j",
                                     "stepStats": []}}},
            page, page, page,
        ]
        mock_post = _make_mock_post(responses)
        monkeypatch.setattr(httpx, "post", mock_post)
        result = get_run_failure_summary("r6")
        assert result["root_cause_error"] is None
        assert any("Only the first 6 events" in s for s in result["suggestions"])
