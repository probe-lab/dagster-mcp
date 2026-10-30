import json
from unittest.mock import MagicMock

import httpx
import pytest

from dagster_mcp.server import (
    cancel_backfill,
    get_backfill,
    list_backfills,
    reexecute_run,
    resume_backfill,
    retry_backfill,
)


def _post_sequence(monkeypatch, *responses):
    def reply(data):
        response = MagicMock(status_code=200)
        response.json.return_value = {"data": data}
        response.text = json.dumps({"data": data})
        return response

    mock_post = MagicMock(side_effect=[reply(data) for data in responses])
    monkeypatch.setattr(httpx, "post", mock_post)
    return mock_post


def _summary(**fields):
    backfill = {
        "backfillId": "bf1",
        "status": "COMPLETED_FAILED",
        "isAssetBackfill": True,
        "title": None,
        "assetSelection": [{"path": ["raw", "orders"]}],
        "jobName": None,
        "partitionSetName": None,
        "numPartitions": 3,
        "fromFailure": False,
        "timestamp": 1000.0,
        "endTimestamp": 2000.0,
        "error": None,
    }
    backfill.update(fields)
    return backfill


class TestListBackfills:
    def test_summary(self, mock_gql):
        mock_gql({"data": {"partitionBackfillsOrError": {
            "__typename": "PartitionBackfills",
            "results": [_summary(error={"message": "daemon crashed"})],
        }}})

        [backfill] = list_backfills()

        assert backfill["backfillId"] == "bf1"
        assert backfill["assetSelection"] == ["raw/orders"]
        assert backfill["error"] == "daemon crashed"

    def test_filters_and_cursor(self, mock_gql):
        mock_post = mock_gql({"data": {"partitionBackfillsOrError": {"results": []}}})

        list_backfills(
            limit=5,
            statuses=["completed_failed", "FAILED"],
            created_after="2026-01-31T00:00:00Z",
            cursor="bf0",
        )

        variables = mock_post.call_args.kwargs["json"]["variables"]
        assert variables == {
            "limit": 5,
            "cursor": "bf0",
            "filters": {
                "statuses": ["COMPLETED_FAILED", "FAILED"],
                "createdAfter": 1769817600.0,
            },
        }

    def test_no_filters_sends_none(self, mock_gql):
        mock_post = mock_gql({"data": {"partitionBackfillsOrError": {"results": []}}})
        assert list_backfills() == []
        assert mock_post.call_args.kwargs["json"]["variables"]["filters"] is None

    def test_python_error_raises(self, mock_gql):
        mock_gql({"data": {"partitionBackfillsOrError": {
            "__typename": "PythonError", "message": "boom"}}})
        with pytest.raises(RuntimeError, match="boom"):
            list_backfills()


class TestGetBackfill:
    def test_asset_backfill(self, mock_gql):
        backfill = _summary(
            __typename="PartitionBackfill",
            description=None,
            user="me@example.com",
            tags=[{"key": "dagster/backfill", "value": "bf1"}],
            assetBackfillData={
                "assetBackfillStatuses": [
                    {"__typename": "AssetPartitionsStatusCounts",
                     "assetKey": {"path": ["raw", "orders"]},
                     "numPartitionsTargeted": 3, "numPartitionsInProgress": 0,
                     "numPartitionsMaterialized": 1, "numPartitionsFailed": 2},
                    {"__typename": "UnpartitionedAssetStatus",
                     "assetKey": {"path": ["report"]},
                     "inProgress": False, "materialized": False, "failed": True},
                ],
                "rootTargetedPartitions": {
                    "partitionKeys": None,
                    "ranges": [{"start": "2026-01-01", "end": "2026-01-03"}],
                },
            },
            partitionStatuses=None,
        )
        mock_post = mock_gql({"data": {
            "partitionBackfillOrError": backfill,
            "runsOrError": {"__typename": "Runs", "count": 2, "results": [
                {"runId": "r1", "status": "FAILURE", "tags": [
                    {"key": "dagster/partition", "value": "2026-01-02"}]},
                {"runId": "r2", "status": "CANCELED", "tags": [
                    {"key": "dagster/asset_partition_range_start", "value": "2026-01-01"},
                    {"key": "dagster/asset_partition_range_end", "value": "2026-01-03"}]},
            ]},
        }})

        result = get_backfill("bf1", run_limit=5)

        assert result["status"] == "COMPLETED_FAILED"
        assert result["error"] is None
        assert result["tags"] == {"dagster/backfill": "bf1"}
        assert result["assets"] == [
            {"asset_key": "raw/orders", "targeted": 3, "materialized": 1, "failed": 2,
             "in_progress": 0},
            {"asset_key": "report", "materialized": False, "failed": True,
             "in_progress": False},
        ]
        assert result["targeted_partitions"] == {
            "ranges": [{"start": "2026-01-01", "end": "2026-01-03"}]
        }
        assert result["failed_runs"] == [
            {"run_id": "r1", "status": "FAILURE", "partition": "2026-01-02"},
            {"run_id": "r2", "status": "CANCELED",
             "partition_range_start": "2026-01-01", "partition_range_end": "2026-01-03"},
        ]
        assert result["failed_runs_total"] == 2
        assert "assetBackfillData" not in result
        assert "__typename" not in result
        variables = mock_post.call_args.kwargs["json"]["variables"]
        assert variables == {
            "backfillId": "bf1",
            "runFilter": {
                "tags": [{"key": "dagster/backfill", "value": "bf1"}],
                "statuses": ["FAILURE", "CANCELED"],
            },
            "runLimit": 5,
        }

    def test_job_backfill_groups_partitions_by_run_status(self, mock_gql):
        statuses = [
            {"partitionName": f"p{i}", "runId": f"r{i}", "runStatus": "SUCCESS"}
            for i in range(250)
        ] + [
            {"partitionName": "bad", "runId": "rb", "runStatus": "FAILURE"},
            {"partitionName": "later", "runId": None, "runStatus": None},
        ]
        mock_gql({"data": {
            "partitionBackfillOrError": _summary(
                __typename="PartitionBackfill",
                isAssetBackfill=False,
                assetSelection=None,
                jobName="daily_job",
                error={"message": "Backfill failed", "className": "Exception",
                       "stack": ["  File x\n"], "causes": []},
                assetBackfillData=None,
                partitionStatuses={"results": statuses},
            ),
            "runsOrError": {"count": 0, "results": []},
        }})

        result = get_backfill("bf1")

        assert result["partitions"]["FAILURE"] == {"count": 1, "keys": ["bad"]}
        assert result["partitions"]["NOT_STARTED"] == {"count": 1, "keys": ["later"]}
        assert result["partitions"]["SUCCESS"]["count"] == 250
        assert len(result["partitions"]["SUCCESS"]["keys"]) == 200
        assert result["partitions"]["SUCCESS"]["truncated"] is True
        assert result["error"] == {
            "message": "Backfill failed", "className": "Exception", "stack": "  File x\n",
        }
        assert result["assetSelection"] == []
        assert "assets" not in result

    def test_run_query_error_is_not_reported_as_no_failed_runs(self, mock_gql):
        mock_gql({"data": {
            "partitionBackfillOrError": _summary(__typename="PartitionBackfill"),
            "runsOrError": {"__typename": "PythonError", "message": "run storage is down"},
        }})
        result = get_backfill("bf1")
        assert result["failed_runs"] is None
        assert result["failed_runs_error"] == "run storage is down"

    def test_not_found(self, mock_gql):
        mock_gql({"data": {
            "partitionBackfillOrError": {"__typename": "BackfillNotFoundError",
                                         "message": "Backfill nope not found"},
            "runsOrError": {"count": 0, "results": []},
        }})
        assert get_backfill("nope") == {
            "backfill_id": "nope", "message": "Backfill nope not found",
        }


def _status(status, asset_backfill=True):
    return {"partitionBackfillOrError": {"__typename": "PartitionBackfill", "id": "bf1",
                                         "status": status,
                                         "isAssetBackfill": asset_backfill}}


class TestCancelBackfill:
    def test_cancels_an_in_progress_backfill(self, monkeypatch):
        mock_post = _post_sequence(
            monkeypatch,
            _status("REQUESTED"),
            {"cancelPartitionBackfill": {"__typename": "CancelBackfillSuccess",
                                         "backfillId": "bf1"}},
        )

        result = cancel_backfill("bf1")

        assert result["status"] == "CANCELING"
        assert result["previous_status"] == "REQUESTED"
        mutation = mock_post.call_args.kwargs["json"]
        assert "cancelPartitionBackfill" in mutation["query"]
        assert mutation["variables"] == {"backfillId": "bf1"}

    @pytest.mark.parametrize("status", ["COMPLETED_SUCCESS", "COMPLETED_FAILED", "FAILED",
                                        "CANCELED", "CANCELING", "FAILING"])
    def test_refuses_a_finished_or_stopping_backfill(self, monkeypatch, status):
        # Dagster would set even a finished backfill to CANCELING.
        mock_post = _post_sequence(monkeypatch, _status(status))

        result = cancel_backfill("bf1")

        assert result["status"] == status
        assert "No change was made" in result["message"]
        mock_post.assert_called_once()

    def test_not_found(self, monkeypatch):
        _post_sequence(monkeypatch, {"partitionBackfillOrError": {
            "__typename": "BackfillNotFoundError", "message": "no such backfill"}})
        assert cancel_backfill("bf1") == {"backfill_id": "bf1", "message": "no such backfill"}

    def test_unauthorized(self, monkeypatch):
        _post_sequence(
            monkeypatch,
            _status("REQUESTED"),
            {"cancelPartitionBackfill": {"__typename": "UnauthorizedError",
                                         "message": "not allowed"}},
        )
        result = cancel_backfill("bf1")
        assert result["message"] == "not allowed"
        assert result["status"] == "REQUESTED"


class TestResumeBackfill:
    def test_resumes_a_failed_backfill(self, monkeypatch):
        mock_post = _post_sequence(
            monkeypatch,
            _status("FAILED"),
            {"resumePartitionBackfill": {"__typename": "ResumeBackfillSuccess",
                                         "backfillId": "bf1"}},
        )

        result = resume_backfill("bf1")

        assert result["status"] == "REQUESTED"
        assert "resumePartitionBackfill" in mock_post.call_args.kwargs["json"]["query"]

    @pytest.mark.parametrize("status", ["REQUESTED", "COMPLETED_SUCCESS", "COMPLETED_FAILED",
                                        "CANCELED"])
    def test_refuses_any_other_status(self, monkeypatch, status):
        mock_post = _post_sequence(monkeypatch, _status(status))

        result = resume_backfill("bf1")

        assert "retry_backfill" in result["message"]
        mock_post.assert_called_once()


_RETRY_SUCCESS = {"reexecutePartitionBackfill": {
    "__typename": "LaunchBackfillSuccess", "backfillId": "bf2"}}


class TestRetryBackfill:
    @pytest.mark.parametrize(
        ("from_failure", "strategy"), [(True, "FROM_FAILURE"), (False, "ALL_STEPS")]
    )
    def test_launches_a_new_backfill(self, monkeypatch, from_failure, strategy):
        mock_post = _post_sequence(monkeypatch, _status("COMPLETED_FAILED"), _RETRY_SUCCESS)

        result = retry_backfill("bf1", from_failure=from_failure)

        assert result["backfillId"] == "bf2"
        assert "note" not in result
        assert mock_post.call_args.kwargs["json"]["variables"] == {
            "params": {"parentRunId": "bf1", "strategy": strategy}
        }

    @pytest.mark.parametrize("status", ["REQUESTED", "CANCELING", "FAILING"])
    def test_refuses_an_unfinished_backfill(self, monkeypatch, status):
        mock_post = _post_sequence(monkeypatch, _status(status))

        result = retry_backfill("bf1")

        assert "No change was made" in result["message"]
        mock_post.assert_called_once()

    def test_job_backfill_from_failure_warns_about_skipped_partitions(self, monkeypatch):
        # Dagster re-runs only partitions whose last run FAILED for a job
        # backfill; canceled and unlaunched ones are silently skipped.
        _post_sequence(monkeypatch, _status("CANCELED", asset_backfill=False), _RETRY_SUCCESS)

        result = retry_backfill("bf1")

        assert "launch_job_with_partitions" in result["note"]

    def test_dagster_refusal_raises(self, monkeypatch):
        _post_sequence(monkeypatch, _status("COMPLETED_SUCCESS"), {"reexecutePartitionBackfill": {
            "__typename": "PythonError",
            "message": "Cannot re-execute from failure an asset backfill that has no "
                       "missing materializations."}})
        with pytest.raises(RuntimeError, match="no missing materializations"):
            retry_backfill("bf1")


_REEXECUTED = {"launchRunReexecution": {
    "__typename": "LaunchRunSuccess", "run": {"runId": "r2", "status": "QUEUED"}}}


def _run(status):
    return {"runOrError": {"__typename": "Run", "status": status}}


class TestReexecuteRun:
    def test_from_failure_keeps_parent_tags(self, monkeypatch):
        mock_post = _post_sequence(monkeypatch, _run("FAILURE"), _REEXECUTED)

        result = reexecute_run("r1", tags={"triggered_by": "agent"})

        assert result["run"] == {"runId": "r2", "status": "QUEUED"}
        assert mock_post.call_args.kwargs["json"]["variables"] == {"params": {
            "parentRunId": "r1",
            "strategy": "FROM_FAILURE",
            "useParentRunTags": True,
            "extraTags": [{"key": "triggered_by", "value": "agent"}],
        }}

    def test_strategy_is_case_insensitive(self, monkeypatch):
        mock_post = _post_sequence(monkeypatch, _run("SUCCESS"), _REEXECUTED)
        reexecute_run("r1", strategy="all_steps")
        params = mock_post.call_args.kwargs["json"]["variables"]["params"]
        assert params["strategy"] == "ALL_STEPS"
        assert "extraTags" not in params

    def test_unknown_strategy(self, mock_gql):
        mock_post = mock_gql({"data": {}})
        with pytest.raises(ValueError, match="FROM_FAILURE"):
            reexecute_run("r1", strategy="RETRY")
        mock_post.assert_not_called()

    @pytest.mark.parametrize("status", ["STARTED", "QUEUED", "STARTING", "CANCELING"])
    @pytest.mark.parametrize("strategy", ["FROM_FAILURE", "ALL_STEPS"])
    def test_refuses_a_run_that_has_not_finished(self, monkeypatch, status, strategy):
        # Dagster checks the status only for FROM_FAILURE; ALL_STEPS would
        # start a second copy of a running run.
        mock_post = _post_sequence(monkeypatch, _run(status))

        result = reexecute_run("r1", strategy=strategy)

        assert result["status"] == status
        assert "No change was made" in result["message"]
        mock_post.assert_called_once()

    def test_unknown_run(self, monkeypatch):
        _post_sequence(monkeypatch, {"runOrError": {
            "__typename": "RunNotFoundError", "message": "Run r1 could not be found."}})
        assert reexecute_run("r1") == {"run_id": "r1", "message": "Run r1 could not be found."}

    def test_dagster_refusal_raises(self, monkeypatch):
        _post_sequence(monkeypatch, _run("SUCCESS"), {"launchRunReexecution": {
            "__typename": "PythonError",
            "message": "Cannot reexecute from failure a run that is not failed or canceled"}})
        with pytest.raises(RuntimeError, match="not failed or canceled"):
            reexecute_run("r1")
