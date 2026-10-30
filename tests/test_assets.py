import json
from unittest.mock import MagicMock

import httpx
import pytest

from dagster_mcp.server import (
    get_asset_details,
    get_asset_health,
    get_recent_materializations,
    search_assets,
)


class TestSearchAssets:
    def test_no_filters(self, mock_gql):
        mock_gql({"data": {"assetNodes": [
            {"assetKey": {"path": ["my_asset"]}, "groupName": "default",
             "description": "desc", "isPartitioned": False, "op": {"name": "my_op"}},
        ]}})
        result = search_assets()
        assert len(result) == 1
        assert result[0]["assetKey"]["path"] == ["my_asset"]

    def test_prefix_filter(self, mock_gql):
        mock_gql({"data": {"assetNodes": [
            {"assetKey": {"path": ["orders_raw"]}, "groupName": "g1",
             "description": "", "isPartitioned": False, "op": None},
            {"assetKey": {"path": ["users_raw"]}, "groupName": "g1",
             "description": "", "isPartitioned": False, "op": None},
        ]}})
        result = search_assets(prefix="orders")
        assert len(result) == 1
        assert result[0]["assetKey"]["path"] == ["orders_raw"]

    def test_group_filter(self, mock_gql):
        mock_gql({"data": {"assetNodes": [
            {"assetKey": {"path": ["a1"]}, "groupName": "analytics",
             "description": "", "isPartitioned": False, "op": None},
            {"assetKey": {"path": ["a2"]}, "groupName": "ingestion",
             "description": "", "isPartitioned": False, "op": None},
        ]}})
        result = search_assets(group="analytics")
        assert len(result) == 1
        assert result[0]["groupName"] == "analytics"

    def test_empty_results(self, mock_gql):
        mock_gql({"data": {"assetNodes": []}})
        assert search_assets() == []


class TestGetAssetDetails:
    def test_single_asset(self, mock_gql):
        mock_gql({"data": {"assetNodes": [
            {"assetKey": {"path": ["my_asset"]}, "description": "A table",
             "groupName": "default", "op": {"name": "my_op"},
             "isObservable": False, "isPartitioned": True,
             "partitionDefinition": {"description": "daily"},
             "dependencyKeys": [{"path": ["upstream"]}],
             "dependedByKeys": [],
             "assetMaterializations": [{"runId": "r1", "timestamp": "1000"}]},
        ]}})
        result = get_asset_details(["my_asset"])
        assert len(result) == 1
        assert result[0]["isPartitioned"] is True

    def test_freshness_policy_is_normalized(self, mock_gql):
        mock_gql({"data": {"assetNodes": [
            {"assetKey": {"path": ["a"]},
             "freshnessPolicy": None,
             "internalFreshnessPolicy": {
                 "__typename": "CronFreshnessPolicy", "deadlineCron": "0 9 * * *",
                 "lowerBoundDeltaSeconds": 3600, "timezone": "UTC"}},
            {"assetKey": {"path": ["b"]},
             "freshnessPolicy": None, "internalFreshnessPolicy": None},
        ]}})
        a, b = get_asset_details(["a", "b"])
        assert a["freshness_policy"] == {
            "type": "cron", "deadline_cron": "0 9 * * *",
            "lower_bound_delta_seconds": 3600, "timezone": "UTC",
        }
        assert "internalFreshnessPolicy" not in a
        assert "freshnessPolicy" not in a
        assert b["freshness_policy"] is None

    def test_multiple_assets(self, mock_gql):
        mock_gql({"data": {"assetNodes": [
            {"assetKey": {"path": ["a1"]}, "description": "", "groupName": "g",
             "op": None, "isObservable": False, "isPartitioned": False,
             "partitionDefinition": None, "dependencyKeys": [],
             "dependedByKeys": [], "assetMaterializations": []},
            {"assetKey": {"path": ["a2"]}, "description": "", "groupName": "g",
             "op": None, "isObservable": False, "isPartitioned": False,
             "partitionDefinition": None, "dependencyKeys": [],
             "dependedByKeys": [], "assetMaterializations": []},
        ]}})
        result = get_asset_details(["a1", "a2"])
        assert len(result) == 2


class TestGetRecentMaterializations:
    def test_materializations(self, mock_gql):
        mock_gql({"data": {"assetOrError": {"__typename": "Asset", "assetMaterializations": [
            {"runId": "r1", "timestamp": "1000", "partition": "2026-01-03",
             "stepKey": "my_asset",
             "assetKey": {"path": ["my_asset"]},
             "metadataEntries": [
                 {"__typename": "IntMetadataEntry", "label": "rows",
                  "description": None, "intValue": 42},
             ]},
        ]}}})
        result = get_recent_materializations("my_asset")
        assert len(result) == 1
        assert result[0]["runId"] == "r1"
        assert result[0]["partition"] == "2026-01-03"
        assert result[0]["metadataEntries"] == [
            {"label": "rows", "description": None, "value": 42}
        ]

    def test_empty(self, mock_gql):
        mock_gql({"data": {"assetOrError": {"assetMaterializations": []}}})
        assert get_recent_materializations("missing") == []

    def test_unknown_asset_raises(self, mock_gql):
        mock_gql({"data": {"assetOrError": {
            "__typename": "AssetNotFoundError", "message": "not found"}}})
        with pytest.raises(RuntimeError, match="'missing' not found"):
            get_recent_materializations("missing")

    def test_limit_partitions_and_nested_key(self, mock_gql):
        mock_post = mock_gql({"data": {"assetOrError": {"assetMaterializations": []}}})
        get_recent_materializations("raw/orders", limit=3, partitions=["2026-01-03"])
        variables = mock_post.call_args.kwargs["json"]["variables"]
        assert variables == {
            "assetKey": {"path": ["raw", "orders"]},
            "limit": 3,
            "partitions": ["2026-01-03"],
        }


def _make_mock_post(responses):
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


def _health_node(key, **fields):
    node = {
        "assetKey": {"path": key.split("/")},
        "groupName": "g",
        "isMaterializable": True,
        "isPartitioned": False,
        "partitionStats": None,
        "freshnessPolicy": None,
        "internalFreshnessPolicy": None,
        "freshnessStatusInfo": None,
        "staleCauses": [],
        "automationCondition": None,
        "assetChecksOrError": {"__typename": "AssetChecks", "checks": []},
    }
    node.update(fields)
    return node


def _latest(key, run=None, materialization=None, in_progress=(), queued=()):
    return {
        "assetKey": {"path": key.split("/")},
        "latestRun": run,
        "latestMaterialization": materialization,
        "inProgressRunIds": list(in_progress),
        "unstartedRunIds": list(queued),
    }


def _health_responses(nodes, latest, health=None, groups=None):
    """The group lookup, then the single health query."""
    return [
        {"data": {"assetNodes": groups or [
            {"assetKey": n["assetKey"], "groupName": "other"} for n in nodes
        ]}},
        {"data": {
            "assetNodes": nodes,
            "assetsLatestInfo": latest,
            "assetsOrError": {"__typename": "AssetConnection", "nodes": health or []},
        }},
    ]


class TestGetAssetHealth:
    def test_healthy_asset(self, monkeypatch):
        monkeypatch.setattr(httpx, "post", _make_mock_post(_health_responses(
            [_health_node("my_asset")],
            [_latest(
                "my_asset",
                run={"runId": "r1", "status": "SUCCESS", "endTime": 1000.0},
                materialization={"runId": "r1", "timestamp": "1000000", "partition": None},
            )],
        )))
        [result] = get_asset_health("my_asset")
        assert result["asset_key"] == "my_asset"
        assert result["issues"] == []
        assert result["latest_run"] == {"run_id": "r1", "status": "SUCCESS", "end_time": 1000.0}
        assert result["latest_run_status"] == "SUCCESS"
        assert result["last_materialization"] == {
            "run_id": "r1", "timestamp": 1000.0, "partition": None,
        }
        assert result["stale"] is False

    def test_failed_latest_run_is_reported_although_an_older_run_materialized(
        self, monkeypatch
    ):
        # The old tool looked up the run of the last materialization, which
        # succeeded, and so hid that the newest run failed.
        monkeypatch.setattr(httpx, "post", _make_mock_post(_health_responses(
            [_health_node("my_asset")],
            [_latest(
                "my_asset",
                run={"runId": "r2", "status": "FAILURE", "endTime": 2000.0},
                materialization={"runId": "r1", "timestamp": "1000000", "partition": None},
            )],
        )))
        [result] = get_asset_health("my_asset")
        assert result["latest_run_status"] == "FAILURE"
        assert result["last_materialization"]["run_id"] == "r1"
        assert result["issues"] == [
            "Latest run r2 ended with FAILURE; use get_run_failure_summary."
        ]

    def test_run_that_failed_after_materializing_the_asset_is_not_an_issue(
        self, monkeypatch
    ):
        # The asset's own step succeeded; a later step of the same run failed.
        monkeypatch.setattr(httpx, "post", _make_mock_post(_health_responses(
            [_health_node("my_asset")],
            [_latest(
                "my_asset",
                run={"runId": "r1", "status": "FAILURE", "endTime": 2000.0},
                materialization={"runId": "r1", "timestamp": "1500000", "partition": None},
            )],
        )))
        [result] = get_asset_health("my_asset")
        assert result["latest_run_status"] == "FAILURE"
        assert result["issues"] == []

    def test_partitions_checks_freshness_and_automation(self, monkeypatch):
        node = _health_node(
            "raw/orders",
            isPartitioned=True,
            partitionStats={"numPartitions": 10, "numMaterialized": 6,
                            "numFailed": 3, "numMaterializing": 1},
            internalFreshnessPolicy={"__typename": "TimeWindowFreshnessPolicy",
                                     "failWindowSeconds": 3600, "warnWindowSeconds": None},
            freshnessStatusInfo={"freshnessStatus": "DEGRADED"},
            staleCauses=[{"reason": "updated dependencies", "dependency": {"path": ["up"]}}],
            automationCondition={"label": "eager"},
            assetChecksOrError={"__typename": "AssetChecks", "checks": [
                {"name": "ok", "executionForLatestMaterialization": {
                    "status": "SUCCEEDED", "evaluation": {"severity": "ERROR"}}},
                {"name": "no_nulls", "executionForLatestMaterialization": {
                    "status": "FAILED", "evaluation": {"severity": "ERROR"}}},
                {"name": "row_count", "executionForLatestMaterialization": {
                    "status": "FAILED", "evaluation": {"severity": "WARN"}}},
                {"name": "broken", "executionForLatestMaterialization": {
                    "status": "EXECUTION_FAILED", "evaluation": None}},
                {"name": "new", "executionForLatestMaterialization": None},
            ]},
        )
        monkeypatch.setattr(httpx, "post", _make_mock_post(_health_responses(
            [node],
            [_latest("raw/orders", in_progress=["r9"], queued=["r10"])],
        )))
        [result] = get_asset_health("raw/orders")
        assert result["partitions"] == {
            "total": 10, "materialized": 6, "failed": 3, "materializing": 1,
        }
        assert result["checks"] == {
            "total": 5,
            "failed": ["no_nulls"],
            "warning": ["row_count"],
            "execution_failed": ["broken"],
            "not_run": ["new"],
        }
        assert result["freshness_policy"] == {
            "type": "time_window", "fail_window_seconds": 3600, "warn_window_seconds": None,
        }
        assert result["freshness_status"] == "DEGRADED"
        assert result["automation_condition"] == "eager"
        assert result["runs_in_progress"] == ["r9"]
        assert result["runs_queued"] == ["r10"]
        assert result["stale_causes"] == ["updated dependencies"]
        assert result["issues"] == [
            "3 failed partition(s); use get_asset_partition_status.",
            "Check 'no_nulls' failed (ERROR severity).",
            "Check 'row_count' failed (WARN severity).",
            "Check 'broken' could not execute.",
            "Freshness status is DEGRADED.",
        ]

    def test_legacy_freshness_policy(self, monkeypatch):
        node = _health_node("a", freshnessPolicy={
            "maximumLagMinutes": 60, "cronSchedule": "0 * * * *", "cronScheduleTimezone": None})
        monkeypatch.setattr(httpx, "post", _make_mock_post(
            _health_responses([node], [_latest("a")])))
        [result] = get_asset_health("a")
        assert result["freshness_policy"] == {
            "type": "legacy", "max_lag_minutes": 60, "cron": "0 * * * *", "cron_timezone": None,
        }

    def test_dagster_plus_health_is_passed_through(self, monkeypatch):
        health = {"assetHealth": "DEGRADED", "materializationStatus": "DEGRADED",
                  "assetChecksStatus": "HEALTHY", "freshnessStatus": "NOT_APPLICABLE"}
        monkeypatch.setattr(httpx, "post", _make_mock_post(_health_responses(
            [_health_node("a")],
            [_latest("a", run={"runId": "r", "status": "SUCCESS", "endTime": 1.0})],
            health=[{"key": {"path": ["a"]}, "assetHealth": health}],
        )))
        [result] = get_asset_health("a")
        assert result["dagster_health"] == health
        assert result["issues"] == ["Dagster reports the asset as DEGRADED."]

    def test_never_materialized(self, monkeypatch):
        monkeypatch.setattr(httpx, "post", _make_mock_post(_health_responses(
            [_health_node("a"), _health_node("external", isMaterializable=False)],
            [_latest("a"), _latest("external")],
            groups=[
                {"assetKey": {"path": ["a"]}, "groupName": "g"},
                {"assetKey": {"path": ["external"]}, "groupName": "g"},
            ],
        )))
        results = get_asset_health("g")
        assert [r["issues"] for r in results] == [["Asset has never been materialized."], []]

    def test_group_query_sends_every_member(self, monkeypatch):
        mock_post = MagicMock(side_effect=[
            MagicMock(status_code=200, json=MagicMock(return_value=r))
            for r in _health_responses(
                [_health_node("a1"), _health_node("a2")],
                [],
                groups=[
                    {"assetKey": {"path": ["a1"]}, "groupName": "analytics"},
                    {"assetKey": {"path": ["a2"]}, "groupName": "Analytics"},
                    {"assetKey": {"path": ["b1"]}, "groupName": "other"},
                ],
            )
        ])
        monkeypatch.setattr(httpx, "post", mock_post)
        result = get_asset_health("analytics")
        assert [r["asset_key"] for r in result] == ["a1", "a2"]
        variables = mock_post.call_args_list[1].kwargs["json"]["variables"]
        assert variables == {"assetKeys": [{"path": ["a1"]}, {"path": ["a2"]}]}

    def test_not_found(self, monkeypatch):
        monkeypatch.setattr(httpx, "post", _make_mock_post([
            {"data": {"assetNodes": []}},
            {"data": {"assetNodes": [], "assetsLatestInfo": [], "assetsOrError": {}}},
        ]))
        result = get_asset_health("nonexistent")
        assert "not found" in result[0].get("message", "").lower()
