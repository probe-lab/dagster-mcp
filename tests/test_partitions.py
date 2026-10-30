import json
from unittest.mock import MagicMock

import httpx
import pytest

from dagster_mcp.server import get_asset_partition_status

DAYS = ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04", "2026-01-05"]


def _node(key, statuses, dimensions=None, partitioned=True, description="Daily"):
    if dimensions is None:
        dimensions = [{"name": "default", "partitionKeys": DAYS}]
    return {
        "assetKey": {"path": key.split("/")},
        "isPartitioned": partitioned,
        "partitionDefinition": {"description": description} if partitioned else None,
        "partitionKeysByDimension": dimensions if partitioned else [],
        "assetPartitionStatuses": statuses,
    }


def _time(*ranges):
    return {
        "__typename": "TimePartitionStatuses",
        "ranges": [
            {"startKey": start, "endKey": end, "status": status} for start, end, status in ranges
        ],
    }


def _default(materialized=(), failed=(), materializing=()):
    return {
        "__typename": "DefaultPartitionStatuses",
        "materializedPartitions": list(materialized),
        "failedPartitions": list(failed),
        "materializingPartitions": list(materializing),
    }


def _respond(mock_gql, *nodes):
    return mock_gql({"data": {"assetNodes": list(nodes)}})


class TestTimePartitions:
    def test_expands_ranges_into_keys(self, mock_gql):
        _respond(
            mock_gql,
            _node(
                "raw/orders",
                _time(
                    ("2026-01-01", "2026-01-02", "MATERIALIZED"),
                    ("2026-01-03", "2026-01-04", "FAILED"),
                ),
            ),
        )

        [result] = get_asset_partition_status(["raw/orders"])

        assert result["asset_key"] == "raw/orders"
        assert result["partitioned"] is True
        assert result["partition_definition"] == "Daily"
        assert result["failed"] == ["2026-01-03", "2026-01-04"]
        assert result["materializing"] == []
        assert result["counts"] == {
            "total": 5,
            "failed": 2,
            "materializing": 0,
            "materialized": 2,
            "missing": 1,
        }
        assert "truncated" not in result
        assert "dimensions" not in result
        assert "unresolved_ranges" not in result

    def test_sends_one_request_with_every_asset_key(self, mock_gql):
        mock_post = _respond(mock_gql)

        get_asset_partition_status(["raw/orders", "daily_revenue"])

        mock_post.assert_called_once()
        variables = mock_post.call_args.kwargs["json"]["variables"]
        assert variables == {
            "assetKeys": [{"path": ["raw", "orders"]}, {"path": ["daily_revenue"]}]
        }

    def test_lists_only_requested_statuses(self, mock_gql):
        _respond(
            mock_gql,
            _node("a", _time(("2026-01-01", "2026-01-01", "MATERIALIZED"))),
        )

        [result] = get_asset_partition_status(["a"], statuses=["missing", "Materialized"])

        assert result["missing"] == DAYS[1:]
        assert result["materialized"] == ["2026-01-01"]
        assert "failed" not in result
        assert "materializing" not in result

    def test_marks_lists_cut_by_limit(self, mock_gql):
        _respond(mock_gql, _node("a", _time(("2026-01-01", "2026-01-05", "FAILED"))))

        [result] = get_asset_partition_status(["a"], limit=2)

        assert result["failed"] == ["2026-01-01", "2026-01-02"]
        assert result["counts"]["failed"] == 5
        assert result["truncated"] == ["failed"]

    def test_reports_ranges_it_cannot_expand(self, mock_gql):
        _respond(
            mock_gql,
            _node(
                "a",
                _time(
                    ("2025-12-30", "2026-01-01", "FAILED"),
                    ("2026-01-02", "2026-01-02", "FAILED"),
                ),
            ),
        )

        [result] = get_asset_partition_status(["a"])

        assert result["failed"] == ["2026-01-02"]
        assert result["unresolved_ranges"] == [
            {"status": "failed", "start": "2025-12-30", "end": "2026-01-01"}
        ]


class TestDefaultPartitions:
    def test_static_partitions(self, mock_gql):
        _respond(
            mock_gql,
            _node(
                "regions",
                _default(materialized=["us"], failed=["eu", "apac"]),
                dimensions=[{"name": "default", "partitionKeys": ["us", "eu", "apac", "latam"]}],
                description="'us', 'eu', 'apac', 'latam'",
            ),
        )

        [result] = get_asset_partition_status(["regions"], statuses=["FAILED", "MISSING"])

        # Listed in partition-definition order, not the order Dagster sent.
        assert result["failed"] == ["eu", "apac"]
        assert result["missing"] == ["latam"]
        assert result["counts"]["total"] == 4

    def test_a_retried_failure_counts_only_as_materializing(self, mock_gql):
        # Dagster can list a key as both failed and in progress, and as both
        # materialized and failed. The UI shows one status per partition.
        _respond(
            mock_gql,
            _node(
                "a",
                _default(
                    materialized=["2026-01-01", "2026-01-02"],
                    failed=["2026-01-02", "2026-01-03"],
                    materializing=["2026-01-03"],
                ),
            ),
        )

        [result] = get_asset_partition_status(["a"], statuses=["FAILED", "MATERIALIZING"])

        assert result["materializing"] == ["2026-01-03"]
        assert result["failed"] == ["2026-01-02"]
        assert result["counts"] == {
            "total": 5,
            "failed": 1,
            "materializing": 1,
            "materialized": 1,
            "missing": 2,
        }


class TestMultiPartitions:
    def test_combines_keys_in_dimension_name_order(self, mock_gql):
        # "region" sorts after "date", so keys read "<date>|<region>" even
        # though Dagster lists the dimensions in another order here.
        _respond(
            mock_gql,
            _node(
                "sales",
                {
                    "__typename": "MultiPartitionStatuses",
                    "primaryDimensionName": "date",
                    "ranges": [
                        {
                            "primaryDimStartKey": "2026-01-01",
                            "primaryDimEndKey": "2026-01-02",
                            "secondaryDim": _default(materialized=["us"], failed=["eu"]),
                        },
                        {
                            "primaryDimStartKey": "2026-01-04",
                            "primaryDimEndKey": "2026-01-04",
                            "secondaryDim": _default(materializing=["us"]),
                        },
                    ],
                },
                dimensions=[
                    {"name": "region", "partitionKeys": ["us", "eu"]},
                    {"name": "date", "partitionKeys": DAYS[:4]},
                ],
            ),
        )

        [result] = get_asset_partition_status(["sales"])

        assert result["dimensions"] == ["date", "region"]
        assert result["failed"] == ["2026-01-01|eu", "2026-01-02|eu"]
        assert result["materializing"] == ["2026-01-04|us"]
        assert result["counts"] == {
            "total": 8,
            "failed": 2,
            "materializing": 1,
            "materialized": 2,
            "missing": 3,
        }

    def test_secondary_name_sorting_first_comes_first(self, mock_gql):
        _respond(
            mock_gql,
            _node(
                "sales",
                {
                    "__typename": "MultiPartitionStatuses",
                    "primaryDimensionName": "time",
                    "ranges": [
                        {
                            "primaryDimStartKey": "2026-01-01",
                            "primaryDimEndKey": "2026-01-01",
                            "secondaryDim": _default(failed=["eu"]),
                        }
                    ],
                },
                dimensions=[
                    {"name": "time", "partitionKeys": DAYS[:1]},
                    {"name": "region", "partitionKeys": ["eu"]},
                ],
            ),
        )

        [result] = get_asset_partition_status(["sales"])

        assert result["failed"] == ["eu|2026-01-01"]

    def test_rejects_statuses_that_do_not_match_the_dimensions(self, mock_gql):
        _respond(
            mock_gql,
            _node(
                "sales",
                {"__typename": "MultiPartitionStatuses", "primaryDimensionName": "x", "ranges": []},
                dimensions=[{"name": "date", "partitionKeys": DAYS}],
            ),
        )

        with pytest.raises(RuntimeError, match="do not match"):
            get_asset_partition_status(["sales"])


class TestOtherAssets:
    def test_unpartitioned_and_unknown_assets(self, mock_gql):
        _respond(mock_gql, _node("plain", _default(), partitioned=False))

        results = get_asset_partition_status(["plain", "nope"])

        assert results == [
            {
                "asset_key": "plain",
                "partitioned": False,
                "message": "Asset is not partitioned; use get_asset_health for its status.",
            },
            {"asset_key": "nope", "message": "Asset not found."},
        ]

    def test_asset_defined_in_two_locations_is_reported_once(self, mock_gql):
        node = _node("a", _time(("2026-01-01", "2026-01-01", "FAILED")))
        _respond(mock_gql, node, node)

        results = get_asset_partition_status(["a", "a"])

        assert len(results) == 1
        assert results[0]["failed"] == ["2026-01-01"]

    def test_unknown_status_type_raises(self, mock_gql):
        _respond(mock_gql, _node("a", {"__typename": "SomethingNew"}))

        with pytest.raises(RuntimeError, match="unexpected partition status type"):
            get_asset_partition_status(["a"])


class TestArguments:
    def test_empty_asset_keys(self, mock_gql):
        mock_post = _respond(mock_gql)

        with pytest.raises(ValueError, match="at least one"):
            get_asset_partition_status([])
        mock_post.assert_not_called()

    def test_unknown_status(self, mock_gql):
        mock_post = _respond(mock_gql)

        with pytest.raises(ValueError, match="'stale'"):
            get_asset_partition_status(["a"], statuses=["FAILED", "stale"])
        mock_post.assert_not_called()

    def test_negative_limit(self, mock_gql):
        with pytest.raises(ValueError, match="limit"):
            get_asset_partition_status(["a"], limit=-1)


def _post_sequence(monkeypatch, *responses):
    def reply(data):
        response = MagicMock(status_code=200)
        response.json.return_value = {"data": data}
        response.text = json.dumps({"data": data})
        return response

    mock_post = MagicMock(side_effect=[reply(data) for data in responses])
    monkeypatch.setattr(httpx, "post", mock_post)
    return mock_post


class TestFailedRuns:
    def test_looks_up_the_run_of_each_failed_partition_in_one_request(self, monkeypatch):
        mock_post = _post_sequence(
            monkeypatch,
            {"assetNodes": [
                _node("a", _time(("2026-01-02", "2026-01-03", "FAILED"))),
                _node("b/c", _default(failed=["2026-01-05"])),
            ]},
            {
                "a0": {"__typename": "AssetNode",
                       "a0p0": {"runId": "r1", "status": "FAILURE"},
                       "a0p1": {"runId": "r1", "status": "FAILURE"}},
                "a1": {"__typename": "AssetNode",
                       "a1p0": {"runId": "r2", "status": "CANCELED"}},
            },
        )

        results = get_asset_partition_status(["a", "b/c"])

        assert results[0]["failed_runs"] == [
            {"partition": "2026-01-02", "run_id": "r1", "status": "FAILURE"},
            {"partition": "2026-01-03", "run_id": "r1", "status": "FAILURE"},
        ]
        assert results[1]["failed_runs"] == [
            {"partition": "2026-01-05", "run_id": "r2", "status": "CANCELED"}
        ]
        assert mock_post.call_count == 2
        lookup = mock_post.call_args_list[1].kwargs["json"]
        assert "latestRunForPartition" in lookup["query"]
        assert lookup["variables"] == {
            "k0": {"path": ["a"]},
            "a0p0": "2026-01-02",
            "a0p1": "2026-01-03",
            "k1": {"path": ["b", "c"]},
            "a1p0": "2026-01-05",
        }

    def test_limits_the_lookup_per_asset(self, monkeypatch):
        mock_post = _post_sequence(
            monkeypatch,
            {"assetNodes": [_node("a", _time(("2026-01-01", "2026-01-05", "FAILED")))]},
            {"a0": {"__typename": "AssetNode", "a0p0": None, "a0p1": None}},
        )

        [result] = get_asset_partition_status(["a"], failed_run_limit=2)

        assert len(result["failed"]) == 5
        assert result["failed_runs"] == [
            {"partition": "2026-01-01", "run_id": None, "status": None},
            {"partition": "2026-01-02", "run_id": None, "status": None},
        ]
        assert set(mock_post.call_args.kwargs["json"]["variables"]) == {"k0", "a0p0", "a0p1"}

    def test_total_lookups_are_capped_across_assets(self, monkeypatch):
        many_days = [f"2026-{m:02d}-{d:02d}" for m in (1, 2, 3) for d in range(1, 29)]
        nodes = [
            _node(key, _default(failed=many_days),
                  dimensions=[{"name": "default", "partitionKeys": many_days}])
            for key in ("a", "b")
        ]
        mock_post = _post_sequence(monkeypatch, {"assetNodes": nodes}, {})

        a, b = get_asset_partition_status(["a", "b"], failed_run_limit=80)

        assert len(a["failed_runs"]) == 80
        assert len(b["failed_runs"]) == 20
        variables = mock_post.call_args.kwargs["json"]["variables"]
        assert len([name for name in variables if not name.startswith("k")]) == 100

    def test_lookup_error_keeps_the_partition_statuses(self, monkeypatch):
        def reply(data, status_code=200):
            response = MagicMock(status_code=status_code)
            response.json.return_value = data
            response.text = json.dumps(data)
            return response

        monkeypatch.setattr(httpx, "post", MagicMock(side_effect=[
            reply({"data": {"assetNodes": [
                _node("a", _time(("2026-01-02", "2026-01-02", "FAILED")))]}}),
            httpx.TimeoutException("slow"),
        ]))

        [result] = get_asset_partition_status(["a"])

        assert result["failed"] == ["2026-01-02"]
        assert "failed_runs" not in result
        assert "timed out" in result["failed_runs_error"]

    @pytest.mark.parametrize(
        "kwargs", [{"failed_run_limit": 0}, {"statuses": ["MATERIALIZING"]}]
    )
    def test_no_lookup_when_turned_off_or_failed_not_requested(self, monkeypatch, kwargs):
        mock_post = _post_sequence(
            monkeypatch,
            {"assetNodes": [_node("a", _time(("2026-01-01", "2026-01-01", "FAILED")))]},
        )

        [result] = get_asset_partition_status(["a"], **kwargs)

        assert "failed_runs" not in result
        mock_post.assert_called_once()

    def test_no_lookup_without_failed_partitions(self, monkeypatch):
        mock_post = _post_sequence(
            monkeypatch,
            {"assetNodes": [_node("a", _time(("2026-01-01", "2026-01-01", "MATERIALIZED")))]},
        )

        get_asset_partition_status(["a"])

        mock_post.assert_called_once()
