import json
from unittest.mock import MagicMock

import httpx
import pytest

from dagster_mcp.server import get_asset_check_results


def _post_sequence(monkeypatch, *responses):
    def reply(data):
        response = MagicMock(status_code=200)
        response.json.return_value = {"data": data}
        response.text = json.dumps({"data": data})
        return response

    mock_post = MagicMock(side_effect=[reply(data) for data in responses])
    monkeypatch.setattr(httpx, "post", mock_post)
    return mock_post


def _asset(key, checks=None, error=None):
    if error is not None:
        checks_or_error = error
    else:
        checks_or_error = {"__typename": "AssetChecks", "checks": checks or []}
    return {"assetKey": {"path": key.split("/")}, "assetChecksOrError": checks_or_error}


def _check(name, blocking=False):
    return {"name": name, "description": f"{name} description", "blocking": blocking}


FAILED = {
    "runId": "r2",
    "status": "FAILED",
    "timestamp": 2000.0,
    "partition": None,
    "evaluation": {
        "severity": "ERROR",
        "description": "3 null rows",
        "metadataEntries": [
            {"__typename": "IntMetadataEntry", "label": "nulls", "description": None,
             "intValue": 3},
        ],
    },
}
PASSED = {"runId": "r1", "status": "SUCCEEDED", "timestamp": 1000.0, "partition": None,
          "evaluation": {"severity": "ERROR", "description": None, "metadataEntries": []}}
CRASHED = {"runId": "r3", "status": "EXECUTION_FAILED", "timestamp": 3000.0,
           "partition": None, "evaluation": None}


class TestGetAssetCheckResults:
    def test_latest_result_of_every_check(self, monkeypatch):
        mock_post = _post_sequence(
            monkeypatch,
            {"assetNodes": [_asset("raw/orders", [_check("no_nulls", True), _check("fresh"),
                                                  _check("new")])]},
            {"e0": [FAILED], "e1": [CRASHED], "e2": []},
        )

        [result] = get_asset_check_results(["raw/orders"])

        assert result["asset_key"] == "raw/orders"
        no_nulls, fresh, new = result["checks"]
        assert no_nulls == {
            "check_name": "no_nulls",
            "description": "no_nulls description",
            "blocking": True,
            "status": "FAILED",
            "executions": [{
                "run_id": "r2",
                "status": "FAILED",
                "timestamp": 2000.0,
                "partition": None,
                "severity": "ERROR",
                "description": "3 null rows",
                "metadata": [{"label": "nulls", "description": None, "value": 3}],
            }],
        }
        assert fresh["status"] == "EXECUTION_FAILED"
        assert fresh["executions"] == [
            {"run_id": "r3", "status": "EXECUTION_FAILED", "timestamp": 3000.0,
             "partition": None}
        ]
        assert new["status"] == "NOT_RUN"
        assert new["executions"] == []

        lookup = mock_post.call_args.kwargs["json"]
        assert lookup["variables"] == {
            "limit": 1,
            "k0": {"path": ["raw", "orders"]}, "c0": "no_nulls",
            "k1": {"path": ["raw", "orders"]}, "c1": "fresh",
            "k2": {"path": ["raw", "orders"]}, "c2": "new",
        }

    def test_history_of_one_check(self, monkeypatch):
        mock_post = _post_sequence(
            monkeypatch,
            {"assetNodes": [_asset("a", [_check("no_nulls"), _check("other")])]},
            {"e0": [FAILED, PASSED]},
        )

        [result] = get_asset_check_results(["a"], check_name="no_nulls", limit=5)

        [check] = result["checks"]
        assert [e["status"] for e in check["executions"]] == ["FAILED", "SUCCEEDED"]
        assert mock_post.call_args.kwargs["json"]["variables"] == {
            "limit": 5, "k0": {"path": ["a"]}, "c0": "no_nulls",
        }

    def test_unknown_check_name(self, monkeypatch):
        mock_post = _post_sequence(monkeypatch, {"assetNodes": [_asset("a", [_check("x")])]})

        [result] = get_asset_check_results(["a"], check_name="nope")

        assert result == {
            "asset_key": "a",
            "checks": [],
            "message": "Asset has no check named 'nope'.",
        }
        mock_post.assert_called_once()

    def test_assets_without_checks_unknown_or_unreadable(self, monkeypatch):
        mock_post = _post_sequence(
            monkeypatch,
            {"assetNodes": [
                _asset("plain"),
                _asset("old", error={"__typename": "AssetCheckNeedsMigrationError",
                                     "message": "run dagster instance migrate"}),
            ]},
        )

        results = get_asset_check_results(["plain", "old", "nope"])

        assert results == [
            {"asset_key": "plain", "checks": []},
            {"asset_key": "old", "message": "run dagster instance migrate"},
            {"asset_key": "nope", "message": "Asset not found."},
        ]
        mock_post.assert_called_once()

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [({"asset_keys": []}, "at least one"), ({"asset_keys": ["a"], "limit": 0}, "limit")],
    )
    def test_bad_arguments(self, mock_gql, kwargs, match):
        mock_post = mock_gql({"data": {}})
        with pytest.raises(ValueError, match=match):
            get_asset_check_results(**kwargs)
        mock_post.assert_not_called()
