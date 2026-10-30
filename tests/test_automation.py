import json
from unittest.mock import MagicMock

import httpx
import pytest

from dagster_mcp.server import get_asset_automation

SENSOR = {
    "__typename": "Sensor",
    "name": "default_automation_condition_sensor",
    "sensorType": "AUTO_MATERIALIZE",
    "sensorState": {
        "status": "RUNNING",
        "ticks": [{"status": "SKIPPED", "timestamp": 2000.0, "skipReason": None, "error": None}],
    },
}
SCHEDULE = {
    "__typename": "Schedule",
    "name": "nightly",
    "cronSchedule": "0 2 * * *",
    "executionTimezone": "UTC",
    "scheduleState": {"status": "RUNNING"},
}
INSTANCE = {
    "autoMaterializePaused": True,
    "useAutoMaterializeSensors": True,
    "daemonHealth": {
        "allDaemonStatuses": [{"daemonType": "ASSET", "required": True, "healthy": True}]
    },
}


def _node(uid, label, true, candidates=1, children=(), op="identity", user_label=None,
          partitioned=False, since=None):
    return {
        "uniqueId": uid,
        "userLabel": user_label,
        "expandedLabel": [label],
        "numTrue": true,
        "numCandidates": candidates,
        "isPartitioned": partitioned,
        "childUniqueIds": list(children),
        "operatorType": op,
        "sinceMetadata": since,
    }


# eager, cut down: the SINCE condition is false, so the NOT conditions after it
# were not evaluated.
EAGER_NODES = [
    _node("root", "", 0, children=["window", "since", "not_missing"], op="and",
          user_label="eager"),
    _node("window", "in_latest_time_window", 1),
    _node("since", "((newly_missing) OR (any_deps_updated)) SINCE (handled)", 0,
          children=["updated"],
          since={"triggerTimestamp": 1767225600.0, "resetTimestamp": 1767229200.0}),
    _node("updated", "any_deps_updated", 0),
    _node("not_missing", "NOT (any_deps_missing)", 0, candidates=0, children=["missing"],
          op="not"),
    _node("missing", "any_deps_missing", 0, candidates=0),
]


def _record(evaluation_id, requested=0, nodes=EAGER_NODES, run_ids=()):
    return {
        "evaluationId": evaluation_id,
        "timestamp": 1000.0 + int(evaluation_id),
        "numRequested": requested,
        "runIds": list(run_ids),
        "rootUniqueId": "root",
        "evaluationNodes": nodes,
    }


def _respond(monkeypatch, *, condition=True, instigators=(SENSOR,), records=(),
             current="5", instance=INSTANCE, records_error=None):
    node = {
        "__typename": "AssetNode",
        "isPartitioned": False,
        "automationCondition": {"label": "eager", "expandedLabel": ["(a)", "AND", "(b)"]}
        if condition
        else None,
        "currentAutoMaterializeEvaluationId": current,
        "targetingInstigators": list(instigators),
    }
    data = {
        "assetNodeOrError": node,
        "assetConditionEvaluationRecordsOrError": records_error
        or {"__typename": "AssetConditionEvaluationRecords", "records": list(records)},
        "instance": instance,
    }
    response = MagicMock(status_code=200)
    response.json.return_value = {"data": data}
    response.text = json.dumps({"data": data})
    mock_post = MagicMock(return_value=response)
    monkeypatch.setattr(httpx, "post", mock_post)
    return mock_post


class TestGetAssetAutomation:
    def test_false_evaluation_names_the_blocking_condition(self, monkeypatch):
        mock_post = _respond(monkeypatch, records=[_record("3"), _record("2", 1, run_ids=["r1"])])
        result = get_asset_automation("raw/orders", limit=2)

        variables = mock_post.call_args.kwargs["json"]["variables"]
        assert variables == {"assetKey": {"path": ["raw", "orders"]}, "limit": 2}
        assert result["automation_condition"] == "eager"
        assert result["issues"] == []
        latest = result["latest_evaluation"]
        assert latest["evaluation_id"] == "3"
        assert latest["result"] == "FALSE"
        assert latest["blocking_conditions"] == [
            "((newly_missing) OR (any_deps_updated)) SINCE (handled)"
        ]
        assert latest["condition_tree"] == [
            "FALSE  eager",
            "  TRUE  in_latest_time_window",
            "  FALSE  ((newly_missing) OR (any_deps_updated)) SINCE (handled)"
            "  [last trigger 2026-01-01T00:00:00Z, last reset 2026-01-01T01:00:00Z]",
            "    FALSE  any_deps_updated",
            "  SKIPPED  NOT (any_deps_missing)",
            "    SKIPPED  any_deps_missing",
        ]
        assert "evaluation 5" in latest["note"]
        assert result["history"] == [
            {"evaluation_id": "3", "timestamp": 1003.0, "result": "FALSE", "requested": 0,
             "run_ids": []},
            {"evaluation_id": "2", "timestamp": 1002.0, "result": "FALSE", "requested": 1,
             "run_ids": ["r1"]},
        ]

    def test_requested_evaluation_has_no_blocking_conditions(self, monkeypatch):
        nodes = [_node("root", "on_cron", 1, user_label="on_cron")]
        _respond(monkeypatch, records=[_record("5", 1, nodes, ["r9"])])
        latest = get_asset_automation("a")["latest_evaluation"]
        assert latest["result"] == "TRUE"
        assert latest["run_ids"] == ["r9"]
        assert "blocking_conditions" not in latest
        assert "note" not in latest

    def test_false_not_condition_is_named(self, monkeypatch):
        nodes = [
            _node("root", "", 0, children=["a", "not"], op="and", user_label="custom"),
            _node("a", "cron_tick_passed", 1),
            _node("not", "NOT (in_progress)", 0, children=["p"], op="not"),
            _node("p", "in_progress", 1),
        ]
        _respond(monkeypatch, records=[_record("5", 0, nodes)])
        latest = get_asset_automation("a")["latest_evaluation"]
        assert latest["blocking_conditions"] == ["NOT (in_progress)"]

    def test_or_names_the_blockers_of_every_branch(self, monkeypatch):
        nodes = [
            _node("root", "", 0, children=["eager", "cron"], op="or", user_label="eager | cron"),
            _node("eager", "", 0, children=["e1", "e2"], op="and", user_label="eager"),
            _node("e1", "in_latest_time_window", 1),
            _node("e2", "any_deps_updated", 0),
            _node("cron", "cron_tick_passed", 0),
        ]
        _respond(monkeypatch, records=[_record("5", 0, nodes)])
        latest = get_asset_automation("a")["latest_evaluation"]
        assert latest["blocking_conditions"] == ["any_deps_updated", "cron_tick_passed"]

    def test_and_of_partly_true_partition_conditions(self, monkeypatch):
        nodes = [
            _node("root", "", 0, candidates=24, children=["a", "b"], op="and",
                  user_label="all_deps_match", partitioned=True),
            _node("a", "upstream_a updated", 3, candidates=24, partitioned=True),
            _node("b", "upstream_b updated", 24, candidates=24, partitioned=True),
        ]
        _respond(monkeypatch, records=[_record("5", 0, nodes)])
        latest = get_asset_automation("a")["latest_evaluation"]
        assert latest["blocking_conditions"] == [
            "upstream_a updated (TRUE for 3 of 24 partitions)"
        ]

    def test_partitioned_states(self, monkeypatch):
        nodes = [
            _node("root", "", 3, candidates=24, children=["m", "w"], op="and", user_label="eager",
                  partitioned=True),
            _node("m", "missing", 1, candidates=None, partitioned=True),
            _node("w", "in_latest_time_window", 0, candidates=None, partitioned=True),
        ]
        _respond(monkeypatch, records=[_record("5", 3, nodes)])
        latest = get_asset_automation("a")["latest_evaluation"]
        assert latest["result"] == "TRUE for 3 of 24 partitions"
        assert latest["condition_tree"] == [
            "TRUE for 3 of 24 partitions  eager",
            "  TRUE for 1 partition  missing",
            "  FALSE  in_latest_time_window",
        ]

    def test_no_condition(self, monkeypatch):
        _respond(monkeypatch, condition=False, instigators=[SCHEDULE])
        result = get_asset_automation("a")
        assert result["automation_condition"] is None
        assert result["issues"] == [
            "The asset has no automation condition, so automation does not start runs for it."
        ]
        assert result["other_instigators"] == [
            {"kind": "schedule", "name": "nightly", "cron": "0 2 * * *", "timezone": "UTC",
             "status": "RUNNING"}
        ]
        assert "latest_evaluation" not in result

    def test_stopped_sensor_failed_tick_and_unhealthy_daemon(self, monkeypatch):
        sensor = {
            **SENSOR,
            "sensorState": {
                "status": "STOPPED",
                "ticks": [{"status": "FAILURE", "timestamp": 5.0, "skipReason": None,
                           "error": {"message": "boom"}}],
            },
        }
        instance = {
            **INSTANCE,
            "daemonHealth": {
                "allDaemonStatuses": [{"daemonType": "ASSET", "required": True, "healthy": False}]
            },
        }
        _respond(monkeypatch, instigators=[sensor], instance=instance)
        result = get_asset_automation("a")
        assert result["automation_sensors"] == [
            {"name": "default_automation_condition_sensor", "type": "AUTO_MATERIALIZE",
             "status": "STOPPED",
             "last_tick": {"status": "FAILURE", "timestamp": 5.0, "skip_reason": None,
                           "error": "boom"}}
        ]
        assert result["issues"] == [
            "Automation sensor 'default_automation_condition_sensor' is STOPPED, so Dagster "
            "does not evaluate this asset's condition.",
            "The last tick of 'default_automation_condition_sensor' failed: boom",
            "The ASSET daemon, which evaluates this asset's automation, is not healthy. "
            "Check get_instance_status.",
        ]
        assert result["latest_evaluation"] is None
        # The sensor ticked, so a missing record means no change for 30 days.
        assert "last 30 days" in result["note"]

    def test_never_evaluated(self, monkeypatch):
        sensor = {**SENSOR, "sensorState": {"status": "RUNNING", "ticks": []}}
        _respond(monkeypatch, instigators=[sensor], current=None)
        result = get_asset_automation("a")
        assert result["issues"] == ["Dagster has not evaluated this asset yet."]
        assert "note" not in result

    def test_user_code_automation_sensor_needs_the_sensor_daemon(self, monkeypatch):
        sensor = {**SENSOR, "name": "my_automation", "sensorType": "AUTOMATION"}
        instance = {
            **INSTANCE,
            "daemonHealth": {
                "allDaemonStatuses": [
                    {"daemonType": "ASSET", "required": True, "healthy": False},
                    {"daemonType": "SENSOR", "required": True, "healthy": False},
                ]
            },
        }
        _respond(monkeypatch, instigators=[sensor], instance=instance, records=[_record("5")],
                 current=None)
        assert get_asset_automation("a")["issues"] == [
            "The SENSOR daemon, which evaluates this asset's automation, is not healthy. "
            "Check get_instance_status."
        ]

    def test_no_automation_sensor(self, monkeypatch):
        other = {**SENSOR, "name": "on_upstream", "sensorType": "STANDARD"}
        _respond(monkeypatch, instigators=[other], records=[_record("5")])
        result = get_asset_automation("a")
        assert result["issues"] == ["No automation sensor targets this asset."]
        assert result["other_instigators"] == [
            {"kind": "sensor", "name": "on_upstream", "type": "STANDARD", "status": "RUNNING"}
        ]

    def test_paused_daemon_without_sensors(self, monkeypatch):
        instance = {**INSTANCE, "useAutoMaterializeSensors": False}
        _respond(monkeypatch, instigators=[], instance=instance, records=[_record("5")])
        assert get_asset_automation("a")["issues"] == [
            "Auto-materialization is paused on this instance."
        ]

    def test_records_error(self, monkeypatch):
        error = {"__typename": "AutoMaterializeAssetEvaluationNeedsMigrationError",
                 "message": "Run dagster instance migrate."}
        _respond(monkeypatch, records_error=error)
        assert get_asset_automation("a")["issues"] == ["Run dagster instance migrate."]

    def test_unknown_asset_raises(self, monkeypatch):
        response = MagicMock(status_code=200)
        response.json.return_value = {"data": {
            "assetNodeOrError": {"__typename": "AssetNotFoundError",
                                 "message": 'Asset key ["x"] not found.'},
        }}
        monkeypatch.setattr(httpx, "post", MagicMock(return_value=response))
        with pytest.raises(RuntimeError, match="not found"):
            get_asset_automation("x")

    @pytest.mark.parametrize("limit", [0, 51])
    def test_limit_is_checked(self, limit):
        with pytest.raises(ValueError, match="limit"):
            get_asset_automation("a", limit=limit)
