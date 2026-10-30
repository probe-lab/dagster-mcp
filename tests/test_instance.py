import pytest

from dagster_mcp.server import get_instance_status, list_code_locations


class TestListCodeLocations:
    def test_locations(self, mock_gql):
        mock_gql({"data": {"workspaceOrError": {
            "__typename": "Workspace", "locationEntries": [
            {"name": "loc1", "loadStatus": "LOADED",
             "locationOrLoadError": {
                 "name": "loc1", "repositories": [{"name": "repo1"}]}},
        ]}}})
        result = list_code_locations()
        assert len(result) == 1
        assert result[0]["name"] == "loc1"

    def test_empty(self, mock_gql):
        mock_gql({"data": {"workspaceOrError": {
            "__typename": "Workspace", "locationEntries": [],
        }}})
        assert list_code_locations() == []

    def test_python_error_raises(self, mock_gql):
        mock_post = mock_gql({"data": {"workspaceOrError": {
            "__typename": "PythonError", "message": "workspace load failed",
        }}})
        with pytest.raises(RuntimeError, match="workspace load failed"):
            list_code_locations()
        query = mock_post.call_args.kwargs["json"]["query"]
        assert "__typename" in query
        assert "... on PythonError { message }" in query

    def test_python_error_without_message_uses_fallback(self, mock_gql):
        mock_gql({"data": {"workspaceOrError": {
            "__typename": "PythonError",
        }}})
        with pytest.raises(RuntimeError, match="No error message was provided"):
            list_code_locations()


def _instance_response(daemons=None, locations=None, queued_count=0, pools=None, max_runs=None):
    return {"data": {
        "instance": {
            "daemonHealth": {"allDaemonStatuses": daemons or []},
            "runQueueConfig": {"maxConcurrentRuns": max_runs} if max_runs is not None else None,
            "concurrencyLimits": pools or [],
        },
        "runsOrError": {"count": queued_count},
        "workspaceOrError": {"locationEntries": locations or []},
    }}


class TestGetInstanceStatus:
    def test_healthy(self, mock_gql):
        mock_gql(_instance_response(
            daemons=[
                {"daemonType": "SCHEDULER", "required": True,
                 "healthy": True, "lastHeartbeatTime": "1000"},
                {"daemonType": "SENSOR", "required": True,
                 "healthy": True, "lastHeartbeatTime": "1000"},
            ],
            locations=[
                {"name": "loc1", "loadStatus": "LOADED",
                 "locationOrLoadError": {"name": "loc1"}},
            ],
        ))
        result = get_instance_status()
        assert result["healthy"] is True
        assert len(result["daemons"]) == 2
        assert result["queued_runs_count"] == 0
        assert result["code_location_errors"] == []
        assert result["pools"] == []
        assert result["max_concurrent_runs"] is None

    def test_queued_count_is_the_server_total(self, mock_gql):
        mock_post = mock_gql(_instance_response(queued_count=4237))
        result = get_instance_status()
        assert result["queued_runs_count"] == 4237
        # Only the total is needed, so no run rows are fetched.
        assert "results" not in mock_post.call_args.kwargs["json"]["query"]

    def test_queued_count_error_raises(self, mock_gql):
        response = _instance_response()
        response["data"]["runsOrError"] = {"message": "storage is down"}
        mock_gql(response)
        with pytest.raises(RuntimeError, match="storage is down"):
            get_instance_status()

    def test_pools_and_run_queue_limit(self, mock_gql):
        mock_gql(_instance_response(
            max_runs=10,
            pools=[{"concurrencyKey": "warehouse", "slotCount": 2,
                    "activeSlotCount": 2, "pendingStepCount": 7}],
        ))
        result = get_instance_status()
        assert result["max_concurrent_runs"] == 10
        assert result["pools"] == [
            {"pool": "warehouse", "limit": 2, "in_use": 2, "waiting_steps": 7}
        ]

    def test_unlimited_run_queue_is_reported_as_null(self, mock_gql):
        # Dagster stores "no limit" as -1.
        mock_gql(_instance_response(max_runs=-1))
        assert get_instance_status()["max_concurrent_runs"] is None

    def test_unhealthy_daemon(self, mock_gql):
        mock_gql(_instance_response(
            daemons=[
                {"daemonType": "SCHEDULER", "required": True,
                 "healthy": False, "lastHeartbeatTime": "500"},
            ],
        ))
        result = get_instance_status()
        assert result["healthy"] is False
        assert result["daemons"][0]["healthy"] is False

    def test_code_location_error(self, mock_gql):
        mock_gql(_instance_response(
            daemons=[
                {"daemonType": "SCHEDULER", "required": True,
                 "healthy": True, "lastHeartbeatTime": "1000"},
            ],
            locations=[
                {"name": "broken_loc", "loadStatus": "LOADED",
                 "locationOrLoadError": {"message": "ImportError: no module"}},
            ],
        ))
        result = get_instance_status()
        assert result["healthy"] is False
        assert len(result["code_location_errors"]) == 1
        assert result["code_location_errors"][0]["name"] == "broken_loc"

    def test_queued_runs_do_not_make_the_instance_unhealthy(self, mock_gql):
        mock_gql(_instance_response(
            daemons=[
                {"daemonType": "SCHEDULER", "required": True,
                 "healthy": True, "lastHeartbeatTime": "1000"},
            ],
            queued_count=3,
        ))
        result = get_instance_status()
        assert result["queued_runs_count"] == 3
        assert result["healthy"] is True
