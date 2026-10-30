"""Dagster MCP server — GraphQL wrapper for self-hosted and Dagster Cloud instances."""

import bisect
import json
import os
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Any, Final, TypedDict
from urllib.parse import urljoin, urlparse

import httpx
from fastmcp import FastMCP

from dagster_mcp.asset_selection import (
    AssetSelectionSyntaxError,
    evaluate_asset_selection,
    parse_asset_selection,
)

DAGSTER_URL = os.environ.get("DAGSTER_URL", "http://localhost:3000")
DAGSTER_API_TOKEN = os.environ.get("DAGSTER_API_TOKEN", "")
DAGSTER_EXTRA_HEADERS = os.environ.get("DAGSTER_EXTRA_HEADERS", "")
READ_ONLY = os.environ.get("DAGSTER_READ_ONLY", "true").lower() in ("true", "1", "yes")

# Multi-env support
_DAGSTER_ENVS_RAW = os.environ.get("DAGSTER_ENVS", "")
_DAGSTER_DEFAULT_ENV = os.environ.get("DAGSTER_DEFAULT_ENV", "")


def _parse_dagster_envs(raw: str) -> dict[str, dict]:
    if not raw:
        return {}
    try:
        envs = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            "DAGSTER_ENVS must be a valid JSON object "
            '(example: \'{"prod": {"url": "https://prod.dagster.io", "token": "..."}, '
            '"dev": {"url": "http://localhost:3000"}}\').'
        ) from exc
    if not isinstance(envs, dict):
        raise RuntimeError("DAGSTER_ENVS must be a JSON object mapping env names to configs.")
    return envs


_ENVS: dict[str, dict] = _parse_dagster_envs(_DAGSTER_ENVS_RAW)

_mode = "read-only" if READ_ONLY else "read-write"
_env_info = (
    f"Available environments: {', '.join(_ENVS)}. Pass env=<name> to each tool. " if _ENVS else ""
)
mcp = FastMCP(
    "dagster",
    instructions=(
        f"Use these tools to monitor and operate a running Dagster 1.13.6+ instance ({_mode} mode). "
        f"{_env_info}"
        "Start with get_instance_status, get_runs, or get_asset_health to find problems, then "
        "drill into specific runs and their step logs, assets, partitions, automation, "
        "backfills, schedules, or sensors."
    ),
)


def _resolve_connection(env: str | None) -> tuple[str, str, str]:
    """Return (graphql_url, api_token, extra_headers_json) for the given env.

    In single-env mode (DAGSTER_ENVS not set), env is ignored and module-level
    DAGSTER_URL / DAGSTER_API_TOKEN / DAGSTER_EXTRA_HEADERS are used.
    """
    if not _ENVS:
        return (
            f"{DAGSTER_URL.rstrip('/')}/graphql",
            DAGSTER_API_TOKEN,
            DAGSTER_EXTRA_HEADERS,
        )

    name = env or _DAGSTER_DEFAULT_ENV
    if not name:
        if len(_ENVS) == 1:
            name = next(iter(_ENVS))
        else:
            raise RuntimeError(
                f"Multiple Dagster envs configured but no env specified. "
                f"Available: {', '.join(_ENVS)}. "
                "Pass env=<name> to the tool or set DAGSTER_DEFAULT_ENV."
            )

    if name not in _ENVS:
        raise RuntimeError(f"Unknown Dagster env '{name}'. Available: {', '.join(_ENVS)}.")

    cfg = _ENVS[name]
    url = cfg.get("url", "http://localhost:3000")
    token = cfg.get("token", "")
    extra = cfg.get("extra_headers", "")
    return f"{url.rstrip('/')}/graphql", token, extra


def _build_headers(
    api_token: str | None = None,
    extra_headers_json: str | None = None,
) -> dict[str, str]:
    if api_token is None:
        api_token = DAGSTER_API_TOKEN
    if extra_headers_json is None:
        extra_headers_json = DAGSTER_EXTRA_HEADERS

    headers: dict[str, str] = {}
    if api_token:
        headers["Dagster-Cloud-Api-Token"] = api_token
    if extra_headers_json:
        try:
            extra_headers = json.loads(extra_headers_json)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                "DAGSTER_EXTRA_HEADERS must be a valid JSON object "
                '(example: \'{"Authorization":"Bearer token"}\').'
            ) from exc

        if not isinstance(extra_headers, dict):
            raise RuntimeError("DAGSTER_EXTRA_HEADERS must be a JSON object.")

        invalid_pairs = [
            (key, value)
            for key, value in extra_headers.items()
            if not isinstance(key, str) or not isinstance(value, str)
        ]
        if invalid_pairs:
            raise RuntimeError("DAGSTER_EXTRA_HEADERS keys and values must be strings.")

        headers.update(extra_headers)
    return headers


def gql(query: str, variables: dict | None = None, env: str | None = None) -> dict:
    graphql_url, api_token, extra_headers_json = _resolve_connection(env)
    headers = _build_headers(api_token, extra_headers_json)
    try:
        response = httpx.post(
            graphql_url,
            json={"query": query, "variables": variables or {}},
            headers=headers,
            timeout=30,
        )
    except httpx.ConnectError as exc:
        base_url = graphql_url.removesuffix("/graphql")
        raise RuntimeError(
            f"Cannot connect to Dagster at {base_url}. "
            "Check that DAGSTER_URL is correct and the instance is running."
        ) from exc
    except httpx.TimeoutException as exc:
        base_url = graphql_url.removesuffix("/graphql")
        raise RuntimeError(
            f"Request to Dagster at {base_url} timed out after 30s."
        ) from exc
    if 300 <= response.status_code < 400:
        location = response.headers.get("location")
        destination = "unknown"
        if location:
            parsed = urlparse(urljoin(graphql_url, location))
            # hostname omits userinfo (unlike netloc) so credentials in Location
            # never reach the message shown to LLM clients or logs.
            host = parsed.hostname
            if host:
                destination = f"{host}:{parsed.port}" if parsed.port else host
        raise RuntimeError(
            f"Dagster returned HTTP {response.status_code} redirect to {destination}. "
            "Check DAGSTER_URL and refresh the configured authentication token or headers."
        )
    if response.status_code >= 400:
        raise RuntimeError(f"Dagster returned HTTP {response.status_code}: {response.text[:500]}")
    try:
        data = response.json()
    except ValueError as exc:
        raise RuntimeError(
            f"Dagster returned HTTP {response.status_code} with a non-JSON response "
            f"(content type: {response.headers.get('content-type', 'unknown')}). "
            "Check DAGSTER_URL and authentication."
        ) from exc
    if not isinstance(data, dict):
        raise RuntimeError("Dagster returned an invalid GraphQL response: expected a JSON object.")
    if "errors" in data:
        messages = [e.get("message", str(e)) for e in data["errors"]]
        raise RuntimeError("Dagster GraphQL error: " + "; ".join(messages))
    if "data" not in data:
        raise RuntimeError("Dagster returned an invalid GraphQL response: missing data field.")
    return data["data"]


# GraphQL selection for MetadataEntry — covers the common concrete types.
# Reused wherever a log event exposes metadataEntries.
_METADATA_ENTRIES_FRAGMENT = """
              metadataEntries {
                __typename
                label
                description
                ... on TextMetadataEntry { text }
                ... on UrlMetadataEntry { url }
                ... on PathMetadataEntry { path }
                ... on JsonMetadataEntry { jsonString }
                ... on MarkdownMetadataEntry { mdStr }
                ... on FloatMetadataEntry { floatValue }
                ... on IntMetadataEntry { intValue }
                ... on BoolMetadataEntry { boolValue }
                ... on PythonArtifactMetadataEntry { module name }
              }"""

# Maps each MetadataEntry __typename to the field holding its value.
_METADATA_VALUE_FIELDS = {
    "TextMetadataEntry": "text",
    "UrlMetadataEntry": "url",
    "PathMetadataEntry": "path",
    "JsonMetadataEntry": "jsonString",
    "MarkdownMetadataEntry": "mdStr",
    "FloatMetadataEntry": "floatValue",
    "IntMetadataEntry": "intValue",
    "BoolMetadataEntry": "boolValue",
}


# Dagster log levels, ordered. level_filter selects this level and above.
_LOG_LEVEL_ORDER: dict[str, int] = {
    "DEBUG": 0,
    "INFO": 1,
    "WARNING": 2,
    "ERROR": 3,
    "CRITICAL": 4,
}


def _flatten_metadata(entries: list[dict] | None) -> list[dict]:
    """Flatten raw metadataEntries into {label, description, value} dicts.

    Value is pulled from the type-specific field; PythonArtifact combines
    module + name; unknown types fall back to None.
    """
    flat = []
    for e in entries or []:
        typename = e.get("__typename")
        if typename == "PythonArtifactMetadataEntry":
            value = f"{e.get('module')}.{e.get('name')}"
        else:
            value = e.get(_METADATA_VALUE_FIELDS.get(typename, ""))
        flat.append(
            {"label": e.get("label"), "description": e.get("description"), "value": value}
        )
    return flat


# ── Runs ──────────────────────────────────────────────────────────────────────


def _timestamp_input(value: str | float | None, name: str) -> float | None:
    """Convert an ISO 8601 string or Unix seconds into Unix seconds for GraphQL.

    A naive ISO string is read as UTC, which is how Dagster shows times.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an ISO 8601 string or Unix seconds, not a boolean.")
    if isinstance(value, int | float):
        return float(value)
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (AttributeError, ValueError) as exc:
        raise ValueError(
            f"{name} must be an ISO 8601 string (e.g. '2026-01-31T12:00:00Z') "
            f"or Unix seconds; got {value!r}."
        ) from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.timestamp()


def _tag_inputs(tags: Mapping[str, str] | None) -> list[dict[str, str]]:
    return [{"key": key, "value": value} for key, value in (tags or {}).items()]


def _tag_value(tags: Sequence[Mapping[str, Any]] | None, key: str) -> str | None:
    for tag in tags or []:
        if tag.get("key") == key:
            return tag.get("value")
    return None


# Tags Dagster puts on runs it launches for partitions and backfills.
_PARTITION_TAG: Final[str] = "dagster/partition"
_PARTITION_RANGE_START_TAG: Final[str] = "dagster/asset_partition_range_start"
_PARTITION_RANGE_END_TAG: Final[str] = "dagster/asset_partition_range_end"
_BACKFILL_TAG: Final[str] = "dagster/backfill"


def _run_partition(run: Mapping[str, Any]) -> dict[str, str] | None:
    """Return the partition (or partition range) a run targeted, from its tags."""
    tags = run.get("tags")
    partition = _tag_value(tags, _PARTITION_TAG)
    if partition is not None:
        return {"partition": partition}
    start = _tag_value(tags, _PARTITION_RANGE_START_TAG)
    end = _tag_value(tags, _PARTITION_RANGE_END_TAG)
    if start is not None and end is not None:
        return {"partition_range_start": start, "partition_range_end": end}
    return None


@mcp.tool()
def get_runs(
    job_name: str | None = None,
    statuses: list[str] | None = None,
    limit: int = 10,
    env: str | None = None,
    tags: dict[str, str] | None = None,
    run_ids: list[str] | None = None,
    created_after: str | float | None = None,
    created_before: str | float | None = None,
    updated_after: str | float | None = None,
    cursor: str | None = None,
) -> list[dict]:
    """List runs, newest first. Start here to discover what has been running.

    Returns runId, status, jobName, creationTime, startTime, endTime,
    parentRunId, rootRunId, and tags for each run. Times are Unix seconds.
    Use the returned runId to drill into details with get_run_status,
    get_run_logs, get_run_stats, or get_run_failure_summary.

    Filters (combined with AND):
    - job_name: exact job name (e.g. 'my_etl_job')
    - statuses: one or more of 'SUCCESS', 'FAILURE', 'CANCELED', 'STARTED',
      'QUEUED', 'STARTING', 'CANCELING', 'NOT_STARTED'
    - tags: run tags that must all match, e.g.
      {'dagster/partition': '2026-01-03'} for the runs of one partition, or
      {'dagster/backfill': '<backfill id>'} for the runs of one backfill
    - run_ids: only these runs
    - created_after / created_before / updated_after: ISO 8601 strings
      (e.g. '2026-01-31T00:00:00Z'; no time zone means UTC) or Unix seconds
    - limit: max runs to return (default 10)
    - cursor: pass the runId of the last run of the previous page to get the
      next page

    Typical workflows:
    - Find recent failures: get_runs(statuses=['FAILURE'])
    - Failures since this morning: get_runs(statuses=['FAILURE'],
      created_after='2026-01-31T06:00:00Z')
    - Why did a partition fail: get_runs(tags={'dagster/partition': '2026-01-03'})
    - Monitor active runs: get_runs(statuses=['STARTED', 'QUEUED'])

    Runs of an asset backfill that cover a partition range carry the tags
    'dagster/asset_partition_range_start' and '..._end' instead of
    'dagster/partition'.
    """
    query = """
    query Runs($limit: Int!, $filter: RunsFilter, $cursor: String) {
      runsOrError(limit: $limit, filter: $filter, cursor: $cursor) {
        __typename
        ... on Runs {
          results {
            runId
            status
            jobName
            creationTime
            startTime
            endTime
            parentRunId
            rootRunId
            tags { key value }
          }
        }
        ... on InvalidPipelineRunsFilterError { message }
        ... on PythonError { message }
      }
    }
    """
    filter_var: dict[str, Any] = {}
    if statuses:
        filter_var["statuses"] = [status.upper() for status in statuses]
    if job_name:
        filter_var["pipelineName"] = job_name
    if tags:
        filter_var["tags"] = _tag_inputs(tags)
    if run_ids:
        filter_var["runIds"] = run_ids
    for name, field, value in (
        ("created_after", "createdAfter", created_after),
        ("created_before", "createdBefore", created_before),
        ("updated_after", "updatedAfter", updated_after),
    ):
        timestamp = _timestamp_input(value, name)
        if timestamp is not None:
            filter_var[field] = timestamp
    data = gql(
        query,
        {"limit": limit, "filter": filter_var or None, "cursor": cursor},
        env=env,
    )
    runs = data.get("runsOrError") or {}
    if runs.get("__typename") not in (None, "Runs"):
        raise RuntimeError(f"Dagster could not list runs: {runs.get('message', 'unknown error')}")
    return runs.get("results", [])


@mcp.tool()
def get_run_status(run_id: str, env: str | None = None) -> dict:
    """Get full details for a single run: status, config, tags, and run lineage.

    Returns: runId, status, startTime, endTime, jobName, tags, runConfigYaml,
    rootRunId, parentRunId, resolvedOpSelection.

    Use rootRunId and parentRunId to understand re-execution chains — if
    parentRunId is set, this run was re-executed from another run.
    resolvedOpSelection shows which steps were selected for re-execution.

    When to use: after get_runs to inspect a specific run, or to check
    whether a run is a re-execution of a previous one.
    """
    query = """
    query RunStatus($runId: ID!) {
      runOrError(runId: $runId) {
        ... on Run {
          runId
          status
          startTime
          endTime
          jobName
          tags { key value }
          runConfigYaml
          rootRunId
          parentRunId
          resolvedOpSelection
        }
        ... on RunNotFoundError { message }
        ... on PythonError { message }
      }
    }
    """
    data = gql(query, {"runId": run_id}, env=env)
    return data.get("runOrError", {})


@mcp.tool()
def get_run_logs(
    run_id: str,
    cursor: str | None = None,
    limit: int = 100,
    level_filter: str | None = None,
    env: str | None = None,
) -> dict:
    """Get structured log events for a run, with optional severity filtering and pagination.

    Returns events with __typename, timestamp, message, level, and (where applicable)
    stepKey and error details. Events include step starts/completions, failures,
    retries, materializations, and run-level events. EngineEvent events also carry
    metadataEntries — a list of {label, description, value} dicts (e.g. run worker
    image, k8s pod name, step keys) surfaced by the engine.

    Parameters:
    - run_id: the run to fetch logs for
    - level_filter: only return events at this level or above, ordered
      DEBUG < INFO < WARNING < ERROR < CRITICAL. Filtering at 'WARNING'
      therefore also returns ERROR and CRITICAL events. ExecutionStepFailureEvent
      and RunFailureEvent are always included when filtering at 'ERROR' or below,
      regardless of their own level field. Default: None (return all events).
    - cursor: pagination cursor returned in previous response. Pass the
      cursor from the last call to get the next page.
    - limit: max events per page (default 100)

    When to use: to investigate what happened during a run. For a quick
    failure diagnosis, prefer get_run_failure_summary instead — it returns
    a consolidated view in a single call. Use get_run_logs when you need
    the full event stream or want to filter by level.
    """
    query = """
    query RunLogs($runId: ID!, $afterCursor: String, $limit: Int!) {
      logsForRun(runId: $runId, afterCursor: $afterCursor, limit: $limit) {
        ... on EventConnection {
          cursor
          hasMore
          events {
            __typename
            ... on MessageEvent {
              timestamp
              message
              level
              stepKey
            }
            ... on LogsCapturedEvent {
              timestamp
              message
              level
              stepKey
              logKey
              fileKey
            }
            ... on ExecutionStepStartEvent {
              timestamp
              message
              level
              stepKey
            }
            ... on ExecutionStepSuccessEvent {
              timestamp
              message
              level
              stepKey
            }
            ... on ExecutionStepOutputEvent {
              timestamp
              message
              level
              stepKey
              outputName
            }
            ... on ExecutionStepInputEvent {
              timestamp
              message
              level
              stepKey
              inputName
            }
            ... on ExecutionStepFailureEvent {
              timestamp
              message
              level
              stepKey
              error { message causes { message } }
            }
            ... on RunFailureEvent {
              timestamp
              message
              level
              error { message causes { message } }
            }
            ... on ExecutionStepUpForRetryEvent {
              timestamp
              message
              level
              stepKey
              secondsToWait
              error { message causes { message } }
            }
            ... on MaterializationEvent {
              timestamp
              message
              level
              stepKey
            }
            ... on ObjectStoreOperationEvent {
              timestamp
              message
              level
              stepKey
            }
            ... on HandledOutputEvent {
              timestamp
              message
              level
              stepKey
            }
            ... on LoadedInputEvent {
              timestamp
              message
              level
              stepKey
            }
            ... on EngineEvent {
              timestamp
              message
              level
              stepKey
              error { message causes { message } }
__METADATA_ENTRIES__
            }
            ... on RunStartEvent {
              timestamp
              message
              level
            }
            ... on RunSuccessEvent {
              timestamp
              message
              level
            }
            ... on RunStartingEvent {
              timestamp
              message
              level
            }
            ... on RunEnqueuedEvent {
              timestamp
              message
              level
            }
            ... on RunDequeuedEvent {
              timestamp
              message
              level
            }
            ... on RunCancelingEvent {
              timestamp
              message
              level
            }
            ... on RunCanceledEvent {
              timestamp
              message
              level
            }
          }
        }
        ... on RunNotFoundError { message }
        ... on PythonError { message }
      }
    }
    """
    query = query.replace("__METADATA_ENTRIES__", _METADATA_ENTRIES_FRAGMENT)
    data = gql(query, {"runId": run_id, "afterCursor": cursor, "limit": limit}, env=env)
    result = data.get("logsForRun", {})

    for event in result.get("events", []):
        if "metadataEntries" in event:
            event["metadataEntries"] = _flatten_metadata(event["metadataEntries"])

    if level_filter and "events" in result:
        upper = level_filter.upper()
        threshold = _LOG_LEVEL_ORDER.get(upper)
        if threshold is None:
            raise ValueError(
                f"level_filter must be one of {', '.join(_LOG_LEVEL_ORDER)}; got {level_filter!r}."
            )
        # Failure events are always surfaced when filtering at ERROR or below:
        # some carry a level that would otherwise exclude them.
        error_types = ("ExecutionStepFailureEvent", "RunFailureEvent")
        error_rank = _LOG_LEVEL_ORDER["ERROR"]
        result["events"] = [
            e
            for e in result["events"]
            if _LOG_LEVEL_ORDER.get(e.get("level"), -1) >= threshold
            or (threshold <= error_rank and e.get("__typename") in error_types)
        ]

    return result


@mcp.tool()
def get_run_stats(run_id: str, env: str | None = None) -> dict:
    """Get per-step execution statistics for a run: timing, materializations, and expectations.

    Returns runId, status, and a stepStats array where each entry has:
    stepKey, status, startTime, endTime, materializations (with labels),
    and expectationResults (with success flag and labels).

    When to use: to find slow steps (compare startTime/endTime), check which
    steps materialized assets, or verify expectation results.
    For failed runs, prefer get_run_failure_summary which includes step stats
    alongside error details and suggestions.
    """
    query = """
    query RunStats($runId: ID!) {
      runOrError(runId: $runId) {
        ... on Run {
          runId
          status
          stepStats {
            stepKey
            status
            startTime
            endTime
            materializations { label }
            expectationResults { success label }
          }
        }
        ... on RunNotFoundError { message }
        ... on PythonError { message }
      }
    }
    """
    data = gql(query, {"runId": run_id}, env=env)
    return data.get("runOrError", {})


# Pages of run events that get_run_failure_summary reads, at the server's page
# size (1000 events by default).
_FAILURE_EVENTS_MAX_PAGES: Final[int] = 20

# Frames kept from the end of a Python stack trace. The innermost frames name
# the failing user code; the outer ones are mostly Dagster's executor.
_STACK_FRAMES_KEPT: Final[int] = 8


def _trim_error(error: Mapping[str, Any] | None) -> dict:
    """Keep an error's message, class, and the end of its stack, for it and its causes."""
    if not error:
        return {}
    trimmed: dict[str, Any] = {"message": error.get("message")}
    if error.get("className"):
        trimmed["className"] = error["className"]
    stack = error.get("stack") or []
    if stack:
        trimmed["stack"] = "".join(stack[-_STACK_FRAMES_KEPT:])
        if len(stack) > _STACK_FRAMES_KEPT:
            trimmed["stack_frames_omitted"] = len(stack) - _STACK_FRAMES_KEPT
    causes = [_trim_error(cause) for cause in error.get("causes") or []]
    if causes:
        trimmed["causes"] = causes
    return trimmed


@mcp.tool()
def get_run_failure_summary(run_id: str, env: str | None = None) -> dict:
    """Get a consolidated failure diagnosis for a run in a single call.

    This is the BEST tool to use when investigating a failed or canceled run.
    It combines status, step stats, and error logs into one response, avoiding
    the need to call get_run_status + get_run_logs + get_run_stats separately.

    Returns:
    - status, job_name, duration_seconds
    - failure_reason: Dagster's reason tag, for example STEP_FAILURE or
      UNEXPECTED_TERMINATION (the run process stopped), if set
    - failed_steps: list of {step_key, duration, error} for each failed step
    - root_cause_error: the run failure's error, or else the first failed
      step's error.
      Each error has message, className, the last frames of its stack trace,
      and its causes. The user code's exception is usually the first cause.
    - retried_steps: {step_key, retries} for each step that Dagster retried,
      with last_error when the step did not end in failure
    - all_step_durations: timing for every step (not just failed ones)
    - suggestions: automated diagnostic hints (e.g. 'Multiple steps failed',
      'Step was retried before failing', 'Run was canceled')

    If the run did not fail, returns {message: 'Run did not fail.'}.

    When to use: always prefer this over get_run_logs for failed runs.
    Use get_run_logs only when you need the full event stream. If the error
    is not clear, get_compute_logs gives what the failed step printed to
    stdout and stderr.
    """
    # 1. Fetch run status + step stats in one query
    status_query = """
    query FailureSummary($runId: ID!) {
      runOrError(runId: $runId) {
        ... on Run {
          runId
          status
          jobName
          startTime
          endTime
          tags { key value }
          stepStats {
            stepKey
            status
            startTime
            endTime
          }
        }
        ... on RunNotFoundError { message }
        ... on PythonError { message }
      }
    }
    """
    run_data = gql(status_query, {"runId": run_id}, env=env).get("runOrError", {})

    if "message" in run_data:
        return run_data

    status = run_data.get("status", "")
    if status not in ("FAILURE", "CANCELED"):
        return {"run_id": run_id, "status": status, "message": "Run did not fail."}

    # 2. Collect error events from logs. The run failure event is the last
    # event, so read every page: the server's page size, up to a limit.
    error_events: list[dict] = []
    cursor = None
    scanned = 0
    complete = False
    for _ in range(_FAILURE_EVENTS_MAX_PAGES):
        log_query = """
        query FailureLogs($runId: ID!, $afterCursor: String) {
          logsForRun(runId: $runId, afterCursor: $afterCursor) {
            ... on EventConnection {
              cursor
              hasMore
              events {
                __typename
                ... on ExecutionStepFailureEvent {
                  timestamp
                  stepKey
                  error { ...FailureError causes { ...FailureError } }
                }
                ... on RunFailureEvent {
                  timestamp
                  error { ...FailureError causes { ...FailureError } }
                }
                ... on ExecutionStepUpForRetryEvent {
                  timestamp
                  stepKey
                  secondsToWait
                  error { ...FailureError causes { ...FailureError } }
                }
              }
            }
            ... on RunNotFoundError { message }
          }
        }
        fragment FailureError on PythonError { message className stack }
        """
        log_data = gql(log_query, {"runId": run_id, "afterCursor": cursor}, env=env).get(
            "logsForRun", {}
        )
        events = log_data.get("events", [])
        scanned += len(events)
        for e in events:
            if e.get("__typename") in (
                "ExecutionStepFailureEvent",
                "RunFailureEvent",
                "ExecutionStepUpForRetryEvent",
            ):
                error_events.append(e)
        if not log_data.get("hasMore"):
            complete = True
            break
        cursor = log_data.get("cursor")

    # 3. Build step durations
    step_stats = run_data.get("stepStats", [])
    all_step_durations = []
    for s in step_stats:
        dur = None
        if s.get("startTime") and s.get("endTime"):
            dur = round(s["endTime"] - s["startTime"], 2)
        all_step_durations.append(
            {
                "step_key": s["stepKey"],
                "status": s["status"],
                "duration_seconds": dur,
            }
        )

    # 4. Build failed steps with errors: the step's failure, else its last retry.
    failed_step_keys = {s["stepKey"] for s in step_stats if s["status"] == "FAILURE"}
    failure_errors: dict[str, dict] = {}
    retry_errors: dict[str, dict] = {}
    retry_counts: dict[str, int] = {}
    for e in error_events:
        sk = e.get("stepKey")
        if not sk:
            continue
        if e.get("__typename") == "ExecutionStepFailureEvent":
            failure_errors.setdefault(sk, _trim_error(e.get("error")))
        elif e.get("__typename") == "ExecutionStepUpForRetryEvent":
            retry_errors[sk] = _trim_error(e.get("error"))
            retry_counts[sk] = retry_counts.get(sk, 0) + 1
    step_errors = {**retry_errors, **failure_errors}

    failed_steps = []
    for s in step_stats:
        if s["stepKey"] in failed_step_keys:
            dur = None
            if s.get("startTime") and s.get("endTime"):
                dur = round(s["endTime"] - s["startTime"], 2)
            failed_steps.append(
                {
                    "step_key": s["stepKey"],
                    "duration_seconds": dur,
                    "error": step_errors.get(s["stepKey"], {}),
                }
            )

    # 5. Root cause error: the run failure's error. A run that failed because
    # steps failed has a run failure event without an error; use the first step's.
    run_failure = [e for e in error_events if e.get("__typename") == "RunFailureEvent"]
    root_cause = _trim_error(run_failure[0].get("error")) if run_failure else {}
    if not root_cause and failed_steps:
        root_cause = failed_steps[0].get("error", {})
    retried_steps = []
    for sk, count in retry_counts.items():
        entry: dict[str, Any] = {"step_key": sk, "retries": count}
        if sk not in failed_step_keys:
            entry["last_error"] = retry_errors[sk]
        retried_steps.append(entry)

    # 6. Suggestions
    suggestions: list[str] = []
    retries = [e for e in error_events if e.get("__typename") == "ExecutionStepUpForRetryEvent"]
    if retries:
        retry_keys = {e["stepKey"] for e in retries}
        suggestions.append(f"Steps retried before failing: {', '.join(sorted(retry_keys))}")
    if len(failed_steps) > 1:
        suggestions.append(
            f"Multiple steps failed ({len(failed_steps)}). "
            f"First failure: {failed_steps[0]['step_key']} — downstream failures may be cascading."
        )
    if status == "CANCELED":
        suggestions.append("Run was canceled, not all steps may have executed.")
    running = [s["stepKey"] for s in step_stats if s["status"] == "IN_PROGRESS"]
    if status == "FAILURE" and running:
        suggestions.append(
            f"Steps were still running when the run failed: {', '.join(running)}. The run "
            "process stopped before they ended; see root_cause_error and failure_reason."
        )
    if not complete:
        suggestions.append(
            f"Only the first {scanned} events of the run were searched; later errors are "
            "not included."
        )

    run_dur = None
    if run_data.get("startTime") and run_data.get("endTime"):
        run_dur = round(run_data["endTime"] - run_data["startTime"], 2)

    return {
        "run_id": run_id,
        "status": status,
        "job_name": run_data.get("jobName"),
        "duration_seconds": run_dur,
        "failure_reason": _tag_value(run_data.get("tags"), "dagster/failure_reason"),
        "failed_steps": failed_steps,
        "root_cause_error": root_cause or None,
        "retried_steps": retried_steps,
        "all_step_durations": all_step_durations,
        "suggestions": suggestions,
    }


# ── Compute logs ──────────────────────────────────────────────────────────────

# Dagster names each process's stdout and stderr files with a random file key
# and records it only in a LogsCapturedEvent, so the run's event log is the
# index. The scan uses the server's page size (DAGSTER_UI_EVENT_LOAD_CHUNK_SIZE,
# 1000 by default): Dagster 1.13.6 rejects a larger limit.
_CAPTURE_EVENTS_MAX_PAGES: Final[int] = 20
_CAPTURES_PER_STEP: Final[int] = 5

_LOG_STREAMS: Final[tuple[str, str]] = ("stdout", "stderr")
_LOG_DEFAULT_BYTES: Final[int] = 10_000
_LOG_MIN_BYTES: Final[int] = 100
_LOG_MAX_BYTES: Final[int] = 500_000
# capturedLogs can read from an offset but cannot report a file's size. To
# return the end of a long log, probe offsets for data: first doubling, then
# in even steps, until the end is known to within max_bytes.
_LOG_PROBE_LIMIT: Final[int] = 1 << 36  # 64 GiB; past this, show the start
_LOG_PROBES_PER_ROUND: Final[int] = 32
_LOG_PROBE_ROUNDS: Final[int] = 8
_LOG_PAST_END: Final[int] = 1 << 40
# Cloud compute log managers keep only this many bytes when they upload a log
# (DAGSTER_TRUNCATE_COMPUTE_LOGS_UPLOAD_BYTES), and add no marker.
_LOG_UPLOAD_CUT_BYTES: Final[int] = 50 * 1024 * 1024

_CAPTURED_LOG_EVENTS_QUERY: Final[str] = """
query CapturedLogEvents($runId: ID!, $afterCursor: String) {
  logsForRun(runId: $runId, afterCursor: $afterCursor) {
    __typename
    ... on EventConnection {
      cursor
      hasMore
      events {
        __typename
        ... on LogsCapturedEvent {
          fileKey
          stepKeys
          externalStdoutUrl
          externalStderrUrl
        }
      }
    }
    ... on RunNotFoundError { message }
    ... on PythonError { message }
  }
}
"""


def _captured_log_events(run_id: str, env: str | None) -> tuple[list[dict], int | None]:
    """Return the run's LogsCapturedEvents, one per file key.

    The second value is None when the scan read all events, otherwise the
    number of events it read.
    """
    captures: dict[str, dict] = {}
    cursor = None
    scanned = 0
    for _ in range(_CAPTURE_EVENTS_MAX_PAGES):
        result = gql(
            _CAPTURED_LOG_EVENTS_QUERY, {"runId": run_id, "afterCursor": cursor}, env=env
        ).get("logsForRun") or {}
        if result.get("__typename") != "EventConnection":
            raise RuntimeError(
                result.get("message") or f"Dagster could not read the logs of run {run_id!r}."
            )
        events = result.get("events") or []
        scanned += len(events)
        for event in events:
            if event.get("__typename") == "LogsCapturedEvent" and event.get("fileKey"):
                captures.setdefault(event["fileKey"], event)
        if not result.get("hasMore"):
            return list(captures.values()), None
        cursor = result.get("cursor")
    return list(captures.values()), scanned


def _log_offsets(cursor: str | None) -> tuple[int, int] | None:
    """Parse a capturedLogs cursor, 'stdout_offset:stderr_offset', into byte offsets."""
    try:
        stdout_offset, stderr_offset = (int(part) for part in (cursor or "").split(":"))
    except ValueError:
        return None
    return stdout_offset, stderr_offset


def _read_captured_logs(
    log_key: Sequence[str],
    reads: Sequence[tuple[str | None, int]],
    env: str | None,
    with_metadata: bool = False,
) -> dict:
    """Send one request with a capturedLogs read per (cursor, limit) pair, aliased r0, r1, ..."""
    definitions = ["$logKey: [String!]!"]
    fields = []
    variables: dict[str, Any] = {"logKey": list(log_key)}
    if with_metadata:
        fields.append(
            "meta: capturedLogsMetadata(logKey: $logKey) "
            "{ stdoutDownloadUrl stdoutLocation stderrDownloadUrl stderrLocation }"
        )
    for index, (cursor, limit) in enumerate(reads):
        definitions += [f"$c{index}: String", f"$l{index}: Int"]
        variables[f"c{index}"] = cursor
        variables[f"l{index}"] = limit
        fields.append(
            f"r{index}: capturedLogs(logKey: $logKey, cursor: $c{index}, limit: $l{index}) "
            "{ stdout stderr cursor }"
        )
    query = f"query CapturedLogs({', '.join(definitions)}) {{\n  " + "\n  ".join(fields) + "\n}"
    return gql(query, variables, env=env)


def _log_tail_text(text: str, max_bytes: int) -> str:
    """Keep the last max_bytes of text, starting at a full line where one is near."""
    # A read that starts inside a multi-byte character decodes it as U+FFFD.
    text = text.lstrip("\ufffd")
    data = text.encode("utf-8")
    if len(data) <= max_bytes:
        return text
    tail = data[-max_bytes:].decode("utf-8", errors="ignore")
    newline = tail.find("\n")
    if 0 <= newline < len(tail) // 2:
        tail = tail[newline + 1 :]
    return tail


def _log_cursor(positions: Mapping[str, int]) -> str:
    """Build a capturedLogs cursor. A stream left out reads from past its end: nothing."""
    return ":".join(str(positions.get(stream, _LOG_PAST_END)) for stream in _LOG_STREAMS)


def _probe_log_ends(
    log_key: Sequence[str],
    bounds: dict[str, tuple[int, int | None]],
    offsets_by_stream: Mapping[str, Sequence[int]],
    env: str | None,
) -> None:
    """Read one byte at each offset, and narrow each stream's (lo, hi) bounds of its end."""
    count = max(len(offsets) for offsets in offsets_by_stream.values())
    reads = [
        (_log_cursor({s: o[i] for s, o in offsets_by_stream.items() if i < len(o)}), 1)
        for i in range(count)
    ]
    answer = _read_captured_logs(log_key, reads, env)
    for stream, offsets in offsets_by_stream.items():
        lo, hi = bounds[stream]
        for i, offset in enumerate(offsets):
            if (answer.get(f"r{i}") or {}).get(stream):
                lo = max(lo, offset)
            elif hi is None or offset < hi:
                hi = offset
        # A log that grew during the probes can show data past hi.
        bounds[stream] = (lo, hi if hi is None or hi > lo else lo + 1)


def _read_log_tails(
    log_key: Sequence[str], streams: Sequence[str], max_bytes: int, env: str | None
) -> tuple[dict[str, dict], dict]:
    """Return {stream: {text, size_bytes, truncated}} with the end of each stream, and metadata."""
    data = _read_captured_logs(log_key, [(None, max_bytes)], env, with_metadata=True)
    head = data.get("r0") or {}
    metadata = data.get("meta") or {}
    offsets = _log_offsets(head.get("cursor"))

    results: dict[str, dict] = {}
    # For each stream longer than the first read: (lo, hi), where the log has a
    # byte at offset lo and none at offset hi. hi is None until a probe finds it.
    bounds: dict[str, tuple[int, int | None]] = {}
    for index, stream in enumerate(_LOG_STREAMS):
        if stream not in streams:
            continue
        text = head.get(stream) or ""
        if offsets is None:
            results[stream] = {"text": text, "size_bytes": None, "truncated": None}
            if len(text.encode("utf-8")) >= max_bytes:
                results[stream]["note"] = (
                    "Dagster gave no log position, so only the start of the log is shown."
                )
        elif offsets[index] < max_bytes:
            results[stream] = {"text": text, "size_bytes": offsets[index], "truncated": False}
        else:
            results[stream] = {"text": text, "size_bytes": None, "truncated": True}
            bounds[stream] = (max_bytes - 1, None)
    # Cloud log managers give a location only for a complete log. For a partial
    # log, every read downloads the whole file again, so do not probe it.
    for stream in list(bounds):
        if not metadata.get(f"{stream}Location"):
            del bounds[stream]
            results[stream]["note"] = (
                "Dagster gave no storage location for this log: the step still runs, or "
                "its process stopped before the end. Only the start of the log is shown, "
                "because each read can download the full log again. Use a larger max_bytes "
                "to see more."
            )
    if not bounds:
        return results, metadata

    doubling = []
    offset = max_bytes
    while offset <= _LOG_PROBE_LIMIT:
        doubling.append(offset)
        offset *= 2
    _probe_log_ends(log_key, bounds, dict.fromkeys(bounds, doubling), env)
    for stream, (_lo, hi) in list(bounds.items()):
        if hi is None:
            del bounds[stream]
            results[stream]["note"] = (
                f"The log is larger than {_LOG_PROBE_LIMIT} bytes, so only its start is shown."
            )

    steps = _LOG_PROBES_PER_ROUND + 1
    for _ in range(_LOG_PROBE_ROUNDS):
        wide = {s: (lo, hi) for s, (lo, hi) in bounds.items() if hi - lo > max_bytes}
        if not wide:
            break
        evenly = {
            s: [lo + (hi - lo) * j // steps for j in range(1, steps)]
            for s, (lo, hi) in wide.items()
        }
        _probe_log_ends(log_key, bounds, evenly, env)

    if not bounds:
        return results, metadata
    starts = {s: max(0, lo + 1 - max_bytes) for s, (lo, _hi) in bounds.items()}
    limit = max(hi - starts[s] for s, (_lo, hi) in bounds.items())
    tail = _read_captured_logs(log_key, [(_log_cursor(starts), limit)], env).get("r0") or {}
    end = _log_offsets(tail.get("cursor"))
    for stream in bounds:
        size = end[_LOG_STREAMS.index(stream)] if end else None
        results[stream] = {
            "text": _log_tail_text(tail.get(stream) or "", max_bytes),
            "size_bytes": size,
            "truncated": size is None or size > max_bytes,
        }
        location = metadata.get(f"{stream}Location") or ""
        if size == _LOG_UPLOAD_CUT_BYTES and not location.startswith("/"):
            results[stream]["note"] = (
                "Dagster keeps only the first 50 MiB of a log that it uploads to cloud "
                "storage, so this text is not the real end of the log."
            )
    return results, metadata


def _log_capture_entry(capture: Mapping[str, Any]) -> dict:
    return {"file_key": capture.get("fileKey"), "step_keys": capture.get("stepKeys") or []}


@mcp.tool()
def get_compute_logs(
    run_id: str,
    step_key: str | None = None,
    stream: str = "both",
    max_bytes: int = _LOG_DEFAULT_BYTES,
    env: str | None = None,
) -> dict:
    """Get what a run's steps printed to stdout and stderr (Dagster's compute logs).

    Dagster saves the stdout and stderr of each process that runs steps. These
    logs often hold the cause of a failure that the event log does not show:
    output of dbt, Spark, or shell commands, print() output, and full Python
    tracebacks.

    Parameters:
    - run_id: the run
    - step_key: the step to read. get_run_failure_summary and get_run_stats
      give the step keys. If you omit it and the run captured logs in more
      than one process, the result lists the captures and their step keys,
      without log text.
    - stream: 'stdout', 'stderr', or 'both' (default). Python logging and
      tracebacks usually go to stderr; print() goes to stdout.
    - max_bytes: bytes to return per stream (default 10000, from 100 to
      500000). For a longer log, the result has the END of the log, where
      errors usually are. Exception: a cloud-stored log of a step that still
      runs, or whose process stopped early, gives the start and a note.

    Returns run_id and captures, one per process. A retried step can have one
    capture per attempt; the last 5 are returned, oldest first. Each has:
    - file_key, step_keys: all steps that ran in that process. With the
      in-process executor, all steps share one log.
    - stdout, stderr: {text, size_bytes, truncated}. truncated is true when
      text is only the end of the log.
    - stdout_location, stderr_location: where Dagster stores the file, for
      example a path or an s3:// URL
    - external_stdout_url, external_stderr_url: links, when the log storage
      gives links instead of text
    """
    stream_key = stream.lower()
    if stream_key not in ("stdout", "stderr", "both"):
        raise ValueError(f"stream must be 'stdout', 'stderr', or 'both'; got {stream!r}.")
    streams = _LOG_STREAMS if stream_key == "both" else (stream_key,)
    if not _LOG_MIN_BYTES <= max_bytes <= _LOG_MAX_BYTES:
        raise ValueError(
            f"max_bytes must be from {_LOG_MIN_BYTES} to {_LOG_MAX_BYTES}; got {max_bytes}."
        )

    captures, scanned = _captured_log_events(run_id, env)
    result: dict[str, Any] = {"run_id": run_id}
    if scanned is not None:
        result["note"] = (
            f"Only the first {scanned} events of the run were searched for captured logs. "
            "Later steps can have logs that are not listed."
        )
    if not captures:
        result["captures"] = []
        result["message"] = (
            "This run has no captured logs. Its steps did not start, or the "
            "instance does not capture compute logs."
        )
        return result

    if step_key is not None:
        selected = [c for c in captures if step_key in (c.get("stepKeys") or [])]
        if not selected:
            known = sorted({key for c in captures for key in c.get("stepKeys") or []})
            shown = ", ".join(known[:50])
            if len(known) > 50:
                shown += f", and {len(known) - 50} more"
            result["captures"] = []
            result["message"] = f"No captured logs for step {step_key!r}. Steps with logs: {shown}."
            return result
        selected = selected[-_CAPTURES_PER_STEP:]
    elif len(captures) > 1:
        result["captures"] = [_log_capture_entry(c) for c in captures]
        result["message"] = (
            f"This run captured logs in {len(captures)} processes. Pass step_key to "
            "read the logs of one step."
        )
        return result
    else:
        selected = captures

    entries = []
    for capture in selected:
        entry = _log_capture_entry(capture)
        log_key = [run_id, "compute_logs", capture["fileKey"]]
        texts, metadata = _read_log_tails(log_key, streams, max_bytes, env)
        for name in streams:
            entry[name] = texts[name]
            location = metadata.get(f"{name}Location")
            if location:
                entry[f"{name}_location"] = location
            external = capture.get(f"external{name.capitalize()}Url")
            if external:
                entry[f"external_{name}_url"] = external
        entries.append(entry)
    result["captures"] = entries
    return result


# ── Assets ────────────────────────────────────────────────────────────────────


def _asset_key_input(asset_key: str) -> dict:
    return {"path": asset_key.split("/")}


def _asset_node_key(node: dict) -> str:
    return "/".join(node.get("assetKey", {}).get("path", []))


@mcp.tool()
def get_recent_materializations(
    asset_key: str,
    limit: int = 5,
    env: str | None = None,
    partitions: list[str] | None = None,
) -> list[dict]:
    """Get the most recent materializations for an asset, newest first, with metadata.

    Returns a list of materializations, each with: runId, timestamp (Unix
    milliseconds, as a string), partition (null for unpartitioned assets),
    stepKey, assetKey, and metadataEntries as {label, description, value}.

    - asset_key: the asset key, '/'-separated for nested keys (e.g. 'raw/orders')
    - limit: max materializations to return (default 5)
    - partitions: only materializations of these partition keys
      (e.g. ['2026-01-03']). Multi-partition keys look like '2026-01-03|eu'.

    When to use: to check when an asset or a partition was last
    materialized, track materialization frequency, or inspect metadata from
    recent runs. For which partitions failed, use get_asset_partition_status.
    For a broader health view, use get_asset_health. Raises when the asset
    does not exist, so an empty list means it has no materializations.
    """
    query = (
        """
    query AssetRuns($assetKey: AssetKeyInput!, $limit: Int!, $partitions: [String!]) {
      assetOrError(assetKey: $assetKey) {
        __typename
        ... on Asset {
          assetMaterializations(limit: $limit, partitions: $partitions) {
            runId
            timestamp
            partition
            stepKey
            assetKey { path }
__METADATA_ENTRIES__
          }
        }
        ... on AssetNotFoundError { message }
      }
    }
    """
    ).replace("__METADATA_ENTRIES__", _METADATA_ENTRIES_FRAGMENT)
    variables = {"assetKey": _asset_key_input(asset_key), "limit": limit, "partitions": partitions}
    asset = gql(query, variables, env=env).get("assetOrError") or {}
    if asset.get("__typename") == "AssetNotFoundError":
        raise RuntimeError(f"Asset {asset_key!r} not found.")
    events = asset.get("assetMaterializations") or []
    for event in events:
        event["metadataEntries"] = _flatten_metadata(event.get("metadataEntries"))
    return events


# Both freshness policy kinds. internalFreshnessPolicy is the newer policy
# (FreshnessPolicy.time_window / .cron in code); freshnessPolicy is the
# deprecated LegacyFreshnessPolicy.
_FRESHNESS_POLICY_SELECTION: Final[str] = """
        freshnessPolicy { maximumLagMinutes cronSchedule cronScheduleTimezone }
        internalFreshnessPolicy {
          __typename
          ... on TimeWindowFreshnessPolicy { failWindowSeconds warnWindowSeconds }
          ... on CronFreshnessPolicy { deadlineCron lowerBoundDeltaSeconds timezone }
        }"""


def _freshness_policy(node: Mapping[str, Any]) -> dict | None:
    """Describe an asset's freshness policy, preferring the newer kind."""
    policy = node.get("internalFreshnessPolicy")
    if policy:
        if policy.get("__typename") == "TimeWindowFreshnessPolicy":
            return {
                "type": "time_window",
                "fail_window_seconds": policy.get("failWindowSeconds"),
                "warn_window_seconds": policy.get("warnWindowSeconds"),
            }
        if policy.get("__typename") == "CronFreshnessPolicy":
            return {
                "type": "cron",
                "deadline_cron": policy.get("deadlineCron"),
                "lower_bound_delta_seconds": policy.get("lowerBoundDeltaSeconds"),
                "timezone": policy.get("timezone"),
            }
        return {"type": policy.get("__typename")}
    legacy = node.get("freshnessPolicy")
    if legacy:
        return {
            "type": "legacy",
            "max_lag_minutes": legacy.get("maximumLagMinutes"),
            "cron": legacy.get("cronSchedule"),
            "cron_timezone": legacy.get("cronScheduleTimezone"),
        }
    return None


@mcp.tool()
def get_asset_details(asset_keys: list[str], env: str | None = None) -> list[dict]:
    """Get detailed metadata for one or more assets: description, lineage, ownership, partitions.

    - asset_keys: list of asset keys, '/'-separated for nested keys
      (e.g. ['my_extract', 'raw/orders'])

    Returns per asset: assetKey, description, groupName, kinds, owners,
    tags, op name, jobNames, repository (name and code location),
    isMaterializable, isExecutable, isObservable, isPartitioned,
    partitionDefinition (description, type, dimensions), pools,
    automationCondition (its label, if the asset uses declarative
    automation), freshness_policy, hasAssetChecks, dependencyKeys (upstream
    assets), dependedByKeys (downstream assets), and the latest
    materialization (runId, timestamp, partition).

    When to use: to understand an asset's lineage (what it depends on
    and what depends on it), who owns it, how it is automated, or how it
    is partitioned. Use search_assets first if you don't know the exact key.
    """
    query = (
        """
    query AssetDetails($assetKeys: [AssetKeyInput!]!) {
      assetNodes(assetKeys: $assetKeys) {
        assetKey { path }
        description
        groupName
        kinds
        owners {
          __typename
          ... on TeamAssetOwner { team }
          ... on UserAssetOwner { email }
        }
        tags { key value }
        op { name }
        jobNames
        repository { name location { name } }
        isMaterializable
        isExecutable
        isObservable
        isPartitioned
        partitionDefinition {
          description
          type
          dimensionTypes { name type }
        }
        pools
        automationCondition { label }
        hasAssetChecks
__FRESHNESS__
        dependencyKeys { path }
        dependedByKeys { path }
        assetMaterializations(limit: 1) {
          runId
          timestamp
          partition
        }
      }
    }
    """
    ).replace("__FRESHNESS__", _FRESHNESS_POLICY_SELECTION)
    keys = [_asset_key_input(k) for k in asset_keys]
    nodes = gql(query, {"assetKeys": keys}, env=env).get("assetNodes", [])
    for node in nodes:
        node["freshness_policy"] = _freshness_policy(node)
        node.pop("freshnessPolicy", None)
        node.pop("internalFreshnessPolicy", None)
    return nodes


@mcp.tool()
def search_assets(
    prefix: str | None = None,
    group: str | None = None,
    env: str | None = None,
) -> list[dict]:
    """Search and list assets by name prefix or group. Use this to discover assets.

    Returns per asset: assetKey, groupName, description, isPartitioned, op name.

    - prefix: case-insensitive substring match on any part of the asset key
      (e.g. 'raw_' finds 'raw_orders', 'raw_users')
    - group: exact match on groupName (case-insensitive, e.g. 'analytics')
    - Both filters can be combined.
    - If neither is passed, returns ALL assets.

    When to use: to discover available assets before calling get_asset_details
    or get_asset_health. Use prefix for fuzzy search, group for scoped listing.
    """
    query = """
    query AllAssets {
      assetNodes {
        assetKey { path }
        groupName
        description
        isPartitioned
        op { name }
      }
    }
    """
    data = gql(query, env=env)
    nodes = data.get("assetNodes", [])
    if prefix:
        prefix_lower = prefix.lower()
        nodes = [n for n in nodes if any(prefix_lower in p.lower() for p in n["assetKey"]["path"])]
    if group:
        group_lower = group.lower()
        nodes = [n for n in nodes if (n.get("groupName") or "").lower() == group_lower]
    return nodes


@mcp.tool()
def resolve_asset_selection(asset_selection: str, env: str | None = None) -> dict:
    """Resolve Dagster asset-selection syntax into concrete assets without launching a run.

    Supported syntax:
    - key predicates (``key:orders`` or bare ``orders``) with ``*`` wildcards
    - ``group:``, ``tag:``, ``kind:``, and ``owner:`` predicates
    - case-insensitive ``and``, ``or``, and ``not`` with parentheses
    - ``roots(...)`` and ``sinks(...)``
    - upstream/downstream traversal such as ``+orders``, ``2+orders``,
      ``orders+``, ``orders+2``, or ``1+orders+2``

    Returns ``asset_keys`` as slash-delimited strings ready to pass to
    materialize_assets or backfill_assets, plus compact GraphQL-shaped asset
    summaries. This tool is read-only and does not filter external, observable,
    non-executable, or partitioned matches.
    """
    try:
        expression = parse_asset_selection(asset_selection)
    except (AssetSelectionSyntaxError, TypeError) as exc:
        return {
            "selection": asset_selection,
            "asset_keys": [],
            "assets": [],
            "message": str(exc),
        }

    query = """
    query AssetSelectionGraph {
      assetNodes {
        assetKey { path }
        groupName
        tags { key value }
        kinds
        owners {
          __typename
          ... on TeamAssetOwner { team }
          ... on UserAssetOwner { email }
        }
        dependencyKeys { path }
        jobNames
        repository {
          name
          location { name }
        }
        isMaterializable
        isExecutable
        isObservable
        isPartitioned
      }
    }
    """
    nodes = gql(query, env=env).get("assetNodes", [])
    try:
        resolved = evaluate_asset_selection(nodes, expression)
    except Exception as exc:  # noqa: BLE001 - report to the agent, never crash the tool
        # Evaluation-time failures (e.g. unexpected node shape) return a
        # structured message rather than an unhandled tool error.
        return {
            "selection": asset_selection,
            "asset_keys": [],
            "assets": [],
            "message": f"Failed to evaluate selection: {exc}",
        }

    fields = (
        "assetKey",
        "groupName",
        "repository",
        "jobNames",
        "isMaterializable",
        "isExecutable",
        "isObservable",
        "isPartitioned",
    )
    assets = [{field: node.get(field) for field in fields} for node in resolved]
    return {
        "selection": asset_selection,
        "asset_keys": ["/".join(node["assetKey"]["path"]) for node in resolved],
        "assets": assets,
    }


_ASSET_HEALTH_QUERY: Final[str] = (
    """
query AssetHealth($assetKeys: [AssetKeyInput!]!) {
  assetNodes(assetKeys: $assetKeys) {
    assetKey { path }
    groupName
    isMaterializable
    isPartitioned
    partitionStats { numPartitions numMaterialized numFailed numMaterializing }
__FRESHNESS__
    freshnessStatusInfo { freshnessStatus }
    staleCauses { reason dependency { path } }
    automationCondition { label }
    assetChecksOrError {
      __typename
      ... on AssetChecks {
        checks {
          name
          executionForLatestMaterialization {
            status
            evaluation { severity }
          }
        }
      }
    }
  }
  assetsLatestInfo(assetKeys: $assetKeys) {
    assetKey { path }
    latestMaterialization { runId timestamp partition }
    latestRun { runId status endTime }
    inProgressRunIds
    unstartedRunIds
  }
  assetsOrError(assetKeys: $assetKeys) {
    __typename
    ... on AssetConnection {
      nodes {
        key { path }
        assetHealth {
          assetHealth
          materializationStatus
          assetChecksStatus
          freshnessStatus
        }
      }
    }
  }
}
"""
).replace("__FRESHNESS__", _FRESHNESS_POLICY_SELECTION)

_FAILED_RUN_STATUSES: Final[frozenset[str]] = frozenset({"FAILURE", "CANCELED"})
_UNHEALTHY_STATUSES: Final[frozenset[str]] = frozenset({"WARNING", "DEGRADED"})


def _check_summary(node: Mapping[str, Any]) -> dict | None:
    """Group an asset's checks by the result of their run against the latest materialization."""
    checks_or_error = node.get("assetChecksOrError") or {}
    if checks_or_error.get("__typename") != "AssetChecks":
        return None
    checks = checks_or_error.get("checks") or []
    if not checks:
        return None
    summary: dict[str, Any] = {
        "total": len(checks),
        "failed": [],
        "warning": [],
        "execution_failed": [],
        "not_run": [],
    }
    for check in checks:
        execution = check.get("executionForLatestMaterialization") or {}
        status = execution.get("status")
        severity = (execution.get("evaluation") or {}).get("severity")
        if status == "FAILED":
            summary["warning" if severity == "WARN" else "failed"].append(check["name"])
        elif status == "EXECUTION_FAILED":
            summary["execution_failed"].append(check["name"])
        elif status is None:
            summary["not_run"].append(check["name"])
    return summary


def _health_issues(entry: Mapping[str, Any]) -> list[str]:
    """Plain-language problems that need attention, derived from one asset's health entry."""
    issues = []
    latest_run = entry.get("latest_run") or {}
    materialization = entry.get("last_materialization") or {}
    # A run can fail in a later step after it materialized this asset; the
    # Dagster UI does not flag the asset then, and neither does this.
    if (
        latest_run.get("status") in _FAILED_RUN_STATUSES
        and latest_run.get("run_id") != materialization.get("run_id")
    ):
        issues.append(
            f"Latest run {latest_run['run_id']} ended with {latest_run['status']}; "
            "use get_run_failure_summary."
        )
    partitions = entry.get("partitions") or {}
    if partitions.get("failed"):
        issues.append(
            f"{partitions['failed']} failed partition(s); use get_asset_partition_status."
        )
    checks = entry.get("checks") or {}
    for name in checks.get("failed", []):
        issues.append(f"Check {name!r} failed (ERROR severity).")
    for name in checks.get("warning", []):
        issues.append(f"Check {name!r} failed (WARN severity).")
    for name in checks.get("execution_failed", []):
        issues.append(f"Check {name!r} could not execute.")
    if entry.get("freshness_status") in _UNHEALTHY_STATUSES:
        issues.append(f"Freshness status is {entry['freshness_status']}.")
    health = (entry.get("dagster_health") or {}).get("assetHealth")
    if health in _UNHEALTHY_STATUSES:
        issues.append(f"Dagster reports the asset as {health}.")
    if entry.get("materializable") and not entry.get("last_materialization") and not (
        latest_run or partitions
    ):
        issues.append("Asset has never been materialized.")
    return issues


@mcp.tool()
def get_asset_health(asset_key_or_group: str, env: str | None = None) -> list[dict]:
    """Get a consolidated health view for a single asset or all assets in a group.

    This is the BEST tool to assess whether assets are healthy and up-to-date.

    - asset_key_or_group: pass either a single asset key (e.g. 'my_report',
      'raw/orders') or a group name (e.g. 'analytics'). If it matches a group,
      returns health for ALL assets in that group.

    Returns per asset:
    - asset_key, group, materializable (false for external or observable
      assets)
    - issues: plain-language problems that need attention (empty if none
      were found): a latest run that failed before it materialized the
      asset, failed partitions, failed checks, a WARNING or DEGRADED
      freshness status, or no materialization at all
    - latest_run: {run_id, status, end_time} of the most recent run that
      targeted the asset, whatever its result. latest_run_status repeats
      its status. Times in this tool are Unix seconds
    - runs_in_progress, runs_queued: run ids of runs that target the asset now
    - last_materialization: {run_id, timestamp, partition} of the latest
      successful materialization
    - partitions: {total, materialized, failed, materializing} for
      partitioned assets
    - checks: {total, failed, warning, execution_failed, not_run} with
      check names, judged against the latest materialization
    - freshness_policy and freshness_status (HEALTHY, WARNING, DEGRADED,
      UNKNOWN) when the asset has a freshness policy
    - stale, stale_causes: whether upstream data or code changed since the
      last materialization
    - automation_condition: label of the asset's automation condition, if any.
      get_asset_automation explains why automation did or did not run it.
    - dagster_health: Dagster's own health status, only on deployments that
      compute it (Dagster+)

    When to use: to check if critical assets are fresh, find failing assets
    in a group, or verify that recent runs succeeded. Prefer this over
    get_recent_materializations when you need a health assessment rather
    than raw materialization history.
    """
    # First try as a group — fetch all assets and filter
    all_query = """
    query AllAssets {
      assetNodes {
        assetKey { path }
        groupName
      }
    }
    """
    all_nodes = gql(all_query, env=env).get("assetNodes", [])
    group_keys = [
        n["assetKey"]["path"]
        for n in all_nodes
        if (n.get("groupName") or "").lower() == asset_key_or_group.lower()
    ]
    if group_keys:
        asset_keys_input = [{"path": k} for k in group_keys]
    else:
        asset_keys_input = [_asset_key_input(asset_key_or_group)]

    data = gql(_ASSET_HEALTH_QUERY, {"assetKeys": asset_keys_input}, env=env)
    nodes = data.get("assetNodes", [])
    if not nodes:
        return [{"asset_key": asset_key_or_group, "message": "Asset not found."}]

    latest_by_key = {
        "/".join(info["assetKey"]["path"]): info
        for info in data.get("assetsLatestInfo") or []
    }
    assets = data.get("assetsOrError") or {}
    asset_nodes = assets.get("nodes") or [] if assets.get("__typename") == "AssetConnection" else []
    health_by_key = {
        "/".join(asset["key"]["path"]): asset.get("assetHealth")
        for asset in asset_nodes
    }

    results = []
    seen: set[str] = set()
    for node in nodes:
        key = _asset_node_key(node)
        if key in seen:
            continue
        seen.add(key)
        latest = latest_by_key.get(key) or {}
        materialization = latest.get("latestMaterialization")
        run = latest.get("latestRun")
        entry: dict[str, Any] = {
            "asset_key": key,
            "group": node.get("groupName"),
            "issues": [],
            "materializable": node.get("isMaterializable"),
            "latest_run": (
                {"run_id": run["runId"], "status": run["status"], "end_time": run.get("endTime")}
                if run
                else None
            ),
            "latest_run_status": run["status"] if run else None,
            "runs_in_progress": latest.get("inProgressRunIds") or [],
            "runs_queued": latest.get("unstartedRunIds") or [],
            "last_materialization": (
                {
                    "run_id": materialization["runId"],
                    # Dagster sends milliseconds as a string; match end_time.
                    "timestamp": float(materialization["timestamp"]) / 1000,
                    "partition": materialization.get("partition"),
                }
                if materialization
                else None
            ),
        }
        stats = node.get("partitionStats")
        if node.get("isPartitioned") and stats:
            entry["partitions"] = {
                "total": stats["numPartitions"],
                "materialized": stats["numMaterialized"],
                "failed": stats["numFailed"],
                "materializing": stats["numMaterializing"],
            }
        checks = _check_summary(node)
        if checks:
            entry["checks"] = checks
        freshness_policy = _freshness_policy(node)
        if freshness_policy:
            entry["freshness_policy"] = freshness_policy
        freshness_status = (node.get("freshnessStatusInfo") or {}).get("freshnessStatus")
        if freshness_status:
            entry["freshness_status"] = freshness_status
        stale_causes = node.get("staleCauses") or []
        entry["stale"] = bool(stale_causes)
        entry["stale_causes"] = [c.get("reason", "") for c in stale_causes]
        automation = (node.get("automationCondition") or {}).get("label")
        if automation:
            entry["automation_condition"] = automation
        if health_by_key.get(key):
            entry["dagster_health"] = health_by_key[key]
        entry["issues"] = _health_issues(entry)
        results.append(entry)
    return results


# Statuses a partition can have, as reported by get_asset_partition_status.
# The first three come from Dagster; "missing" is every other partition key.
_PARTITION_STATUS_NAMES: Final[tuple[str, ...]] = (
    "failed",
    "materializing",
    "materialized",
    "missing",
)

_ASSET_PARTITION_STATUS_QUERY: Final[str] = """
query AssetPartitionStatus($assetKeys: [AssetKeyInput!]!) {
  assetNodes(assetKeys: $assetKeys) {
    assetKey { path }
    isPartitioned
    partitionDefinition { description }
    partitionKeysByDimension { name partitionKeys }
    assetPartitionStatuses {
      __typename
      ...DefaultStatuses
      ...TimeStatuses
      ... on MultiPartitionStatuses {
        primaryDimensionName
        ranges {
          primaryDimStartKey
          primaryDimEndKey
          secondaryDim {
            __typename
            ...DefaultStatuses
            ...TimeStatuses
          }
        }
      }
    }
  }
}
fragment DefaultStatuses on DefaultPartitionStatuses {
  materializedPartitions
  failedPartitions
  materializingPartitions
}
fragment TimeStatuses on TimePartitionStatuses {
  ranges { startKey endKey status }
}
"""


def _keys_by_status_1d(
    statuses: Mapping[str, Any],
    keys: Sequence[str],
    unresolved: list[dict[str, str]],
) -> dict[str, list[str]]:
    """Map materialized/failed/materializing to partition keys for one dimension.

    Time-window statuses arrive as key ranges; they are expanded with ``keys``,
    the dimension's ordered partition keys. A range whose endpoint is not a
    current key cannot be expanded, so it goes to ``unresolved`` rather than
    being dropped without notice.
    """
    typename = statuses.get("__typename")
    if typename == "DefaultPartitionStatuses":
        return {
            "materialized": list(statuses.get("materializedPartitions") or []),
            "failed": list(statuses.get("failedPartitions") or []),
            "materializing": list(statuses.get("materializingPartitions") or []),
        }
    if typename != "TimePartitionStatuses":
        raise RuntimeError(f"Dagster returned an unexpected partition status type {typename!r}.")

    positions = {key: index for index, key in enumerate(keys)}
    result: dict[str, list[str]] = {"materialized": [], "failed": [], "materializing": []}
    for status_range in statuses.get("ranges") or []:
        status = str(status_range.get("status", "")).lower()
        start_key = status_range.get("startKey")
        end_key = status_range.get("endKey")
        start = positions.get(start_key)
        end = positions.get(end_key)
        if status not in result or start is None or end is None:
            unresolved.append({"status": status, "start": start_key, "end": end_key})
            continue
        result[status].extend(keys[start : end + 1])
    return result


def _multi_partition_key(keys_by_dimension: Mapping[str, str]) -> str:
    # Dagster joins a multi-partition key's parts in dimension-name order.
    return "|".join(key for _, key in sorted(keys_by_dimension.items()))


def _partition_status_lists(
    node: Mapping[str, Any],
) -> tuple[list[str], dict[str, list[str]], list[dict[str, str]]]:
    """Return (all keys in order, keys by Dagster status, unresolved ranges) for an asset."""
    dimensions = {
        dimension["name"]: list(dimension.get("partitionKeys") or [])
        for dimension in node.get("partitionKeysByDimension") or []
    }
    statuses = node.get("assetPartitionStatuses") or {}
    unresolved: list[dict[str, str]] = []

    if statuses.get("__typename") != "MultiPartitionStatuses":
        keys = next(iter(dimensions.values()), [])
        return keys, _keys_by_status_1d(statuses, keys, unresolved), unresolved

    primary = statuses.get("primaryDimensionName")
    secondaries = [name for name in dimensions if name != primary]
    if primary not in dimensions or len(secondaries) != 1:
        raise RuntimeError(
            "Dagster returned multi-partition statuses that do not match the "
            f"asset's dimensions (primary {primary!r}, dimensions {sorted(dimensions)})."
        )
    secondary = secondaries[0]
    primary_keys = dimensions[primary]
    secondary_keys = dimensions[secondary]
    primary_positions = {key: index for index, key in enumerate(primary_keys)}

    def combine(primary_key: str, secondary_key: str) -> str:
        return _multi_partition_key({primary: primary_key, secondary: secondary_key})

    by_status: dict[str, list[str]] = {"materialized": [], "failed": [], "materializing": []}
    for status_range in statuses.get("ranges") or []:
        start_key = status_range.get("primaryDimStartKey")
        end_key = status_range.get("primaryDimEndKey")
        start = primary_positions.get(start_key)
        end = primary_positions.get(end_key)
        if start is None or end is None:
            unresolved.append({"status": "unknown", "start": start_key, "end": end_key})
            continue
        secondary_by_status = _keys_by_status_1d(
            status_range.get("secondaryDim") or {}, secondary_keys, unresolved
        )
        for primary_key in primary_keys[start : end + 1]:
            for status, keys in secondary_by_status.items():
                by_status[status].extend(combine(primary_key, key) for key in keys)

    all_keys = [combine(p, s) for p in primary_keys for s in secondary_keys]
    return all_keys, by_status, unresolved


@mcp.tool()
def get_asset_partition_status(
    asset_keys: list[str],
    statuses: list[str] | None = None,
    limit: int = 1000,
    env: str | None = None,
    failed_run_limit: int = 20,
) -> list[dict]:
    """List which partitions of partitioned assets failed, are running, are done, or are missing.

    This is the same per-partition view the Dagster UI shows on an asset's
    Partitions tab. Use it to find failed partitions before re-launching them.

    - asset_keys: one or more asset keys, '/'-separated for nested keys
      (e.g. ['raw/orders', 'daily_revenue'])
    - statuses: which partition lists to return. Any of 'FAILED',
      'MATERIALIZING', 'MATERIALIZED', 'MISSING' (case-insensitive).
      Default: ['FAILED', 'MATERIALIZING']. Counts for all four are always
      returned.
    - limit: max partition keys per list (default 1000)
    - failed_run_limit: for up to this many failed partitions per asset,
      also return the run that failed them (default 20; 0 turns it off).
      One call looks up at most 100 partitions in total

    Returns per asset:
    - asset_key, partitioned, partition_definition
    - counts: {total, failed, materializing, materialized, missing}
    - one list of partition keys per requested status (e.g. 'failed'), in
      partition order
    - truncated: names of lists cut at `limit` (only when a list was cut)
    - dimensions: dimension names, only for multi-partitioned assets. Their
      keys are joined with '|' in dimension-name order (e.g. '2024-01-01|eu')
    - unresolved_ranges: only when Dagster reported a range whose endpoints
      are not current partition keys, so its keys could not be listed
    - failed_runs: {partition, run_id, status} for the first failed
      partitions, when 'FAILED' is requested. Pass run_id to
      get_run_failure_summary to see why the partition failed. One run can
      cover many partitions. If the lookup fails, failed_runs_error has the
      reason and the partition lists are still complete

    Each partition has exactly one status. A partition that failed and is
    being retried counts as 'materializing'. 'failed' means the latest
    attempt failed and no later run has materialized it.

    Unpartitioned or unknown assets return a 'message' instead.

    To re-run failed partitions, pass an asset's 'failed' keys to
    backfill_assets(asset_keys=[that asset], partition_keys=...). Assets in
    one backfill must share a partition definition and partition keys, so
    launch one backfill per asset when their failed partitions differ.
    """
    if not asset_keys:
        raise ValueError("asset_keys must contain at least one asset key.")
    requested = [status.lower() for status in statuses or ["FAILED", "MATERIALIZING"]]
    invalid = [status for status in requested if status not in _PARTITION_STATUS_NAMES]
    if invalid:
        raise ValueError(
            "statuses must be any of FAILED, MATERIALIZING, MATERIALIZED, MISSING; "
            f"got {invalid!r}."
        )
    if limit < 0 or failed_run_limit < 0:
        raise ValueError("limit and failed_run_limit must not be negative.")

    data = gql(
        _ASSET_PARTITION_STATUS_QUERY,
        {"assetKeys": [_asset_key_input(key) for key in asset_keys]},
        env=env,
    )
    nodes_by_key: dict[str, dict] = {}
    for node in data.get("assetNodes") or []:
        # An asset defined in several code locations appears once per
        # location; its partition status is stored per asset key, so any copy
        # gives the same answer.
        nodes_by_key.setdefault(_asset_node_key(node), node)

    results = []
    for asset_key in dict.fromkeys(asset_keys):
        node = nodes_by_key.get(asset_key)
        if node is None:
            results.append({"asset_key": asset_key, "message": "Asset not found."})
            continue
        if not node.get("isPartitioned"):
            results.append(
                {
                    "asset_key": asset_key,
                    "partitioned": False,
                    "message": "Asset is not partitioned; use get_asset_health for its status.",
                }
            )
            continue

        all_keys, by_status, unresolved = _partition_status_lists(node)

        # Dagster can report one key under several statuses; keep the one the
        # UI shows, in the order MATERIALIZING > FAILED > MATERIALIZED.
        materializing = list(dict.fromkeys(by_status["materializing"]))
        claimed = set(materializing)
        failed = [key for key in dict.fromkeys(by_status["failed"]) if key not in claimed]
        claimed.update(failed)
        materialized = [
            key for key in dict.fromkeys(by_status["materialized"]) if key not in claimed
        ]
        claimed.update(materialized)
        missing = [key for key in all_keys if key not in claimed]

        order = {key: index for index, key in enumerate(all_keys)}
        lists = {
            name: sorted(keys, key=lambda key: order.get(key, len(order)))
            for name, keys in (
                ("failed", failed),
                ("materializing", materializing),
                ("materialized", materialized),
                ("missing", missing),
            )
        }

        entry: dict[str, Any] = {
            "asset_key": asset_key,
            "partitioned": True,
            "partition_definition": (node.get("partitionDefinition") or {}).get("description"),
            "counts": {"total": len(all_keys)} | {name: len(keys) for name, keys in lists.items()},
        }
        dimension_names = [d["name"] for d in node.get("partitionKeysByDimension") or []]
        if len(dimension_names) > 1:
            entry["dimensions"] = sorted(dimension_names)
        truncated = []
        for name in dict.fromkeys(requested):
            entry[name] = lists[name][:limit]
            if len(lists[name]) > limit:
                truncated.append(name)
        if truncated:
            entry["truncated"] = truncated
        if unresolved:
            entry["unresolved_ranges"] = unresolved
        results.append(entry)

    if "failed" in requested and failed_run_limit:
        _add_failed_partition_runs(results, failed_run_limit, env)
    return results


# Upper bound on latestRunForPartition lookups in one request. Each costs
# Dagster two storage reads, and the request shares gql's 30 s timeout.
_FAILED_RUN_LOOKUP_MAX: Final[int] = 100


def _add_failed_partition_runs(results: list[dict], per_asset: int, env: str | None) -> None:
    """Add the latest run of the first failed partitions to each entry, in one request."""
    targets = []
    budget = _FAILED_RUN_LOOKUP_MAX
    for entry in results:
        if entry.get("failed") and budget > 0:
            partitions = entry["failed"][: min(per_asset, budget)]
            budget -= len(partitions)
            targets.append((entry, partitions))
    if not targets:
        return
    definitions = []
    fields = []
    variables: dict[str, Any] = {}
    for asset_index, (entry, partitions) in enumerate(targets):
        definitions.append(f"$k{asset_index}: AssetKeyInput!")
        variables[f"k{asset_index}"] = _asset_key_input(entry["asset_key"])
        runs = []
        for partition_index, partition in enumerate(partitions):
            name = f"a{asset_index}p{partition_index}"
            definitions.append(f"${name}: String!")
            variables[name] = partition
            runs.append(f"{name}: latestRunForPartition(partition: ${name}) {{ runId status }}")
        fields.append(
            f"a{asset_index}: assetNodeOrError(assetKey: $k{asset_index}) "
            "{ __typename ... on AssetNode { " + " ".join(runs) + " } }"
        )
    query = (
        f"query PartitionRuns({', '.join(definitions)}) {{\n  "
        + "\n  ".join(fields)
        + "\n}"
    )
    try:
        data = gql(query, variables, env=env)
    except RuntimeError as exc:
        for entry, _ in targets:
            entry["failed_runs_error"] = f"Could not look up the failed runs: {exc}"
        return
    for asset_index, (entry, partitions) in enumerate(targets):
        node = data.get(f"a{asset_index}") or {}
        failed_runs = []
        for partition_index, partition in enumerate(partitions):
            run = node.get(f"a{asset_index}p{partition_index}") or {}
            failed_runs.append(
                {"partition": partition, "run_id": run.get("runId"), "status": run.get("status")}
            )
        entry["failed_runs"] = failed_runs


_ASSET_CHECKS_QUERY: Final[str] = """
query AssetChecks($assetKeys: [AssetKeyInput!]!) {
  assetNodes(assetKeys: $assetKeys) {
    assetKey { path }
    assetChecksOrError {
      __typename
      ... on AssetChecks { checks { name description blocking } }
      ... on Error { message }
    }
  }
}
"""

_CHECK_EXECUTION_FRAGMENT: Final[str] = (
    """
fragment CheckExecution on AssetCheckExecution {
  runId
  status
  timestamp
  partition
  evaluation {
    severity
    description
__METADATA_ENTRIES__
  }
}
"""
).replace("__METADATA_ENTRIES__", _METADATA_ENTRIES_FRAGMENT)


def _check_execution(execution: Mapping[str, Any]) -> dict:
    evaluation = execution.get("evaluation") or {}
    result = {
        "run_id": execution.get("runId"),
        "status": execution.get("status"),
        "timestamp": execution.get("timestamp"),
        "partition": execution.get("partition"),
    }
    if evaluation:
        result["severity"] = evaluation.get("severity")
        result["description"] = evaluation.get("description")
        result["metadata"] = _flatten_metadata(evaluation.get("metadataEntries"))
    return result


@mcp.tool()
def get_asset_check_results(
    asset_keys: list[str],
    check_name: str | None = None,
    limit: int = 1,
    env: str | None = None,
) -> list[dict]:
    """Get the results of the asset checks (data quality checks) defined on assets.

    - asset_keys: one or more asset keys, '/'-separated for nested keys
    - check_name: only this check (default: every check on each asset)
    - limit: executions to return per check, newest first (default 1, the
      latest result only)

    Returns per asset: asset_key and checks, a list with, per check:
    - check_name, description, blocking (a failed blocking check stops
      downstream assets in the same run)
    - status of the latest execution: SUCCEEDED, FAILED (the check ran and
      found a problem), EXECUTION_FAILED (the check itself raised an
      error), IN_PROGRESS, SKIPPED, or NOT_RUN (no execution yet)
    - executions: {run_id, status, timestamp (Unix seconds), partition,
      severity (ERROR or WARN), description, metadata} per execution

    Assets that are unknown, or whose checks Dagster cannot read, return a
    'message' instead. To see why a check could not execute, pass its run_id
    to get_run_failure_summary.
    """
    if not asset_keys:
        raise ValueError("asset_keys must contain at least one asset key.")
    if limit < 1:
        raise ValueError("limit must be at least 1.")

    data = gql(
        _ASSET_CHECKS_QUERY,
        {"assetKeys": [_asset_key_input(key) for key in asset_keys]},
        env=env,
    )
    nodes_by_key: dict[str, dict] = {}
    for node in data.get("assetNodes") or []:
        nodes_by_key.setdefault(_asset_node_key(node), node)

    results: list[dict] = []
    wanted: list[tuple[dict, str]] = []  # (check entry, asset key)
    for asset_key in dict.fromkeys(asset_keys):
        node = nodes_by_key.get(asset_key)
        if node is None:
            results.append({"asset_key": asset_key, "message": "Asset not found."})
            continue
        checks_or_error = node.get("assetChecksOrError") or {}
        if checks_or_error.get("__typename") != "AssetChecks":
            results.append(
                {
                    "asset_key": asset_key,
                    "message": checks_or_error.get("message")
                    or f"Dagster cannot read asset checks ({checks_or_error.get('__typename')}).",
                }
            )
            continue
        checks = [
            {
                "check_name": check["name"],
                "description": check.get("description"),
                "blocking": check.get("blocking"),
            }
            for check in checks_or_error.get("checks") or []
            if check_name is None or check["name"] == check_name
        ]
        entry: dict[str, Any] = {"asset_key": asset_key, "checks": checks}
        if check_name is not None and not checks:
            entry["message"] = f"Asset has no check named {check_name!r}."
        results.append(entry)
        wanted.extend((check, asset_key) for check in checks)

    if not wanted:
        return results

    definitions = ["$limit: Int!"]
    fields = []
    variables: dict[str, Any] = {"limit": limit}
    for index, (check, asset_key) in enumerate(wanted):
        definitions += [f"$k{index}: AssetKeyInput!", f"$c{index}: String!"]
        variables[f"k{index}"] = _asset_key_input(asset_key)
        variables[f"c{index}"] = check["check_name"]
        fields.append(
            f"e{index}: assetCheckExecutions(assetKey: $k{index}, checkName: $c{index}, "
            "limit: $limit) { ...CheckExecution }"
        )
    query = (
        f"query AssetCheckExecutions({', '.join(definitions)}) {{\n  "
        + "\n  ".join(fields)
        + "\n}"
        + _CHECK_EXECUTION_FRAGMENT
    )
    executions = gql(query, variables, env=env)
    for index, (check, _) in enumerate(wanted):
        history = [_check_execution(e) for e in executions.get(f"e{index}") or []]
        check["status"] = history[0]["status"] if history else "NOT_RUN"
        check["executions"] = history
    return results


# ── Automation ────────────────────────────────────────────────────────────────

# Sensors of these types evaluate automation conditions. Dagster's own
# default_automation_condition_sensor reports AUTO_MATERIALIZE.
_AUTOMATION_SENSOR_TYPES: Final[frozenset[str]] = frozenset({"AUTOMATION", "AUTO_MATERIALIZE"})
_CONDITION_LABEL_MAX: Final[int] = 300
_SINCE_TIMES: Final = (("last trigger", "triggerTimestamp"), ("last reset", "resetTimestamp"))

_ASSET_AUTOMATION_QUERY: Final[str] = """
query AssetAutomation($assetKey: AssetKeyInput!, $limit: Int!) {
  assetNodeOrError(assetKey: $assetKey) {
    __typename
    ... on AssetNode {
      isPartitioned
      automationCondition { label expandedLabel }
      currentAutoMaterializeEvaluationId
      targetingInstigators {
        __typename
        ... on Sensor {
          name
          sensorType
          sensorState {
            status
            ticks(limit: 1) { status timestamp skipReason error { message } }
          }
        }
        ... on Schedule {
          name
          cronSchedule
          executionTimezone
          scheduleState { status }
        }
      }
    }
    ... on AssetNotFoundError { message }
  }
  assetConditionEvaluationRecordsOrError(assetKey: $assetKey, limit: $limit) {
    __typename
    ... on AssetConditionEvaluationRecords {
      records {
        evaluationId
        timestamp
        numRequested
        runIds
        rootUniqueId
        evaluationNodes {
          uniqueId
          userLabel
          expandedLabel
          numTrue
          numCandidates
          isPartitioned
          childUniqueIds
          operatorType
          sinceMetadata { triggerTimestamp resetTimestamp }
        }
      }
    }
    ... on AutoMaterializeAssetEvaluationNeedsMigrationError { message }
  }
  instance {
    autoMaterializePaused
    useAutoMaterializeSensors
    daemonHealth { allDaemonStatuses { daemonType required healthy } }
  }
}
"""


def _iso_utc(timestamp: float | None) -> str | None:
    if timestamp is None:
        return None
    return datetime.fromtimestamp(timestamp, UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _condition_label(node: Mapping[str, Any]) -> str:
    label = node.get("userLabel") or " ".join(node.get("expandedLabel") or []) or "?"
    if len(label) > _CONDITION_LABEL_MAX:
        label = label[: _CONDITION_LABEL_MAX - 3] + "..."
    return label


def _condition_state(node: Mapping[str, Any]) -> str:
    """TRUE, FALSE, or SKIPPED (not evaluated); for partitions, 'TRUE for n [of m] partitions'."""
    true = node.get("numTrue") or 0
    candidates = node.get("numCandidates")
    if candidates == 0:
        return "SKIPPED"
    if not true:
        return "FALSE"
    if not node.get("isPartitioned"):
        return "TRUE"
    of = f" of {candidates}" if candidates is not None else ""
    return f"TRUE for {true}{of} partition{'' if (candidates or true) == 1 else 's'}"


def _condition_tree(nodes: Mapping[str, Mapping[str, Any]], root_id: str) -> list[str]:
    """Render an evaluation as indented lines: '<state>  <condition>'."""
    lines: list[str] = []

    def walk(node_id: str, depth: int, path: frozenset[str]) -> None:
        node = nodes.get(node_id)
        if node is None or node_id in path:
            return
        line = f"{'  ' * depth}{_condition_state(node)}  {_condition_label(node)}"
        since = node.get("sinceMetadata") or {}
        times = [
            f"{name} {_iso_utc(since[key])}"
            for name, key in _SINCE_TIMES
            if since.get(key)
        ]
        if times:
            line += f"  [{', '.join(times)}]"
        lines.append(line)
        for child in node.get("childUniqueIds") or []:
            walk(child, depth + 1, path | {node_id})

    walk(root_id, 0, frozenset())
    return lines


def _blocking_conditions(
    nodes: Mapping[str, Mapping[str, Any]], node_id: str, path: frozenset[str] = frozenset()
) -> list[str]:
    """Name the conditions that kept a false condition from being true.

    For AND, the false children. For OR, the blockers of every branch, because
    all branches were false. Other conditions, such as NOT and SINCE, name
    themselves.
    """
    node = nodes.get(node_id)
    if node is None or node_id in path or node.get("numTrue") or node.get("numCandidates") == 0:
        return []
    child_ids = [child for child in node.get("childUniqueIds") or [] if child in nodes]
    if node.get("operatorType") in ("and", "or"):
        found = [
            label
            for child in child_ids
            for label in _blocking_conditions(nodes, child, path | {node_id})
        ]
        if found:
            return found
        if node.get("operatorType") == "and":
            # Each child was true for some partitions, but never all for the same one.
            partial = [
                f"{_condition_label(nodes[child])} ({_condition_state(nodes[child])})"
                for child in child_ids
                if nodes[child].get("numCandidates") is not None
                and (nodes[child].get("numTrue") or 0) < nodes[child]["numCandidates"]
            ]
            if partial:
                return partial
    return [_condition_label(node)]


def _evaluation_summary(record: Mapping[str, Any]) -> dict:
    nodes = {node["uniqueId"]: node for node in record.get("evaluationNodes") or []}
    root = nodes.get(record.get("rootUniqueId") or "")
    return {
        "evaluation_id": record.get("evaluationId"),
        "timestamp": record.get("timestamp"),
        "result": _condition_state(root) if root else None,
        "requested": record.get("numRequested") or 0,
        "run_ids": record.get("runIds") or [],
    }


@mcp.tool()
def get_asset_automation(asset_key: str, limit: int = 5, env: str | None = None) -> dict:
    """Explain whether and why Dagster's automation starts runs for an asset.

    Use this when an asset with an automation condition (for example eager or
    on_cron) did not run, or ran when you did not expect it.

    Parameters:
    - asset_key: '/'-separated for nested keys
    - limit: evaluation records to return, newest first (default 5, max 50)

    Returns:
    - automation_condition: the condition's label, or None
    - automation_sensors: the sensors that evaluate the condition, with
      status and last tick {status, timestamp, skip_reason, error}
    - other_instigators: schedules and other sensors that target the asset
    - issues: problems that stop automation, for example a stopped sensor
    - latest_evaluation: the newest record, with:
      - requested: runs or partitions requested; run_ids
      - blocking_conditions: the false conditions that stopped a request
      - condition_tree: one line per condition, '<state>  <condition>',
        indented by nesting. The state is TRUE, FALSE, or SKIPPED (Dagster
        did not evaluate it, because an earlier condition was false). For a
        partitioned asset, TRUE says for how many partitions.
    - history: {evaluation_id, timestamp, result, requested, run_ids} per record

    Dagster stores a record only when the result changes or it requests a
    run, and deletes records after 30 days. A record that is older than the
    last sensor tick is still the current state. Timestamps are Unix
    seconds; the tree uses UTC.
    """
    if not 1 <= limit <= 50:
        raise ValueError("limit must be from 1 to 50.")
    data = gql(
        _ASSET_AUTOMATION_QUERY,
        {"assetKey": _asset_key_input(asset_key), "limit": limit},
        env=env,
    )
    node = data.get("assetNodeOrError") or {}
    if node.get("__typename") != "AssetNode":
        raise RuntimeError(node.get("message") or f"Asset {asset_key!r} not found.")

    condition = node.get("automationCondition")
    automation_sensors = []
    other_instigators = []
    for instigator in node.get("targetingInstigators") or []:
        if instigator.get("__typename") == "Sensor":
            state = instigator.get("sensorState") or {}
            tick = (state.get("ticks") or [None])[0]
            sensor = {
                "name": instigator.get("name"),
                "type": instigator.get("sensorType"),
                "status": state.get("status"),
            }
            if instigator.get("sensorType") in _AUTOMATION_SENSOR_TYPES:
                sensor["last_tick"] = tick and {
                    "status": tick.get("status"),
                    "timestamp": tick.get("timestamp"),
                    "skip_reason": tick.get("skipReason"),
                    "error": (tick.get("error") or {}).get("message"),
                }
                automation_sensors.append(sensor)
            else:
                other_instigators.append({"kind": "sensor", **sensor})
        elif instigator.get("__typename") == "Schedule":
            other_instigators.append(
                {
                    "kind": "schedule",
                    "name": instigator.get("name"),
                    "cron": instigator.get("cronSchedule"),
                    "timezone": instigator.get("executionTimezone"),
                    "status": (instigator.get("scheduleState") or {}).get("status"),
                }
            )

    result: dict[str, Any] = {
        "asset_key": asset_key,
        "automation_condition": condition.get("label")
        or " ".join(condition.get("expandedLabel") or [])
        if condition
        else None,
        "automation_sensors": automation_sensors,
        "other_instigators": other_instigators,
    }
    issues: list[str] = []
    if not condition:
        issues.append(
            "The asset has no automation condition, so automation does not start runs for it."
        )
        result["issues"] = issues
        return result

    instance = data.get("instance") or {}
    # AUTO_MATERIALIZE sensors and the legacy daemon run in the ASSET daemon;
    # AUTOMATION sensors run in user code, through the SENSOR daemon.
    daemon_types: set[str] = set()
    if instance.get("useAutoMaterializeSensors", True):
        if not automation_sensors:
            issues.append("No automation sensor targets this asset.")
        for sensor in automation_sensors:
            daemon_types.add("SENSOR" if sensor["type"] == "AUTOMATION" else "ASSET")
            if sensor["status"] != "RUNNING":
                issues.append(
                    f"Automation sensor {sensor['name']!r} is {sensor['status']}, so Dagster "
                    "does not evaluate this asset's condition."
                )
            tick = sensor.get("last_tick") or {}
            if tick.get("status") == "FAILURE":
                error = tick.get("error") or "no message"
                issues.append(f"The last tick of {sensor['name']!r} failed: {error}")
    else:
        daemon_types.add("ASSET")
        if instance.get("autoMaterializePaused"):
            issues.append("Auto-materialization is paused on this instance.")
    daemons = (instance.get("daemonHealth") or {}).get("allDaemonStatuses") or []
    for daemon in daemons:
        daemon_type = daemon.get("daemonType")
        if daemon_type in daemon_types and daemon.get("required") and not daemon.get("healthy"):
            issues.append(
                f"The {daemon_type} daemon, which evaluates this asset's automation, is not "
                "healthy. Check get_instance_status."
            )

    records_or_error = data.get("assetConditionEvaluationRecordsOrError") or {}
    records = records_or_error.get("records") or []
    if records_or_error.get("__typename") not in (None, "AssetConditionEvaluationRecords"):
        issues.append(records_or_error.get("message") or "Dagster cannot read evaluations.")
    elif not records:
        evaluated = node.get("currentAutoMaterializeEvaluationId") is not None or any(
            sensor.get("last_tick") for sensor in automation_sensors
        )
        if evaluated:
            # The asset daemon deletes records after 30 days (EVALUATIONS_TTL_DAYS).
            result["note"] = (
                "There is no evaluation record from the last 30 days. Dagster stores a record "
                "only when the result changes, and deletes records after 30 days. So the "
                "result did not change in that time, or the asset is new and Dagster did not "
                "evaluate it yet."
            )
        else:
            issues.append("Dagster has not evaluated this asset yet.")

    result["issues"] = issues
    result["latest_evaluation"] = None
    result["history"] = [_evaluation_summary(record) for record in records]
    if records:
        latest = records[0]
        nodes = {n["uniqueId"]: n for n in latest.get("evaluationNodes") or []}
        root_id = latest.get("rootUniqueId") or ""
        entry = result["history"][0].copy()
        if not entry["requested"]:
            entry["blocking_conditions"] = _blocking_conditions(nodes, root_id)
        entry["condition_tree"] = _condition_tree(nodes, root_id)
        current = node.get("currentAutoMaterializeEvaluationId")
        try:
            newer = current is not None and int(current) > int(latest.get("evaluationId"))
        except (TypeError, ValueError):
            newer = False
        if newer:
            entry["note"] = (
                f"Dagster evaluated the asset again up to evaluation {current}. Those "
                "evaluations stored no record, so their result was the same."
            )
        result["latest_evaluation"] = entry
    return result


# ── Jobs & Schedules & Sensors ────────────────────────────────────────────────


class JobInfo(TypedDict):
    """Public job metadata returned by :func:`list_jobs`."""

    repository: str
    location: str
    job: str
    description: str


_LIST_JOBS_QUERY: Final[str] = """
query ListJobs {
  repositoriesOrError {
    __typename
    ... on RepositoryConnection {
      nodes {
        name
        location { name }
        jobs {
          name
          description
        }
      }
    }
    ... on RepositoryNotFoundError { message }
    ... on PythonError { message }
  }
}
"""

# Rides along in the same request as repositoriesOrError so a failed or
# still-loading code location can be told apart from "no such repository"
# at no extra round-trip.
_WORKSPACE_LOAD_STATUS_SELECTION: Final[str] = """  workspaceOrError {
    __typename
    ... on Workspace {
      locationEntries {
        name
        loadStatus
        locationOrLoadError {
          __typename
          ... on PythonError { message }
        }
      }
    }
    ... on PythonError { message }
  }"""

_LIST_JOB_REPOSITORIES_QUERY: Final[str] = (
    """
query ListJobRepositories {
  repositoriesOrError {
    __typename
    ... on RepositoryConnection {
      nodes {
        name
        location { name }
      }
    }
    ... on RepositoryNotFoundError { message }
    ... on PythonError { message }
  }
"""
    + _WORKSPACE_LOAD_STATUS_SELECTION
    + """
}
"""
)

_LIST_REPOSITORY_JOBS_QUERY: Final[str] = (
    """
query ListRepositoryJobs(%s) {
%s
"""
    + _WORKSPACE_LOAD_STATUS_SELECTION
    + """
}
fragment RepositoryJobsConnection on RepositoryConnection {
  nodes {
    name
    location { name }
    jobs {
      name
      description
    }
  }
}
"""
)


def _jobs_from_repositories(
    repositories: Sequence[Mapping[str, Any]],
) -> list[JobInfo]:
    """Flatten repository job metadata into the public list_jobs schema."""
    return [
        JobInfo(
            repository=repo["name"],
            location=repo["location"]["name"],
            job=job["name"],
            description=job.get("description") or "",
        )
        for repo in repositories
        for job in repo.get("jobs", [])
    ]


def _location_unavailable_reason(entry: object) -> tuple[str, str] | None:
    """Return ``(name, reason)`` when a workspace location entry is unavailable.

    A location is healthy if either signal says so: its ``locationOrLoadError``
    resolves to a ``RepositoryLocation``, or its ``loadStatus`` is ``LOADED``.
    ``None`` also covers "this entry is too malformed to say anything about" —
    the caller records nothing either way.
    """
    if not isinstance(entry, Mapping):
        return None
    name = entry.get("name")
    if not isinstance(name, str):
        return None

    location_or_error = entry.get("locationOrLoadError")
    typename = (
        location_or_error.get("__typename")
        if isinstance(location_or_error, Mapping)
        else None
    )
    if typename == "RepositoryLocation":
        return None

    if typename == "PythonError":
        message = location_or_error.get("message")
        if not isinstance(message, str) or not message:
            message = "Dagster did not provide an error message"
        return name, message

    load_status = entry.get("loadStatus")
    if load_status == "LOADED":
        # Loaded, and not a PythonError: treat as available. An added or
        # renamed locationOrLoadError member must not read as an outage —
        # that would raise for every healthy location.
        return None
    status = load_status if isinstance(load_status, str) else "UNKNOWN"
    return name, f"code location is still loading (loadStatus={status})"


def _unavailable_code_locations(data: Mapping[str, Any]) -> dict[str, str]:
    """Map code-location name -> why it is unavailable, for locations Dagster cannot serve.

    Diagnostic data only: any non-``Workspace`` result, or malformed shape,
    yields ``{}`` ("nothing known") rather than raising, so this can never
    turn a working call into a failure.
    """
    workspace = data.get("workspaceOrError")
    if not isinstance(workspace, Mapping) or workspace.get("__typename") != "Workspace":
        return {}

    entries = workspace.get("locationEntries")
    if not isinstance(entries, list):
        return {}

    unavailable: dict[str, str] = {}
    for entry in entries:
        resolved = _location_unavailable_reason(entry)
        if resolved is not None:
            name, reason = resolved
            unavailable[name] = reason
    return unavailable


def _workspace_location_entries(
    response: object, context: str
) -> list[dict[str, Any]]:
    """Decode and validate a Dagster ``WorkspaceOrError`` union result."""
    if not isinstance(response, Mapping):
        raise RuntimeError(
            f"Malformed Dagster workspaceOrError response while {context}: "
            "expected a union result"
        )

    typename = response.get("__typename")
    if typename == "Workspace":
        entries = response.get("locationEntries")
        if not isinstance(entries, list) or any(
            not isinstance(entry, dict) for entry in entries
        ):
            raise RuntimeError(
                f"Malformed Dagster Workspace while {context}: "
                "expected locationEntries to be a list of code locations"
            )
        return entries

    if typename == "PythonError":
        message = response.get("message")
        if not isinstance(message, str):
            message = "No error message was provided"
        raise RuntimeError(f"Dagster failed while {context}: {message}")

    raise RuntimeError(
        f"Unexpected Dagster workspaceOrError typename {typename!r} while {context}"
    )


def _classify_launch_result(
    data: Mapping[str, Any], field: str, context: str
) -> tuple[dict[str, Any] | None, str | None]:
    """Classify a launch mutation union as success or failure.

    Dagster answers a launch mutation with a union. A failure such as
    ``PipelineNotFoundError`` is a *successful* GraphQL response — ``gql`` sees
    no ``errors`` key and returns normally — so a selection set that omits that
    member decodes to ``{}`` and the caller reports a launch that never
    happened. Selecting ``__typename`` is what makes that detectable, including
    for union members the selection does not name.

    Returns ``(payload, None)`` when the run launched, or ``(None, message)``
    when it did not. Callers decide whether to raise or to return the message
    alongside their own context.
    """
    response = data.get(field)
    if not isinstance(response, Mapping):
        return None, (
            f"Dagster returned no {field} payload while {context}. "
            "The job, repository, or code location name is probably wrong."
        )

    typename = response.get("__typename")
    if typename in ("LaunchRunSuccess", "LaunchBackfillSuccess"):
        return dict(response), None

    if typename == "RunConfigValidationInvalid":
        errors = response.get("errors")
        details = "; ".join(
            e.get("message", str(e)) for e in errors if isinstance(e, Mapping)
        ) if isinstance(errors, list) else ""
        return None, (
            f"Dagster rejected the run config while {context}: "
            f"{details or 'No error message was provided'}"
        )

    if typename is None:
        return None, (
            f"Dagster returned an unrecognized {field} result while {context}. "
            "This usually means the selector matched no job — check "
            "repository_name (it is NOT always '__repository__'), "
            "repository_location, and job_name against list_jobs."
        )

    message = response.get("message")
    if not isinstance(message, str):
        # InvalidStepError is the one selected member without a message field.
        step_key = response.get("invalidStepKey")
        if isinstance(step_key, str):
            message = f"invalid step key {step_key!r}"
        else:
            message = "No error message was provided"
    return None, f"Dagster failed while {context} ({typename}): {message}"


def _unwrap_launch_result(
    data: Mapping[str, Any], field: str, context: str
) -> dict[str, Any]:
    """Decode a launch mutation union, raising on anything but success."""
    payload, error = _classify_launch_result(data, field, context)
    if error is not None:
        raise RuntimeError(error)
    return payload or {}


def _raise_for_unavailable_code_locations(
    data: Mapping[str, Any],
    context: str,
    *,
    location_name: str | None = None,
) -> None:
    """Refuse a workspace search that could omit a relevant code location.

    ``workspaceOrError`` is diagnostic data added as a sibling field to the
    repository query. When present, its union and entries are decoded strictly.
    When ``location_name`` is supplied, unrelated unavailable locations cannot
    affect the targeted lookup and therefore do not block it.
    """
    if "workspaceOrError" not in data:
        return

    entries = _workspace_location_entries(data.get("workspaceOrError"), context)
    unavailable = {}
    for entry in entries:
        resolved = _location_unavailable_reason(entry)
        if resolved is not None:
            name, reason = resolved
            unavailable[name] = reason

    if location_name is not None:
        unavailable = {
            name: reason
            for name, reason in unavailable.items()
            if name == location_name
        }
    if not unavailable:
        return

    locations = ", ".join(
        f"{name!r} ({reason})" for name, reason in sorted(unavailable.items())
    )
    raise RuntimeError(
        f"Dagster cannot safely complete {context} because the following code "
        f"locations are unavailable: {locations}. Pass location_name to target "
        "a specific code location (and repository_name if needed), or check "
        "list_code_locations for details."
    )


def _repository_nodes(
    response: object,
    context: str,
    *,
    not_found_message: str | None = None,
) -> list[dict[str, Any]]:
    """Decode and validate a Dagster ``RepositoriesOrError`` union result.

    ``not_found_message``, when given, is raised instead of the default
    "no match" empty list on ``RepositoryNotFoundError`` — used when a
    relevant code location is known to be unavailable, so callers don't
    have to duplicate that raise themselves.
    """
    if not isinstance(response, Mapping):
        raise RuntimeError(
            f"Malformed Dagster repositoriesOrError response while {context}: "
            "expected a union result"
        )

    typename = response.get("__typename")
    if typename == "RepositoryConnection":
        nodes = response.get("nodes")
        if not isinstance(nodes, list) or any(
            not isinstance(node, dict) for node in nodes
        ):
            raise RuntimeError(
                f"Malformed Dagster RepositoryConnection while {context}: "
                "expected nodes to be a list of repositories"
            )
        return nodes

    if typename == "RepositoryNotFoundError":
        if not_found_message is not None:
            raise RuntimeError(not_found_message)
        return []

    if typename == "PythonError":
        message = response.get("message")
        if not isinstance(message, str):
            message = "No error message was provided"
        raise RuntimeError(f"Dagster failed while {context}: {message}")

    raise RuntimeError(
        f"Unexpected Dagster repositoriesOrError typename {typename!r} while {context}"
    )


def _repository_jobs(
    selectors: Sequence[Mapping[str, str]],
    env: str | None,
    *,
    discovered: bool = False,
) -> list[JobInfo]:
    """Fetch jobs for each repository selector in one batched request.

    ``discovered`` marks selectors sourced from a successful discovery
    request moments earlier: a ``RepositoryNotFoundError`` there means the
    code location went away mid-call, so it always raises. Otherwise a
    ``RepositoryNotFoundError`` only raises when ``workspaceOrError`` (fetched
    in the same request) shows that selector's code location is unavailable —
    a load failure or still-loading location must not silently look like "no
    such repository".
    """
    variable_definitions = ", ".join(
        f"$repositorySelector{index}: RepositorySelector!"
        for index in range(len(selectors))
    )
    repository_fields = "\n".join(
        f"""  repository{index}: repositoriesOrError(
    repositorySelector: $repositorySelector{index}
  ) {{
    __typename
    ...RepositoryJobsConnection
    ... on RepositoryNotFoundError {{ message }}
    ... on PythonError {{ message }}
  }}"""
        for index in range(len(selectors))
    )
    query = _LIST_REPOSITORY_JOBS_QUERY % (
        variable_definitions,
        repository_fields,
    )
    variables = {
        f"repositorySelector{index}": dict(selector)
        for index, selector in enumerate(selectors)
    }
    data = gql(query, variables, env=env)
    unavailable = _unavailable_code_locations(data)

    result: list[JobInfo] = []
    for index, selector in enumerate(selectors):
        repository_name = selector["repositoryName"]
        location_name = selector["repositoryLocationName"]
        context = (
            f"loading repository {repository_name!r} at location {location_name!r}"
        )
        if discovered:
            not_found_message = (
                f"Repository {repository_name!r} at code location "
                f"{location_name!r} was found during discovery moments earlier "
                "but is no longer available. The code location may have "
                "reloaded or failed between requests — check list_code_locations."
            )
        elif location_name in unavailable:
            not_found_message = (
                f"Code location {location_name!r} is unavailable "
                f"({unavailable[location_name]}); list_jobs cannot confirm "
                f"whether repository {repository_name!r} exists there. "
                "Check list_code_locations for details."
            )
        else:
            not_found_message = None

        repositories = _repository_nodes(
            data.get(f"repository{index}"),
            context,
            not_found_message=not_found_message,
        )
        result.extend(_jobs_from_repositories(repositories))
    return result


@mcp.tool()
def list_jobs(
    env: str | None = None,
    repository_name: str | None = None,
    location_name: str | None = None,
) -> list[JobInfo]:
    """List jobs across code locations, optionally filtered by repository or location.

    Returns per job: repository name, code location name, job name, and description.
    repository_name and location_name are independent exact-match filters; when
    both are supplied, a job must match both.

    An empty result when filtering means no match was found among loaded code
    locations. If a code location relevant to the filter failed to load or is
    still loading, list_jobs raises instead of silently reporting no jobs —
    check list_code_locations for details. Calling with no filters lists
    whatever is loaded and never raises for a broken location; use
    list_code_locations or get_instance_status to check load health directly.

    When to use: as a starting point to explore what jobs exist, or to find the
    exact job name and repository_location needed for launch_job.
    """
    if repository_name is not None and location_name is not None:
        return _repository_jobs(
            [
                {
                    "repositoryName": repository_name,
                    "repositoryLocationName": location_name,
                }
            ],
            env,
        )

    if repository_name is None and location_name is None:
        data = gql(_LIST_JOBS_QUERY, env=env)
        repositories = _repository_nodes(
            data.get("repositoriesOrError"), "listing jobs"
        )
        return _jobs_from_repositories(repositories)

    data = gql(_LIST_JOB_REPOSITORIES_QUERY, env=env)
    discovery_context = (
        f"discovering repositories for list_jobs with repository {repository_name!r}"
        if repository_name is not None
        else f"discovering repositories for list_jobs at location {location_name!r}"
    )
    repositories = _repository_nodes(data.get("repositoriesOrError"), discovery_context)
    matches = [
        repo
        for repo in repositories
        if (repository_name is None or repo["name"] == repository_name)
        and (location_name is None or repo["location"]["name"] == location_name)
    ]

    # Exactly one filter is set here: both-set and both-None returned above.
    unavailable = _unavailable_code_locations(data)
    if location_name is not None:
        if location_name in unavailable:
            raise RuntimeError(
                f"Code location {location_name!r} is unavailable "
                f"({unavailable[location_name]}); list_jobs cannot confirm whether "
                "it contains matching jobs. Check list_code_locations for details."
            )
    elif not matches and unavailable:
        locations_desc = ", ".join(
            f"{name!r} ({reason})" for name, reason in sorted(unavailable.items())
        )
        raise RuntimeError(
            f"No repository named {repository_name!r} was found among loaded "
            "code locations, but the following code locations are unavailable "
            f"and could not be searched: {locations_desc}. Repository "
            f"{repository_name!r} may exist there — check list_code_locations "
            "for details."
        )

    selectors = [
        {
            "repositoryName": repo["name"],
            "repositoryLocationName": repo["location"]["name"],
        }
        for repo in matches
    ]
    if not selectors:
        return []
    return _repository_jobs(selectors, env, discovered=True)


@mcp.tool()
def list_schedules(env: str | None = None) -> list[dict]:
    """List all schedules with their status, cron expression, target job, and next tick.

    Returns per schedule: name, cron expression, status (RUNNING/STOPPED),
    next_tick timestamp, target job name, repository, and code location.

    When to use: to check which schedules are active, verify cron timing,
    or find schedules that are stopped and might need attention.
    If a schedule is RUNNING but jobs aren't executing, use
    get_tick_history to inspect recent ticks for errors.
    Raises when a code location is unavailable rather than returning a partial
    list that could be mistaken for the complete set of schedules.
    """
    query = (
        """
    query ListSchedules {
      repositoriesOrError {
        __typename
        ... on RepositoryConnection {
          nodes {
            name
            location { name }
            schedules {
              name
              cronSchedule
              scheduleState { status }
              futureTicks(limit: 1) { results { timestamp } }
              pipelineName
            }
          }
        }
        ... on RepositoryNotFoundError { message }
        ... on PythonError { message }
      }
    """
        + _WORKSPACE_LOAD_STATUS_SELECTION
        + """
    }
    """
    )
    data = gql(query, env=env)
    repos = _repository_nodes(data.get("repositoriesOrError"), "listing schedules")
    _raise_for_unavailable_code_locations(data, "listing schedules")
    result = []
    for repo in repos:
        for sched in repo.get("schedules", []):
            next_ticks = sched.get("futureTicks", {}).get("results", [])
            result.append(
                {
                    "repository": repo["name"],
                    "location": repo["location"]["name"],
                    "schedule": sched["name"],
                    "cron": sched.get("cronSchedule"),
                    "status": sched.get("scheduleState", {}).get("status"),
                    "next_tick": next_ticks[0]["timestamp"] if next_ticks else None,
                    "job": sched.get("pipelineName"),
                }
            )
    return result


@mcp.tool()
def list_sensors(env: str | None = None) -> list[dict]:
    """List all sensors with their status and target jobs.

    Returns per sensor: name, status (RUNNING/STOPPED), list of target job names,
    repository, and code location.

    When to use: to check which sensors are active and what jobs they trigger.
    If a sensor is RUNNING but not producing runs, use get_tick_history to
    inspect recent ticks — it will show skipped ticks, errors, or runs launched.
    Raises when a code location is unavailable rather than returning a partial
    list that could be mistaken for the complete set of sensors.
    """
    query = (
        """
    query ListSensors {
      repositoriesOrError {
        __typename
        ... on RepositoryConnection {
          nodes {
            name
            location { name }
            sensors {
              name
              sensorState { status }
              targets { pipelineName }
            }
          }
        }
        ... on RepositoryNotFoundError { message }
        ... on PythonError { message }
      }
    """
        + _WORKSPACE_LOAD_STATUS_SELECTION
        + """
    }
    """
    )
    data = gql(query, env=env)
    repos = _repository_nodes(data.get("repositoriesOrError"), "listing sensors")
    _raise_for_unavailable_code_locations(data, "listing sensors")
    result = []
    for repo in repos:
        for sensor in repo.get("sensors", []):
            targets = [t["pipelineName"] for t in sensor.get("targets", [])]
            result.append(
                {
                    "repository": repo["name"],
                    "location": repo["location"]["name"],
                    "sensor": sensor["name"],
                    "status": sensor.get("sensorState", {}).get("status"),
                    "targets": targets,
                }
            )
    return result


def _normalize_instigator_type(value: str) -> str:
    """Upper-case and validate an instigator type ('SCHEDULE' or 'SENSOR')."""
    normalized = value.upper()
    if normalized not in ("SCHEDULE", "SENSOR"):
        raise ValueError("instigator_type must be 'SCHEDULE' or 'SENSOR'.")
    return normalized


def _locate_instigators(
    instigator_name: str,
    instigator_type: str,
    repository_name: str | None = None,
    location_name: str | None = None,
    env: str | None = None,
    include_state: bool = False,
) -> list[dict]:
    """Locate every schedule/sensor matching a name across all repositories.

    Returns a list of {"repositoryName", "repositoryLocationName", "name",
    "state"} where "state" is the InstigationState ({id, selectorId}) when
    include_state is set, else {}. Instigator names are only unique within a
    repository, so several code locations may expose the same name; callers
    must handle >1 match. repository_name / location_name narrow the search
    when given. instigator_type must already be upper-cased and validated.

    Resolving instigation state hits the instance storage for every schedule
    and sensor in the workspace, so only the stop tools — which need the
    origin/selector ids — ask for it.
    """
    state_fields = " scheduleState { id selectorId }" if include_state else ""
    sensor_state_fields = " sensorState { id selectorId }" if include_state else ""
    locate = (
        """
    query Locate {
      repositoriesOrError {
        __typename
        ... on RepositoryConnection {
          nodes {
            name
            location { name }
            schedules { name__SCHEDULE_STATE__ }
            sensors { name__SENSOR_STATE__ }
          }
        }
        ... on RepositoryNotFoundError { message }
        ... on PythonError { message }
      }
    """
        .replace("__SCHEDULE_STATE__", state_fields)
        .replace("__SENSOR_STATE__", sensor_state_fields)
        + _WORKSPACE_LOAD_STATUS_SELECTION
        + """
    }
    """
    )
    data = gql(locate, env=env)
    context = f"locating {instigator_type.lower()} {instigator_name!r}"
    repos = _repository_nodes(data.get("repositoriesOrError"), context)
    _raise_for_unavailable_code_locations(
        data, context, location_name=location_name
    )
    field = "schedules" if instigator_type == "SCHEDULE" else "sensors"
    state_key = "scheduleState" if instigator_type == "SCHEDULE" else "sensorState"
    matches = []
    for repo in repos:
        repo_name = repo["name"]
        loc_name = repo["location"]["name"]
        if repository_name is not None and repo_name != repository_name:
            continue
        if location_name is not None and loc_name != location_name:
            continue
        for item in repo.get(field, []):
            if item.get("name") == instigator_name:
                matches.append(
                    {
                        "repositoryName": repo_name,
                        "repositoryLocationName": loc_name,
                        "name": instigator_name,
                        "state": item.get(state_key) or {},
                    }
                )
    return matches


def _resolve_instigator(
    instigator_name: str,
    instigator_type: str,
    repository_name: str | None = None,
    location_name: str | None = None,
    env: str | None = None,
    include_state: bool = False,
) -> tuple[dict | None, dict | None]:
    """Resolve a name to exactly one instigator.

    Returns (located, None) on a unique match, or (None, error_dict) when the
    instigator is missing or ambiguous across code locations. Never mutates.
    Pass include_state when the caller needs the instigation-state ids.
    """
    matches = _locate_instigators(
        instigator_name,
        instigator_type,
        repository_name=repository_name,
        location_name=location_name,
        env=env,
        include_state=include_state,
    )
    kind = instigator_type.capitalize()
    if not matches:
        return None, {
            "name": instigator_name,
            "instigator_type": instigator_type,
            "message": f"{kind} '{instigator_name}' not found.",
        }
    if len(matches) > 1:
        candidates = [
            {
                "repository": m["repositoryName"],
                "location": m["repositoryLocationName"],
            }
            for m in matches
        ]
        listed = ", ".join(
            f"{c['location']}/{c['repository']}" for c in candidates
        )
        return None, {
            "name": instigator_name,
            "instigator_type": instigator_type,
            "candidates": candidates,
            "message": (
                f"Ambiguous: {kind.lower()} '{instigator_name}' exists in "
                f"{listed} — pass repository_name/location_name to "
                "disambiguate. No change was made."
            ),
        }
    return matches[0], None


# ``repository_name``/``location_name`` follow ``env`` to preserve the existing
# positional call signature.
@mcp.tool()
def get_tick_history(
    instigator_name: str,
    instigator_type: str,
    limit: int = 20,
    env: str | None = None,
    repository_name: str | None = None,
    location_name: str | None = None,
) -> dict:
    """Get recent tick history for a schedule or sensor — essential for detecting silent failures.

    - instigator_name: exact name of the schedule or sensor (from list_schedules/list_sensors)
    - instigator_type: 'SCHEDULE' or 'SENSOR'
    - limit: max ticks to return (default 20)
    - repository_name / location_name: optional, to disambiguate when the same
      name exists in several code locations (see list_schedules/list_sensors)

    Returns per tick: tick_id, status (SUCCESS/FAILURE/SKIPPED), timestamp,
    error message (if failed), and run_ids (runs launched by this tick).

    When to use: when a schedule or sensor is RUNNING but data is not being
    produced. Common patterns to look for:
    - All ticks SKIPPED: sensor condition not met, or misconfigured
    - Ticks with FAILURE status: the schedule/sensor code is erroring
    - Ticks with SUCCESS but empty run_ids: sensor evaluated but decided not to launch
    - Missing ticks: daemon may be unhealthy (check get_instance_status)
    """
    instigator_type = _normalize_instigator_type(instigator_type)

    # Resolve repo + location for the named instigator (selector needs all three).
    located, error = _resolve_instigator(
        instigator_name,
        instigator_type,
        repository_name=repository_name,
        location_name=location_name,
        env=env,
    )
    if error is not None:
        return error
    selector = {
        key: located[key]
        for key in ("repositoryName", "repositoryLocationName", "name")
    }

    query = """
    query TickHistory($selector: InstigationSelector!, $limit: Int!) {
      instigationStateOrError(instigationSelector: $selector) {
        __typename
        ... on InstigationState {
          ticks(limit: $limit) {
            tickId
            status
            timestamp
            error { message }
            runIds
          }
        }
        ... on InstigationStateNotFoundError { message }
        ... on PythonError { message }
      }
    }
    """
    state = gql(query, {"selector": selector, "limit": limit}, env=env).get(
        "instigationStateOrError", {}
    )
    if state.get("__typename") != "InstigationState":
        return {
            "name": instigator_name,
            "instigator_type": instigator_type,
            "message": state.get("message", "Unknown error"),
        }
    return {
        "name": instigator_name,
        "instigator_type": instigator_type,
        "ticks": [
            {
                "tick_id": t["tickId"],
                "status": t["status"],
                "timestamp": t["timestamp"],
                "error": t.get("error", {}).get("message") if t.get("error") else None,
                "run_ids": t.get("runIds", []),
            }
            for t in state.get("ticks", [])
        ],
    }


# ── Schedules & sensors (write) ───────────────────────────────────────────────


def _missing_state_ids_message(kind: str, name: str) -> str:
    return (
        f"Could not resolve instigation state ids for {kind.lower()} "
        f"'{name}'; this Dagster version may not expose "
        "InstigationState.id/selectorId."
    )


def start_schedule(
    schedule_name: str,
    repository_name: str | None = None,
    location_name: str | None = None,
    env: str | None = None,
) -> dict:
    """Start (enable) a Dagster schedule so it launches runs on its cron interval.

    - schedule_name: exact schedule name (from list_schedules)
    - repository_name / location_name: optional, to disambiguate when the same
      schedule name exists in several code locations
    - env: optional environment key; defaults to the configured instance

    Returns {name, instigator_type, repository, location, status} on success
    (status is the new InstigationStatus, normally RUNNING). On failure returns
    {name, instigator_type, message} — e.g. schedule not found, ambiguous
    across code locations (nothing is changed), or the API token lacks the
    START_SCHEDULE permission.

    When to use: to re-enable a schedule that was previously stopped. This is
    a persistent instance-level change that survives daemon restarts and code
    reloads. Ticks missed while the schedule was stopped are NOT backfilled —
    use backfill_assets or launch_job_with_partitions to catch up.
    """
    located, error = _resolve_instigator(
        schedule_name,
        "SCHEDULE",
        repository_name=repository_name,
        location_name=location_name,
        env=env,
    )
    if error is not None:
        return error

    selector = {
        "repositoryName": located["repositoryName"],
        "repositoryLocationName": located["repositoryLocationName"],
        "scheduleName": schedule_name,
    }
    query = """
    mutation StartSchedule($selector: ScheduleSelector!) {
      startSchedule(scheduleSelector: $selector) {
        __typename
        ... on ScheduleStateResult { scheduleState { id name status selectorId } }
        ... on Error { message }
      }
    }
    """
    result = gql(query, {"selector": selector}, env=env).get("startSchedule", {})
    typename = result.get("__typename")
    if typename == "ScheduleStateResult":
        state = result.get("scheduleState") or {}
        return {
            "name": schedule_name,
            "instigator_type": "SCHEDULE",
            "repository": located["repositoryName"],
            "location": located["repositoryLocationName"],
            "status": state.get("status"),
            "message": f"Schedule '{schedule_name}' started.",
        }
    return {
        "name": schedule_name,
        "instigator_type": "SCHEDULE",
        "repository": located["repositoryName"],
        "location": located["repositoryLocationName"],
        "message": result.get("message", f"Unknown error ({typename})."),
    }


def stop_schedule(
    schedule_name: str,
    repository_name: str | None = None,
    location_name: str | None = None,
    env: str | None = None,
) -> dict:
    """Stop (disable) a Dagster schedule so it stops launching runs on its cron.

    - schedule_name: exact schedule name (from list_schedules)
    - repository_name / location_name: optional, to disambiguate when the same
      schedule name exists in several code locations. Pass location_name (and
      repository_name if needed) to target a healthy location when another
      code location is unavailable
    - env: optional environment key; defaults to the configured instance

    Returns {name, instigator_type, repository, location, status} on success
    (status is the new InstigationStatus, normally STOPPED). On failure returns
    {name, instigator_type, message} — e.g. schedule not found, ambiguous
    across code locations (nothing is stopped), or the API token lacks the
    STOP_RUNNING_SCHEDULE permission.

    When to use: to halt a schedule that is launching bad or unwanted runs.
    This is a persistent instance-level change: the schedule stays stopped
    across daemon restarts and code reloads until someone calls
    start_schedule, and it overrides any default_status declared in code.
    It does NOT terminate runs the schedule already launched — use
    terminate_run for those. Cron ticks missed while stopped are never
    backfilled when the schedule is restarted.
    """
    located, error = _resolve_instigator(
        schedule_name,
        "SCHEDULE",
        repository_name=repository_name,
        location_name=location_name,
        env=env,
        include_state=True,
    )
    if error is not None:
        return error
    state = located["state"]
    if not state.get("id") or not state.get("selectorId"):
        return {
            "name": schedule_name,
            "instigator_type": "SCHEDULE",
            "repository": located["repositoryName"],
            "location": located["repositoryLocationName"],
            "message": _missing_state_ids_message("schedule", schedule_name),
        }

    query = """
    mutation StopSchedule($originId: String!, $selectorId: String!) {
      stopRunningSchedule(scheduleOriginId: $originId, scheduleSelectorId: $selectorId) {
        __typename
        ... on ScheduleStateResult { scheduleState { id name status selectorId } }
        ... on Error { message }
      }
    }
    """
    variables = {"originId": state["id"], "selectorId": state["selectorId"]}
    result = gql(query, variables, env=env).get("stopRunningSchedule", {})
    typename = result.get("__typename")
    if typename == "ScheduleStateResult":
        new_state = result.get("scheduleState") or {}
        return {
            "name": schedule_name,
            "instigator_type": "SCHEDULE",
            "repository": located["repositoryName"],
            "location": located["repositoryLocationName"],
            "status": new_state.get("status"),
            "message": f"Schedule '{schedule_name}' stopped.",
        }
    return {
        "name": schedule_name,
        "instigator_type": "SCHEDULE",
        "repository": located["repositoryName"],
        "location": located["repositoryLocationName"],
        "message": result.get("message", f"Unknown error ({typename})."),
    }


def start_sensor(
    sensor_name: str,
    repository_name: str | None = None,
    location_name: str | None = None,
    env: str | None = None,
) -> dict:
    """Start (enable) a Dagster sensor so it resumes evaluating and launching runs.

    - sensor_name: exact sensor name (from list_sensors)
    - repository_name / location_name: optional, to disambiguate when the same
      sensor name exists in several code locations
    - env: optional environment key; defaults to the configured instance

    Returns {name, instigator_type, repository, location, status} on success
    (status is the new InstigationStatus, normally RUNNING). On failure returns
    {name, instigator_type, message} — e.g. sensor not found, ambiguous across
    code locations (nothing is changed), or the API token lacks the
    EDIT_SENSOR permission.

    When to use: to re-enable a sensor that was previously stopped once the
    underlying issue is fixed. This is a persistent instance-level change that
    survives daemon restarts and code reloads. Use get_tick_history afterwards
    to confirm the sensor is ticking again.
    """
    located, error = _resolve_instigator(
        sensor_name,
        "SENSOR",
        repository_name=repository_name,
        location_name=location_name,
        env=env,
    )
    if error is not None:
        return error

    selector = {
        "repositoryName": located["repositoryName"],
        "repositoryLocationName": located["repositoryLocationName"],
        "sensorName": sensor_name,
    }
    query = """
    mutation StartSensor($selector: SensorSelector!) {
      startSensor(sensorSelector: $selector) {
        __typename
        ... on Sensor { name sensorState { id status selectorId } }
        ... on Error { message }
      }
    }
    """
    result = gql(query, {"selector": selector}, env=env).get("startSensor", {})
    typename = result.get("__typename")
    if typename == "Sensor":
        state = result.get("sensorState") or {}
        return {
            "name": sensor_name,
            "instigator_type": "SENSOR",
            "repository": located["repositoryName"],
            "location": located["repositoryLocationName"],
            "status": state.get("status"),
            "message": f"Sensor '{sensor_name}' started.",
        }
    return {
        "name": sensor_name,
        "instigator_type": "SENSOR",
        "repository": located["repositoryName"],
        "location": located["repositoryLocationName"],
        "message": result.get("message", f"Unknown error ({typename})."),
    }


def stop_sensor(
    sensor_name: str,
    repository_name: str | None = None,
    location_name: str | None = None,
    env: str | None = None,
) -> dict:
    """Stop a Dagster sensor so it stops evaluating and launching runs.

    - sensor_name: sensor name (from list_sensors or get_tick_history)
    - repository_name / location_name: optional, to disambiguate when the same
      sensor name exists in several code locations. Pass location_name (and
      repository_name if needed) to target a healthy location when another
      code location is unavailable
    - env: optional environment key; defaults to the configured instance

    Returns {name, instigator_type, repository, location, status} on success
    (status is the new InstigationStatus, normally STOPPED). On failure returns
    {name, instigator_type, message} — e.g. sensor not found, ambiguous across
    code locations (nothing is stopped), or the API token lacks the
    EDIT_SENSOR permission.

    When to use: to halt a runaway or erroring sensor found via
    get_tick_history or get_instance_status. This is a persistent
    instance-level change: the sensor stays stopped across daemon restarts and
    code reloads until someone calls start_sensor. It does NOT cancel runs the
    sensor already launched — use terminate_run for those. If the sensor
    declares a default_status in code, a stop recorded here overrides it.
    """
    located, error = _resolve_instigator(
        sensor_name,
        "SENSOR",
        repository_name=repository_name,
        location_name=location_name,
        env=env,
        include_state=True,
    )
    if error is not None:
        return error
    state = located["state"]
    if not state.get("id") or not state.get("selectorId"):
        return {
            "name": sensor_name,
            "instigator_type": "SENSOR",
            "repository": located["repositoryName"],
            "location": located["repositoryLocationName"],
            "message": _missing_state_ids_message("sensor", sensor_name),
        }

    query = """
    mutation StopSensor($originId: String!, $selectorId: String!) {
      stopSensor(jobOriginId: $originId, jobSelectorId: $selectorId) {
        __typename
        ... on StopSensorMutationResult { instigationState { id name status selectorId } }
        ... on Error { message }
      }
    }
    """
    variables = {"originId": state["id"], "selectorId": state["selectorId"]}
    result = gql(query, variables, env=env).get("stopSensor", {})
    typename = result.get("__typename")
    if typename == "StopSensorMutationResult":
        new_state = result.get("instigationState") or {}
        return {
            "name": sensor_name,
            "instigator_type": "SENSOR",
            "repository": located["repositoryName"],
            "location": located["repositoryLocationName"],
            "status": new_state.get("status"),
            "message": f"Sensor '{sensor_name}' stopped.",
        }
    return {
        "name": sensor_name,
        "instigator_type": "SENSOR",
        "repository": located["repositoryName"],
        "location": located["repositoryLocationName"],
        "message": result.get("message", f"Unknown error ({typename})."),
    }


# ── Code Locations ────────────────────────────────────────────────────────────


@mcp.tool()
def list_code_locations(env: str | None = None) -> list[dict]:
    """List all code locations and their load status.

    Returns per location: name, loadStatus (LOADED/LOADING), and either the
    repositories within it or a PythonError if loading failed.

    When to use: after a deployment to verify code locations loaded correctly,
    or when get_instance_status reports code location errors.
    If a location failed to load, use reload_code_location to retry.
    """
    query = """
    query CodeLocations {
      workspaceOrError {
        __typename
        ... on Workspace {
          locationEntries {
            name
            loadStatus
            locationOrLoadError {
              ... on RepositoryLocation {
                name
                repositories { name }
              }
              ... on PythonError { message }
            }
          }
        }
        ... on PythonError { message }
      }
    }
    """
    data = gql(query, env=env)
    return _workspace_location_entries(
        data.get("workspaceOrError"), "listing code locations"
    )


@mcp.tool()
def get_instance_status(env: str | None = None) -> dict:
    """Get a global health check of the Dagster instance. START HERE for any monitoring workflow.

    Returns:
    - healthy: boolean — true only if all required daemons are healthy AND
      no code locations have errors
    - daemons: list of {type, healthy, last_heartbeat, required} for each daemon
      (scheduler, sensor, run coordinator, backfill, etc.)
    - queued_runs_count: number of runs waiting in queue (high count = bottleneck)
    - max_concurrent_runs: the run queue's limit on runs in progress, or
      null when there is no limit
    - pools: per concurrency pool, {pool, limit, in_use, waiting_steps}.
      A pool with waiting steps and no free slots holds runs back even when
      the run queue has room
    - code_location_errors: list of {name, error} for locations that failed to load

    When to use: as the FIRST call in any diagnostic or monitoring flow.
    If healthy=false, check daemons for unhealthy entries and
    code_location_errors for loading failures. If runs stay queued, compare
    queued_runs_count with max_concurrent_runs and check pools.
    Follow up with list_code_locations or get_runs as needed.
    """
    query = """
    query InstanceStatus {
      instance {
        daemonHealth {
          allDaemonStatuses {
            daemonType
            required
            healthy
            lastHeartbeatTime
          }
        }
        runQueueConfig { maxConcurrentRuns }
        concurrencyLimits {
          concurrencyKey
          slotCount
          activeSlotCount
          pendingStepCount
        }
      }
      runsOrError(filter: {statuses: [QUEUED]}, limit: 1) {
        ... on Runs { count }
        ... on PythonError { message }
      }
      workspaceOrError {
        ... on Workspace {
          locationEntries {
            name
            loadStatus
            locationOrLoadError {
              ... on PythonError { message }
            }
          }
        }
      }
    }
    """
    data = gql(query, env=env)
    instance = data.get("instance") or {}

    daemons = [
        {
            "type": d["daemonType"],
            "healthy": d["healthy"],
            "last_heartbeat": d.get("lastHeartbeatTime"),
            "required": d["required"],
        }
        for d in (instance.get("daemonHealth") or {}).get("allDaemonStatuses", [])
    ]

    runs_or_error = data.get("runsOrError") or {}
    if "count" not in runs_or_error:
        raise RuntimeError(
            "Dagster could not count queued runs: "
            f"{runs_or_error.get('message', 'unknown error')}"
        )

    pools = [
        {
            "pool": pool["concurrencyKey"],
            "limit": pool["slotCount"],
            "in_use": pool["activeSlotCount"],
            "waiting_steps": pool["pendingStepCount"],
        }
        for pool in instance.get("concurrencyLimits") or []
    ]

    location_entries = (data.get("workspaceOrError") or {}).get("locationEntries", [])
    code_location_errors = []
    for loc in location_entries:
        err = loc.get("locationOrLoadError") or {}
        if "message" in err:
            code_location_errors.append({"name": loc["name"], "error": err["message"]})

    all_required_healthy = all(d["healthy"] for d in daemons if d["required"])
    healthy = all_required_healthy and len(code_location_errors) == 0
    # Dagster uses -1 for "no limit".
    max_runs = (instance.get("runQueueConfig") or {}).get("maxConcurrentRuns")

    return {
        "healthy": healthy,
        "daemons": daemons,
        "queued_runs_count": runs_or_error["count"],
        "max_concurrent_runs": max_runs if max_runs is not None and max_runs >= 0 else None,
        "pools": pools,
        "code_location_errors": code_location_errors,
    }


def reload_code_location(location_name: str, env: str | None = None) -> dict:
    """Reload a code location to pick up new code (e.g. after a deploy).

    - location_name: exact name of the code location (from list_code_locations)

    Returns the new load status. If the location is not found or reload
    is not supported, returns an error message.

    When to use: after deploying new code, or when list_code_locations shows
    a location in an error state. This is equivalent to clicking 'Reload'
    in the Dagster UI.
    """
    query = """
    mutation ReloadLocation($location: String!) {
      reloadRepositoryLocation(repositoryLocationName: $location) {
        ... on WorkspaceLocationEntry {
          name
          loadStatus
          locationOrLoadError {
            ... on RepositoryLocation { name }
            ... on PythonError { message }
          }
        }
        ... on ReloadNotSupported { message }
        ... on RepositoryLocationNotFound { message }
        ... on PythonError { message }
      }
    }
    """
    data = gql(query, {"location": location_name}, env=env)
    return data.get("reloadRepositoryLocation", {})


# ── Backfills ─────────────────────────────────────────────────────────────────


_BACKFILL_SUMMARY_FIELDS: Final[str] = """
    backfillId: id
    status
    isAssetBackfill
    title
    assetSelection { path }
    jobName
    partitionSetName
    numPartitions
    fromFailure
    timestamp
    endTimestamp
    error { message }"""


def _backfill_summary(backfill: Mapping[str, Any]) -> dict:
    summary = dict(backfill)
    summary["assetSelection"] = [
        "/".join(key["path"]) for key in backfill.get("assetSelection") or []
    ]
    summary["error"] = (backfill.get("error") or {}).get("message")
    return summary


@mcp.tool()
def list_backfills(
    limit: int = 10,
    env: str | None = None,
    statuses: list[str] | None = None,
    created_after: str | float | None = None,
    created_before: str | float | None = None,
    cursor: str | None = None,
) -> list[dict]:
    """List recent backfills, newest first, with their status.

    Returns per backfill: backfillId, status, isAssetBackfill, title,
    assetSelection (asset keys of an asset backfill), jobName and
    partitionSetName (of a job backfill), numPartitions, fromFailure,
    timestamp and endTimestamp (Unix seconds), and error (message, if it
    failed).

    - limit: max backfills to return (default 10)
    - statuses: only backfills with these statuses: REQUESTED (in
      progress), COMPLETED_SUCCESS, COMPLETED_FAILED, FAILING, FAILED,
      CANCELING, CANCELED
    - created_after / created_before: ISO 8601 strings or Unix seconds
    - cursor: the backfillId of the last backfill of the previous page

    When to use: to monitor in-progress backfills or review recent ones.
    Use get_backfill for which assets, partitions, and runs failed.
    """
    query = (
        """
    query Backfills($limit: Int!, $cursor: String, $filters: BulkActionsFilter) {
      partitionBackfillsOrError(cursor: $cursor, limit: $limit, filters: $filters) {
        __typename
        ... on PartitionBackfills {
          results {
__FIELDS__
          }
        }
        ... on PythonError { message }
      }
    }
    """
    ).replace("__FIELDS__", _BACKFILL_SUMMARY_FIELDS)
    filters: dict[str, Any] = {}
    if statuses:
        filters["statuses"] = [status.upper() for status in statuses]
    for name, field, value in (
        ("created_after", "createdAfter", created_after),
        ("created_before", "createdBefore", created_before),
    ):
        timestamp = _timestamp_input(value, name)
        if timestamp is not None:
            filters[field] = timestamp
    data = gql(
        query,
        {"limit": limit, "cursor": cursor, "filters": filters or None},
        env=env,
    )
    backfills = data.get("partitionBackfillsOrError") or {}
    if backfills.get("__typename") not in (None, "PartitionBackfills"):
        raise RuntimeError(
            f"Dagster could not list backfills: {backfills.get('message', 'unknown error')}"
        )
    return [_backfill_summary(b) for b in backfills.get("results") or []]


_GET_BACKFILL_QUERY: Final[str] = (
    """
query Backfill($backfillId: String!, $runFilter: RunsFilter!, $runLimit: Int!) {
  partitionBackfillOrError(backfillId: $backfillId) {
    __typename
    ... on PartitionBackfill {
__FIELDS__
      description
      user
      tags { key value }
      error { message className stack causes { message className stack } }
      assetBackfillData {
        assetBackfillStatuses {
          __typename
          ... on AssetPartitionsStatusCounts {
            assetKey { path }
            numPartitionsTargeted
            numPartitionsInProgress
            numPartitionsMaterialized
            numPartitionsFailed
          }
          ... on UnpartitionedAssetStatus {
            assetKey { path }
            inProgress
            materialized
            failed
          }
        }
        rootTargetedPartitions {
          partitionKeys
          ranges { start end }
        }
      }
      partitionStatuses {
        results { partitionName runId runStatus }
      }
    }
    ... on BackfillNotFoundError { message }
    ... on PythonError { message }
  }
  runsOrError(filter: $runFilter, limit: $runLimit) {
    __typename
    ... on Runs {
      count
      results { runId status startTime endTime tags { key value } }
    }
    ... on PythonError { message }
  }
}
"""
).replace("__FIELDS__", _BACKFILL_SUMMARY_FIELDS)

# Partition keys returned per list by get_backfill before it truncates.
_BACKFILL_PARTITION_LIST_LIMIT: Final[int] = 200


def _capped(keys: Sequence[str], limit: int = _BACKFILL_PARTITION_LIST_LIMIT) -> dict:
    result: dict[str, Any] = {"count": len(keys), "keys": list(keys[:limit])}
    if len(keys) > limit:
        result["truncated"] = True
    return result


@mcp.tool()
def get_backfill(backfill_id: str, run_limit: int = 20, env: str | None = None) -> dict:
    """Get one backfill in detail: status, error, per-asset progress, and its failed runs.

    - backfill_id: the backfill id (from list_backfills, backfill_assets, or
      launch_job_with_partitions)
    - run_limit: max failed or canceled runs to list (default 20)

    Returns the fields of list_backfills, plus:
    - error: message, class, stack trace, and causes when the backfill
      itself failed
    - assets (asset backfills): per asset, partitions targeted,
      materialized, failed, and in progress; or materialized / failed /
      in_progress flags for unpartitioned assets
    - targeted_partitions (asset backfills): the partition keys or ranges
      the backfill was asked to run
    - partitions (job backfills): partition keys grouped by the status of
      their run, e.g. {'FAILURE': {count, keys}, 'SUCCESS': {...}}
    - failed_runs: {run_id, status, partition or partition range} for the
      backfill's failed and canceled runs; failed_runs_total is the full count
    - tags, description, user

    When to use: after a backfill reports COMPLETED_FAILED or FAILED, to find
    which assets and partitions failed and which runs to inspect with
    get_run_failure_summary. To re-run only what failed, use retry_backfill.
    """
    data = gql(
        _GET_BACKFILL_QUERY,
        {
            "backfillId": backfill_id,
            "runFilter": {
                "tags": [{"key": _BACKFILL_TAG, "value": backfill_id}],
                "statuses": ["FAILURE", "CANCELED"],
            },
            "runLimit": run_limit,
        },
        env=env,
    )
    backfill = data.get("partitionBackfillOrError") or {}
    if backfill.get("__typename") != "PartitionBackfill":
        return {
            "backfill_id": backfill_id,
            "message": backfill.get("message", "Backfill not found."),
        }

    result = _backfill_summary(backfill)
    result.pop("__typename", None)
    result["error"] = _trim_error(backfill.get("error")) or None
    result["tags"] = {tag["key"]: tag["value"] for tag in backfill.get("tags") or []}
    for key in ("description", "user"):
        result[key] = backfill.get(key)
    result.pop("assetBackfillData", None)
    result.pop("partitionStatuses", None)

    asset_data = backfill.get("assetBackfillData") or {}
    if asset_data:
        assets = []
        for status in asset_data.get("assetBackfillStatuses") or []:
            key = "/".join(status["assetKey"]["path"])
            if status.get("__typename") == "AssetPartitionsStatusCounts":
                assets.append(
                    {
                        "asset_key": key,
                        "targeted": status["numPartitionsTargeted"],
                        "materialized": status["numPartitionsMaterialized"],
                        "failed": status["numPartitionsFailed"],
                        "in_progress": status["numPartitionsInProgress"],
                    }
                )
            else:
                assets.append(
                    {
                        "asset_key": key,
                        "materialized": status.get("materialized"),
                        "failed": status.get("failed"),
                        "in_progress": status.get("inProgress"),
                    }
                )
        result["assets"] = assets
        targeted = asset_data.get("rootTargetedPartitions") or {}
        if targeted.get("partitionKeys"):
            result["targeted_partitions"] = _capped(targeted["partitionKeys"])
        elif targeted.get("ranges"):
            result["targeted_partitions"] = {"ranges": targeted["ranges"]}

    statuses = (backfill.get("partitionStatuses") or {}).get("results")
    if statuses:
        by_status: dict[str, list[str]] = {}
        for partition in statuses:
            by_status.setdefault(partition.get("runStatus") or "NOT_STARTED", []).append(
                partition["partitionName"]
            )
        result["partitions"] = {status: _capped(keys) for status, keys in by_status.items()}

    runs = data.get("runsOrError") or {}
    if runs.get("__typename") not in (None, "Runs"):
        result["failed_runs"] = None
        result["failed_runs_error"] = runs.get("message", "Dagster could not list the runs.")
        return result
    result["failed_runs"] = [
        {"run_id": run["runId"], "status": run["status"], **(_run_partition(run) or {})}
        for run in runs.get("results") or []
    ]
    result["failed_runs_total"] = runs.get("count")
    return result


# ── Actions ───────────────────────────────────────────────────────────────────


def _get_materialization_asset_nodes(
    asset_keys: list[str],
    env: str | None = None,
) -> list[dict]:
    query = """
    query MaterializationAssetNodes($assetKeys: [AssetKeyInput!]!) {
      assetNodes(assetKeys: $assetKeys) {
        assetKey { path }
        groupName
        jobNames
        isMaterializable
        isExecutable
        isObservable
        isPartitioned
        repository {
          name
          location { name }
        }
        assetChecksOrError {
          __typename
          ... on AssetChecks {
            checks {
              name
              jobNames
            }
          }
        }
      }
    }
    """
    return gql(
        query,
        {"assetKeys": [_asset_key_input(key) for key in asset_keys]},
        env=env,
    ).get("assetNodes", [])


def _get_materialization_requirements(
    asset_keys: list[str],
    env: str | None = None,
) -> dict:
    query = """
    query MaterializationRequirements($assetKeys: [AssetKeyInput!]!) {
      assetNodeAdditionalRequiredKeys(assetKeys: $assetKeys) {
        path
      }
      assetNodeDefinitionCollisions(assetKeys: $assetKeys) {
        assetKey { path }
        repositories {
          name
          location { name }
        }
      }
    }
    """
    return gql(
        query,
        {"assetKeys": [_asset_key_input(key) for key in asset_keys]},
        env=env,
    )


def _collision_message(collisions: list[dict]) -> str:
    details = []
    for collision in collisions:
        key = "/".join(collision.get("assetKey", {}).get("path", []))
        repositories = ", ".join(
            f"{repository.get('location', {}).get('name', '?')}/{repository.get('name', '?')}"
            for repository in collision.get("repositories", [])
        )
        details.append(f"{key} ({repositories})")
    return "Selected assets have definition collisions: " + "; ".join(details)


def materialize_assets(
    asset_keys: list[str],
    run_config: dict | None = None,
    tags: dict[str, str] | None = None,
    env: str | None = None,
) -> dict:
    """Materialize concrete, unpartitioned asset keys with optional run configuration.

    Use resolve_asset_selection first when starting from a Dagster selection
    expression. This tool deliberately accepts only concrete slash-delimited
    asset keys, then re-fetches their current definitions before launching.

    The selected assets must be materializable, executable, unpartitioned,
    defined in one repository, and share a common asset job. Dagster-required
    neighbors from non-subsettable multi-assets are included automatically and
    reported in ``required_asset_keys_added``.

    For partitioned assets, pass the resolved keys to backfill_assets instead.
    """
    if not asset_keys:
        return {"message": "asset_keys must contain at least one asset key."}
    if any(not isinstance(key, str) or not key for key in asset_keys):
        return {"message": "Every asset key must be a non-empty string."}
    invalid_key_paths = sorted(
        {
            key
            for key in asset_keys
            if key.startswith("/")
            or key.endswith("/")
            or any(not segment for segment in key.split("/"))
            or "*" in key
        }
    )
    if invalid_key_paths:
        return {
            "message": (
                "materialize_assets accepts concrete slash-delimited asset keys, "
                "not selection expressions or empty path segments. Invalid keys: "
                + ", ".join(invalid_key_paths)
                + ". Use resolve_asset_selection first."
            )
        }

    requested_keys = sorted(set(asset_keys))
    launched_key_set = set(requested_keys)
    required_key_set: set[str] = set()
    # Expand to a fixed point because a newly added non-subsettable multi-asset
    # neighbor can introduce additional required neighbors of its own.
    while True:
        launched_keys = sorted(launched_key_set)
        nodes = _get_materialization_asset_nodes(launched_keys, env=env)
        nodes_by_key = {_asset_node_key(node): node for node in nodes}
        missing_keys = sorted(set(launched_keys) - set(nodes_by_key))
        if missing_keys:
            return {
                "asset_keys": launched_keys,
                "required_asset_keys_added": sorted(required_key_set),
                "message": f"Asset definitions were not found for: {', '.join(missing_keys)}.",
            }

        # Dagster's requirement and collision resolvers assume every key exists,
        # so missing definitions must be rejected before calling them.
        requirements = _get_materialization_requirements(launched_keys, env=env)
        collisions = requirements.get("assetNodeDefinitionCollisions", [])
        if collisions:
            return {
                "asset_keys": launched_keys,
                "required_asset_keys_added": sorted(required_key_set),
                "message": _collision_message(collisions),
            }
        additional_required_keys = {
            "/".join(asset_key.get("path", []))
            for asset_key in requirements.get("assetNodeAdditionalRequiredKeys", [])
        } - launched_key_set
        if not additional_required_keys:
            break
        required_key_set |= additional_required_keys
        launched_key_set |= additional_required_keys

    required_keys = sorted(required_key_set)

    invalid_assets: list[str] = []
    for key in launched_keys:
        node = nodes_by_key[key]
        if node.get("isPartitioned"):
            reason = "partitioned; use backfill_assets"
        elif node.get("isObservable"):
            reason = "observable, not materializable"
        elif not node.get("isMaterializable"):
            reason = "external or otherwise non-materializable"
        elif not node.get("isExecutable"):
            reason = "not executable"
        else:
            continue
        invalid_assets.append(f"{key} ({reason})")
    if invalid_assets:
        return {
            "asset_keys": launched_keys,
            "required_asset_keys_added": required_keys,
            "message": "Cannot materialize selected assets: " + "; ".join(invalid_assets),
        }

    repositories = {
        (
            node.get("repository", {}).get("location", {}).get("name") or "(unknown)",
            node.get("repository", {}).get("name") or "(unknown)",
        )
        for node in nodes_by_key.values()
    }
    if len(repositories) != 1:
        details = ", ".join(
            f"{location}/{repository}" for location, repository in sorted(repositories)
        )
        return {
            "asset_keys": launched_keys,
            "required_asset_keys_added": required_keys,
            "message": (
                "Assets must be defined in one repository to launch a single run. "
                f"Found: {details}."
            ),
        }
    repository_location, repository_name = next(iter(repositories))

    # GraphQL asset launches still target a named job, so every selected asset
    # must belong to at least one job in common.
    common_jobs: set[str] | None = None
    jobs_by_asset: list[str] = []
    for key in launched_keys:
        jobs = set(nodes_by_key[key].get("jobNames") or [])
        common_jobs = jobs if common_jobs is None else common_jobs & jobs
        jobs_by_asset.append(f"{key}: {', '.join(sorted(jobs)) or '(none)'}")
    if not common_jobs:
        return {
            "asset_keys": launched_keys,
            "required_asset_keys_added": required_keys,
            "message": (
                "Selected assets do not share a common job. "
                + "; ".join(jobs_by_asset)
            ),
        }

    # Prefer Dagster's implicit asset job for ad hoc materialization, then use
    # deterministic name ordering when several compatible jobs remain.
    job_name = min(
        common_jobs,
        key=lambda name: (
            0 if name == "__ASSET_JOB" else 1 if name.startswith("__ASSET_JOB") else 2,
            name,
        ),
    )

    # Explicitly select only checks that can execute in the chosen asset job.
    asset_check_selection: list[dict] = []
    seen_checks: set[tuple[str, str]] = set()
    for key in launched_keys:
        checks_or_error = nodes_by_key[key].get("assetChecksOrError") or {}
        if checks_or_error.get("__typename") != "AssetChecks":
            continue
        for check in checks_or_error.get("checks", []):
            check_jobs = check.get("jobNames") or []
            check_name = check.get("name")
            if not check_name or (check_jobs and job_name not in check_jobs):
                continue
            check_key = (key, check_name)
            if check_key in seen_checks:
                continue
            seen_checks.add(check_key)
            asset_check_selection.append(
                {
                    "assetKey": _asset_key_input(key),
                    "name": check_name,
                }
            )

    execution_metadata = None
    if tags:
        execution_metadata = {
            "tags": [{"key": key, "value": value} for key, value in tags.items()]
        }

    mutation = """
    mutation MaterializeAssets(
      $locationName: String!,
      $repoName: String!,
      $jobName: String!,
      $assetSelection: [AssetKeyInput!]!,
      $assetCheckSelection: [AssetCheckHandleInput!],
      $executionMetadata: ExecutionMetadata,
      $runConfigData: RunConfigData
    ) {
      launchRun(executionParams: {
        selector: {
          repositoryLocationName: $locationName,
          repositoryName: $repoName,
          jobName: $jobName,
          assetSelection: $assetSelection,
          assetCheckSelection: $assetCheckSelection
        },
        runConfigData: $runConfigData,
        executionMetadata: $executionMetadata
      }) {
        __typename
        ... on LaunchRunSuccess { run { runId status } }
        ... on InvalidSubsetError { message }
        ... on PipelineNotFoundError { message }
        ... on InvalidStepError { invalidStepKey }
        ... on UnauthorizedError { message }
        ... on PythonError { message }
        ... on PresetNotFoundError { message }
        ... on ConflictingExecutionParamsError { message }
        ... on RunConfigValidationInvalid { errors { message } }
      }
    }
    """
    launch_data = gql(
        mutation,
        {
            "locationName": repository_location,
            "repoName": repository_name,
            "jobName": job_name,
            "assetSelection": [_asset_key_input(key) for key in launched_keys],
            "assetCheckSelection": asset_check_selection,
            "runConfigData": run_config or {},
            "executionMetadata": execution_metadata,
        },
        env=env,
    )

    result = {
        "job_name": job_name,
        "repository_location": repository_location,
        "repository_name": repository_name,
        "requested_asset_keys": requested_keys,
        "asset_keys": launched_keys,
        "required_asset_keys_added": required_keys,
    }

    launch_result, error = _classify_launch_result(
        launch_data, "launchRun", f"materializing {', '.join(launched_keys)}"
    )
    if error is not None:
        # Unlike the other launch tools this one reports rather than raises, so
        # the caller keeps the preflight context — which assets resolved, which
        # job was chosen — when diagnosing the failure. "launched_asset_keys" is
        # deliberately absent: nothing launched, and an agent reads that key as
        # proof that it did.
        result["error"] = "launch_failed"
        result["message"] = error
        return result

    result["launched_asset_keys"] = launched_keys
    result.update(launch_result or {})
    return result


def terminate_run(run_id: str, env: str | None = None) -> dict:
    """Terminate a running or queued Dagster run.

    - run_id: the runId to terminate (get it from get_runs)

    Returns the run's final status on success, or an error message if the
    run was not found or could not be terminated.

    When to use: to stop a stuck, hung, or runaway run. Only works on runs
    with status STARTED or QUEUED. Already-finished runs cannot be terminated.
    """
    query = """
    mutation TerminateRun($runId: String!) {
      terminateRun(runId: $runId) {
        ... on TerminateRunSuccess { run { runId status } }
        ... on TerminateRunFailure { message }
        ... on RunNotFoundError { message }
        ... on PythonError { message }
      }
    }
    """
    data = gql(query, {"runId": run_id}, env=env)
    return data.get("terminateRun", {})


def launch_job(
    job_name: str,
    repository_location: str,
    repository_name: str = "__repository__",
    asset_keys: list[str] | None = None,
    tags: dict[str, str] | None = None,
    run_config: dict | None = None,
    env: str | None = None,
) -> dict:
    """Launch a job or materialize specific assets. Use list_jobs first to find valid names.

    For new asset-centric workflows, prefer resolve_asset_selection followed by
    materialize_assets. ``asset_keys`` remains supported for compatibility and
    is sent to Dagster as an asset selection, not an op selection.

    Required parameters:
    - job_name: name of the job (from list_jobs, e.g. 'my_etl_job')
    - repository_location: code location name (from list_jobs, e.g. 'my_project')
    - repository_name: defaults to '__repository__', which is correct only for
      a code location that did not name its repository. Pass the real name if
      list_jobs shows one (e.g. 'misc_repo'); a wrong value raises
      PipelineNotFoundError rather than launching anything.

    Optional parameters:
    - asset_keys: list of asset key strings to materialize. Use this with the
      job that targets them (often '__ASSET_JOB' or a custom asset job name).
      Example: ['raw_orders', 'clean_orders']
    - tags: dict of key-value tags to attach to the run.
      Example: {'triggered_by': 'dataops_agent', 'priority': 'high'}
    - run_config: dict of run configuration to pass to the job. This is the
      same YAML/dict you would enter in the Dagster UI Launchpad.
      Example: {'ops': {'my_op': {'config': {'start_date': '2026-03-01'}}}}

    Returns the launched run's runId and status on success, or an error message.

    When to use: to re-run a failed job, trigger an ad-hoc materialization,
    or launch a job with custom config or tags. After launching, use
    get_run_status or get_runs to monitor progress.
    For partitioned ASSET backfills — especially an asset inside a multi-asset
    /dbt op, or one with a single_run backfill policy — use backfill_assets.
    """
    execution_metadata: dict = {}
    if tags:
        execution_metadata["tags"] = [{"key": k, "value": v} for k, v in tags.items()]

    asset_selection = (
        [_asset_key_input(asset_key) for asset_key in asset_keys]
        if asset_keys
        else None
    )

    query = """
    mutation LaunchJob(
      $locationName: String!,
      $repoName: String!,
      $jobName: String!,
      $assetSelection: [AssetKeyInput!],
      $executionMetadata: ExecutionMetadata,
      $runConfigData: RunConfigData
    ) {
      launchRun(executionParams: {
        selector: {
          repositoryLocationName: $locationName,
          repositoryName: $repoName,
          jobName: $jobName,
          assetSelection: $assetSelection
        },
        runConfigData: $runConfigData,
        executionMetadata: $executionMetadata
      }) {
        __typename
        ... on LaunchRunSuccess { run { runId status } }
        ... on InvalidSubsetError { message }
        ... on PipelineNotFoundError { message }
        ... on InvalidStepError { invalidStepKey }
        ... on UnauthorizedError { message }
        ... on PythonError { message }
        ... on PresetNotFoundError { message }
        ... on ConflictingExecutionParamsError { message }
        ... on RunConfigValidationInvalid { errors { message } }
      }
    }
    """
    variables = {
        "locationName": repository_location,
        "repoName": repository_name,
        "jobName": job_name,
        "assetSelection": asset_selection,
        "runConfigData": run_config or {},
        "executionMetadata": execution_metadata or None,
    }
    data = gql(query, variables, env=env)
    return _unwrap_launch_result(
        data, "launchRun", f"launching job {job_name!r}"
    )


def launch_job_with_partitions(
    job_name: str,
    repository_location: str,
    partition_keys: list[str],
    repository_name: str = "__repository__",
    partition_set_name: str | None = None,
    tags: dict[str, str] | None = None,
    from_failure: bool = False,
    env: str | None = None,
) -> dict:
    """Launch a partitioned job for one or more partition keys.

    Use list_jobs to find job names. Use get_asset_details to check if an asset
    is partitioned (isPartitioned field) and what partition definition it uses.

    Required parameters:
    - job_name: name of the partitioned job (from list_jobs)
    - repository_location: code location name (from list_jobs)
    - partition_keys: one or more partition key strings to run.
      Examples: ['2024-01-01'], ['2024-01-01', '2024-01-02', '2024-01-03']

    Optional parameters:
    - repository_name: defaults to '__repository__', which is correct only for
      a code location that did not name its repository. Pass the real name if
      list_jobs shows one (e.g. 'misc_repo'); a wrong value raises
      PipelineNotFoundError rather than launching anything.
    - partition_set_name: partition set name; defaults to '{job_name}_partition_set'.
      Override this if the job uses a non-standard partition set name.
    - tags: additional key-value tags to attach to the launched runs.
      Example: {'triggered_by': 'dataops_agent'}
    - from_failure: if True, only re-run the failed steps within the given
      partitions (useful for retrying partially-failed partitioned runs)

    Returns backfillId on success — even for a single partition, Dagster creates
    a backfill record. Use list_backfills to monitor progress.

    When to use: to run a job for a specific date/partition, backfill historical
    partitions, or retry failed partitions. For non-partitioned jobs, use launch_job.
    NOTE: this launches one run PER partition via the job's partition set. For
    asset-selection backfills that respect BackfillPolicy.single_run (one ranged
    run), use backfill_assets.
    """
    resolved_partition_set = partition_set_name or f"{job_name}_partition_set"
    tag_list = [{"key": k, "value": v} for k, v in tags.items()] if tags else []

    query = """
    mutation LaunchPartitionBackfill($backfillParams: LaunchBackfillParams!) {
      launchPartitionBackfill(backfillParams: $backfillParams) {
        __typename
        ... on LaunchBackfillSuccess { backfillId }
        ... on PartitionSetNotFoundError { message }
        ... on PipelineNotFoundError { message }
        ... on PythonError { message }
        ... on UnauthorizedError { message }
        ... on RunConfigValidationInvalid { errors { message } }
      }
    }
    """
    variables = {
        "backfillParams": {
            "selector": {
                "repositorySelector": {
                    "repositoryLocationName": repository_location,
                    "repositoryName": repository_name,
                },
                "partitionSetName": resolved_partition_set,
            },
            "partitionNames": partition_keys,
            "tags": tag_list,
            "fromFailure": from_failure,
        }
    }
    data = gql(query, variables, env=env)
    return _unwrap_launch_result(
        data, "launchPartitionBackfill", f"backfilling job {job_name!r}"
    )


# ``run_config`` follows ``env`` to preserve the existing positional call signature.
def backfill_assets(
    asset_keys: list[str],
    partition_start: str | None = None,
    partition_end: str | None = None,
    partition_keys: list[str] | None = None,
    tags: dict[str, str] | None = None,
    env: str | None = None,
    run_config: dict | None = None,
) -> dict:
    """Launch a partition backfill for specific ASSETS (asset selection, not job).

    This is the MCP equivalent of the UI's "Materialize → partition range".
    Dagster resolves each asset's BackfillPolicy server-side: assets with
    BackfillPolicy.single_run() get ONE ranged run (vars min/max spanning the
    whole range); others get per-partition runs.

    Use this instead of:
    - launch_job(asset_keys=...): fails with DagsterInvalidSubsetError for an
      asset inside a multi-asset op (e.g. one dbt model inside a dbt_assets op),
      and cannot target partitions.
    - launch_job_with_partitions: partition-set (whole-job) selection that
      creates one run PER partition key — breaks single_run backfill policies.

    Required parameters:
    - asset_keys: asset key strings, e.g. ['clean_es_email_activity_detail'].
      Nested keys use '/' as the path separator (e.g. 'raw_chargebee_dlt/customer').
      Multiple assets must share a partition definition (same rule as the UI).

    Optional parameters:
    - partition_start / partition_end: inclusive bounds. Resolved against the
      asset's actual partition keys (fetched via GraphQL), so any partition
      definition works (daily/weekly/monthly/static). Omitted bounds default
      to the first / latest key. Use the SAME format the asset uses
      (e.g. '2026-05-25' for daily).
    - partition_keys: explicit key list; overrides partition_start/end.
    - tags: extra run tags, e.g. {'triggered_by': 'dataops_agent'}.
    - run_config: dict of run configuration applied to the backfill runs.
      This is the same structure used in the Dagster Launchpad.

    Returns {'backfillId': ...} on success or {'message': <error>} on failure.
    Monitor with list_backfills; runs carry the dagster/backfill tag.
    """
    if partition_keys is None:
        query = """
        query AssetPartitionKeys($assetKeys: [AssetKeyInput!]!) {
          assetNodes(assetKeys: $assetKeys) {
            partitionKeys
          }
        }
        """
        nodes = gql(
            query,
            {"assetKeys": [{"path": asset_keys[0].split("/")}]},
            env=env,
        ).get("assetNodes", [])
        all_keys: list[str] = nodes[0].get("partitionKeys", []) if nodes else []
        if not all_keys:
            return {
                "message": (
                    f"Asset '{asset_keys[0]}' is not partitioned (or not found) — "
                    "use materialize_assets for unpartitioned assets."
                )
            }

        def _bound_index(bound: str, default: int, side: str) -> int | dict:
            if bound is None:
                return default
            try:
                return all_keys.index(bound)
            except ValueError:
                # Best-effort "nearest keys" hint; assumes lexicographically
                # sorted partition keys (true for daily/monthly, not static).
                pos = bisect.bisect_left(all_keys, bound)
                nearest = all_keys[max(0, pos - 1) : pos + 2]
                return {
                    "message": (
                        f"partition_{side} '{bound}' is not a partition key of "
                        f"'{asset_keys[0]}'. Nearest keys: {nearest}."
                    )
                }

        start_idx = _bound_index(partition_start, 0, "start")
        if isinstance(start_idx, dict):
            return start_idx
        end_idx = _bound_index(partition_end, len(all_keys) - 1, "end")
        if isinstance(end_idx, dict):
            return end_idx
        if start_idx > end_idx:
            return {
                "message": (
                    f"partition_start '{all_keys[start_idx]}' is after "
                    f"partition_end '{all_keys[end_idx]}'."
                )
            }
        partition_keys = all_keys[start_idx : end_idx + 1]

    tag_list = [{"key": k, "value": v} for k, v in tags.items()] if tags else []

    query = """
    mutation LaunchAssetBackfill($backfillParams: LaunchBackfillParams!) {
      launchPartitionBackfill(backfillParams: $backfillParams) {
        __typename
        ... on LaunchBackfillSuccess { backfillId }
        ... on PartitionSetNotFoundError { message }
        ... on PipelineNotFoundError { message }
        ... on PythonError { message }
        ... on UnauthorizedError { message }
        ... on InvalidSubsetError { message }
        ... on RunConfigValidationInvalid { errors { message } }
      }
    }
    """
    backfill_params = {
        "assetSelection": [_asset_key_input(key) for key in asset_keys],
        "partitionNames": partition_keys,
        "tags": tag_list,
    }
    if run_config is not None:
        backfill_params["runConfigData"] = run_config
    variables = {"backfillParams": backfill_params}
    data = gql(query, variables, env=env)
    return _unwrap_launch_result(
        data, "launchPartitionBackfill", f"backfilling assets {asset_keys!r}"
    )


# ── Backfill and run retries (write) ─────────────────────────────────────────

_BACKFILL_STATUS_QUERY: Final[str] = """
query BackfillStatus($backfillId: String!) {
  partitionBackfillOrError(backfillId: $backfillId) {
    __typename
    ... on PartitionBackfill { id status isAssetBackfill }
    ... on BackfillNotFoundError { message }
    ... on PythonError { message }
  }
}
"""

# A backfill in one of these statuses is finished; Dagster calls them terminal.
_BACKFILL_TERMINAL_STATUSES: Final[frozenset[str]] = frozenset(
    {"COMPLETED", "COMPLETED_SUCCESS", "COMPLETED_FAILED", "FAILED", "CANCELED"}
)


def _backfill_state(backfill_id: str, env: str | None) -> tuple[dict, str | None]:
    """Return ({status, isAssetBackfill}, None), or ({}, message) when it cannot be found."""
    backfill = gql(_BACKFILL_STATUS_QUERY, {"backfillId": backfill_id}, env=env).get(
        "partitionBackfillOrError"
    ) or {}
    if backfill.get("__typename") != "PartitionBackfill":
        return {}, backfill.get("message") or f"Backfill {backfill_id!r} not found."
    return backfill, None


def _backfill_status(backfill_id: str, env: str | None) -> tuple[str | None, str | None]:
    """Return (status, None) for a backfill, or (None, message) when it cannot be found."""
    backfill, error = _backfill_state(backfill_id, env)
    return backfill.get("status"), error


def _backfill_mutation(
    mutation: str, field: str, success_type: str, backfill_id: str, env: str | None
) -> str | None:
    """Run a cancel or resume mutation; return an error message, or None on success."""
    result = gql(mutation, {"backfillId": backfill_id}, env=env).get(field) or {}
    if result.get("__typename") == success_type:
        return None
    return result.get("message") or f"Dagster returned {result.get('__typename')!r}."


def cancel_backfill(backfill_id: str, env: str | None = None) -> dict:
    """Cancel an in-progress backfill: launch no more runs and stop the ones in progress.

    - backfill_id: the backfill to cancel (from list_backfills)

    Returns {backfill_id, previous_status, status: 'CANCELING'} on success.
    The backfill daemon then terminates its in-progress runs and moves the
    backfill to CANCELED. Partitions that already materialized stay
    materialized.

    Refuses, and changes nothing, when the backfill already finished
    (COMPLETED_SUCCESS, COMPLETED_FAILED, FAILED, CANCELED) or is already
    stopping (CANCELING, or FAILING: Dagster is canceling its runs and will
    mark it FAILED). Dagster itself would mark even a finished backfill as
    canceling.
    """
    status, error = _backfill_status(backfill_id, env)
    if error is not None:
        return {"backfill_id": backfill_id, "message": error}
    if status in _BACKFILL_TERMINAL_STATUSES or status in ("CANCELING", "FAILING"):
        return {
            "backfill_id": backfill_id,
            "status": status,
            "message": f"Backfill is {status}; there is nothing to cancel. No change was made.",
        }
    error = _backfill_mutation(
        """
        mutation CancelBackfill($backfillId: String!) {
          cancelPartitionBackfill(backfillId: $backfillId) {
            __typename
            ... on CancelBackfillSuccess { backfillId }
            ... on Error { message }
          }
        }
        """,
        "cancelPartitionBackfill",
        "CancelBackfillSuccess",
        backfill_id,
        env,
    )
    if error is not None:
        return {"backfill_id": backfill_id, "status": status, "message": error}
    return {
        "backfill_id": backfill_id,
        "previous_status": status,
        "status": "CANCELING",
        "message": "Cancel requested. Runs in progress will be terminated.",
    }


def resume_backfill(backfill_id: str, env: str | None = None) -> dict:
    """Resume a backfill that FAILED because Dagster hit an error while running it.

    - backfill_id: the backfill to resume (from list_backfills)

    A backfill has status FAILED when the backfill daemon itself failed,
    for example on a code location error. Resume sets it back to REQUESTED
    so the daemon continues where it stopped. Returns {backfill_id,
    previous_status, status: 'REQUESTED'} on success.

    Refuses, and changes nothing, for any status other than FAILED. To run
    the failed partitions of a COMPLETED_FAILED backfill again, use
    retry_backfill.
    """
    status, error = _backfill_status(backfill_id, env)
    if error is not None:
        return {"backfill_id": backfill_id, "message": error}
    if status != "FAILED":
        return {
            "backfill_id": backfill_id,
            "status": status,
            "message": (
                f"Only a FAILED backfill can be resumed; this one is {status}. "
                "No change was made. Use retry_backfill to re-run failed partitions."
            ),
        }
    error = _backfill_mutation(
        """
        mutation ResumeBackfill($backfillId: String!) {
          resumePartitionBackfill(backfillId: $backfillId) {
            __typename
            ... on ResumeBackfillSuccess { backfillId }
            ... on Error { message }
          }
        }
        """,
        "resumePartitionBackfill",
        "ResumeBackfillSuccess",
        backfill_id,
        env,
    )
    if error is not None:
        return {"backfill_id": backfill_id, "status": status, "message": error}
    return {
        "backfill_id": backfill_id,
        "previous_status": status,
        "status": "REQUESTED",
        "message": "Backfill resumed.",
    }


def retry_backfill(backfill_id: str, from_failure: bool = True, env: str | None = None) -> dict:
    """Launch a new backfill that re-runs what a finished backfill did not complete.

    - backfill_id: the finished backfill to retry (from list_backfills)
    - from_failure: True (default) re-runs what did not succeed. False
      re-runs every partition of the original backfill.
      - Asset backfill: re-runs every partition that did not materialize:
        failed, canceled, and never launched. Dagster refuses when nothing
        is left to run.
      - Job backfill: re-runs only partitions whose last run FAILED, from
        the failed step. Canceled and never-launched partitions are skipped,
        and Dagster does not refuse when there are none; the result then
        carries a 'note'. Use launch_job_with_partitions for those
        partitions.

    Returns {backfillId} of the NEW backfill. It keeps the tags of the
    original and adds 'dagster/parent_backfill_id'. Monitor it with
    get_backfill.

    Refuses, and changes nothing, unless the original backfill is finished
    (COMPLETED_SUCCESS, COMPLETED_FAILED, FAILED, or CANCELED).
    """
    backfill, error = _backfill_state(backfill_id, env)
    if error is not None:
        return {"backfill_id": backfill_id, "message": error}
    status = backfill.get("status")
    if status not in _BACKFILL_TERMINAL_STATUSES:
        return {
            "backfill_id": backfill_id,
            "status": status,
            "message": (
                f"Backfill is {status}; only a finished backfill can be retried. "
                "No change was made."
            ),
        }
    mutation = """
    mutation RetryBackfill($params: ReexecutionParams!) {
      reexecutePartitionBackfill(reexecutionParams: $params) {
        __typename
        ... on LaunchBackfillSuccess { backfillId }
        ... on InvalidStepError { invalidStepKey }
        ... on RunConfigValidationInvalid { errors { message } }
        ... on Error { message }
      }
    }
    """
    params = {
        "parentRunId": backfill_id,
        "strategy": "FROM_FAILURE" if from_failure else "ALL_STEPS",
    }
    data = gql(mutation, {"params": params}, env=env)
    result = _unwrap_launch_result(
        data, "reexecutePartitionBackfill", f"retrying backfill {backfill_id!r}"
    )
    if from_failure and not backfill.get("isAssetBackfill"):
        result["note"] = (
            "Job backfill: only partitions whose last run FAILED are re-run. "
            "Canceled and never-launched partitions are not; use "
            "launch_job_with_partitions for them."
        )
    return result


_REEXECUTION_STATUS_QUERY: Final[str] = """
query RunStatusForReexecution($runId: ID!) {
  runOrError(runId: $runId) {
    __typename
    ... on Run { status }
    ... on RunNotFoundError { message }
    ... on PythonError { message }
  }
}
"""

_FINISHED_RUN_STATUSES: Final[frozenset[str]] = frozenset({"SUCCESS", "FAILURE", "CANCELED"})

_REEXECUTION_STRATEGIES: Final[tuple[str, ...]] = (
    "FROM_FAILURE",
    "FROM_ASSET_FAILURE",
    "ALL_STEPS",
)


def reexecute_run(
    run_id: str,
    strategy: str = "FROM_FAILURE",
    tags: dict[str, str] | None = None,
    env: str | None = None,
) -> dict:
    """Re-execute a finished run, by default only the steps that failed or did not run.

    - run_id: the run to re-execute (from get_runs)
    - strategy:
      - 'FROM_FAILURE' (default): re-run the failed steps, the steps
        downstream of them, and steps that never ran. Outputs of steps that
        succeeded are reused
      - 'FROM_ASSET_FAILURE': like FROM_FAILURE, but only the assets that
        failed inside a multi-asset step
      - 'ALL_STEPS': re-run the whole run
    - tags: extra tags for the new run. The original run's tags are kept

    Returns the new run's {run {runId, status}}. The new run has parentRunId
    set to run_id; see it with get_run_status.

    Use this for a failed run launched by launch_job, materialize_assets, a
    schedule, or a sensor. For the runs of a backfill, use retry_backfill.

    Refuses, and changes nothing, unless the run is finished (SUCCESS,
    FAILURE, or CANCELED); Dagster itself would start ALL_STEPS next to a
    run that is still going. Dagster refuses FROM_FAILURE on a run that
    succeeded.
    """
    normalized = strategy.upper()
    if normalized not in _REEXECUTION_STRATEGIES:
        raise ValueError(
            f"strategy must be one of {', '.join(_REEXECUTION_STRATEGIES)}; got {strategy!r}."
        )
    run = gql(_REEXECUTION_STATUS_QUERY, {"runId": run_id}, env=env).get("runOrError") or {}
    if run.get("__typename") != "Run":
        return {"run_id": run_id, "message": run.get("message") or f"Run {run_id!r} not found."}
    if run.get("status") not in _FINISHED_RUN_STATUSES:
        return {
            "run_id": run_id,
            "status": run.get("status"),
            "message": (
                f"Run is {run.get('status')}; only a finished run can be re-executed. "
                "No change was made."
            ),
        }
    mutation = """
    mutation ReexecuteRun($params: ReexecutionParams!) {
      launchRunReexecution(reexecutionParams: $params) {
        __typename
        ... on LaunchRunSuccess { run { runId status } }
        ... on InvalidStepError { invalidStepKey }
        ... on RunConfigValidationInvalid { errors { message } }
        ... on Error { message }
      }
    }
    """
    params: dict[str, Any] = {
        "parentRunId": run_id,
        "strategy": normalized,
        "useParentRunTags": True,
    }
    if tags:
        params["extraTags"] = _tag_inputs(tags)
    data = gql(mutation, {"params": params}, env=env)
    return _unwrap_launch_result(data, "launchRunReexecution", f"re-executing run {run_id!r}")


# ── Write tools (only registered when DAGSTER_READ_ONLY=false) ────────────────

if not READ_ONLY:
    mcp.tool()(reload_code_location)
    mcp.tool()(terminate_run)
    mcp.tool()(launch_job)
    mcp.tool()(materialize_assets)
    mcp.tool()(launch_job_with_partitions)
    mcp.tool()(backfill_assets)
    mcp.tool()(start_schedule)
    mcp.tool()(stop_schedule)
    mcp.tool()(start_sensor)
    mcp.tool()(stop_sensor)
    mcp.tool()(cancel_backfill)
    mcp.tool()(resume_backfill)
    mcp.tool()(retry_backfill)
    mcp.tool()(reexecute_run)


def main():
    mcp.run()


if __name__ == "__main__":
    main()
