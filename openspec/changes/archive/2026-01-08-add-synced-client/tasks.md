# Tasks for add-synced-client

## Implementation Tasks

1. [x] Add `_find_closest_time_index()` method to `OhioBgcProvider`
   - Input: target time (datetime.time)
   - Output: index of closest matching glucose reading
   - Handle midnight wrap-around for circular time matching
   - **File:** `src/bgc_providers/ohio_bgc_provider.py`

2. [x] Add `simulate_synced_glucose_stream()` method to `OhioBgcProvider`
   - Yields glucose readings with current system timestamps
   - Starts from closest time-of-day match
   - Wraps around dataset when reaching the end
   - **File:** `src/bgc_providers/ohio_bgc_provider.py`

3. [x] Add `stream_synced_data()` function to client module
   - Similar to `stream_data()` but uses synced stream
   - Sends FHIR payloads with current timestamps
   - **File:** `scripts/serving/client.py`

4. [x] Add `serve synced-client` CLI command
   - Options: `--patient`, `--verbose`
   - Calls `stream_synced_data()`
   - **File:** `cli.py`

## Validation Tasks

5. [x] Verify CLI command registered: `uv run python cli.py serve --help` shows `synced-client`
6. [x] Verify unit test: `_find_closest_time_index()` returns correct index for noon (5111)
7. [x] Verify synced stream yields current timestamps (2026 in ISO time string)
