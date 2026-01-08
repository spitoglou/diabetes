# Add Synced Client

## Summary
Add a new CLI command `serve synced-client` that streams CGM data using the current system date/time instead of historical timestamps from the dataset. The sequence starts from the data point closest to the current time of day, making simulations appear as real-time data.

## Motivation
The current `serve client` command streams data with historical timestamps from the Ohio dataset. For realistic simulation scenarios (demos, testing, dashboard integration), we need data that appears to be generated in real-time with current timestamps while still following realistic glucose progression patterns from the dataset.

## Scope
- **In scope:**
  - New CLI command `serve synced-client`
  - Method to find closest time-of-day match in dataset
  - Timestamp replacement with current system time
  - Support for patient selection via `--patient` flag
  
- **Out of scope:**
  - Changes to existing `serve client` command
  - Changes to FHIR payload structure
  - Changes to server-side data handling

## Approach
1. Add a helper method to `OhioBgcProvider` that finds the dataset index closest to a given time of day
2. Add a new generator method `simulate_synced_glucose_stream()` that yields readings with current system timestamps
3. Add a new function `stream_synced_data()` in `scripts/serving/client.py`
4. Add CLI command `serve synced-client` in `cli.py`

## Risks
- **Low:** Minor addition with no changes to existing functionality
- Time matching may have edge cases around midnight wrap-around (mitigated by calculating minimum circular distance)
