# IBM Bob Development Log

## Improvement 1 — Reliable Route Failure Handling

**Project:** CommuteIQ  
**Component:** FastAPI backend  
**AI development assistant:** IBM Bob

### Problem identified

The `/v2/predict` endpoint could encounter uninitialized variables when geocoding or OSRM routing failed. This could result in an unexpected HTTP 500 error.

The endpoint also substituted a fixed 12 km distance when routing failed, potentially presenting an unsupported distance as a valid calculation.

### IBM Bob's contribution

IBM Bob analyzed the backend, identified the routing failure, proposed a targeted correction, implemented error handling, and generated automated regression tests.

### Changes implemented

- Initialized routing-related variables before the routing operation.
- Replaced the hardcoded fallback with an HTTP 503 response when routing cannot be completed.
- Added automated tests for geocoding failures, routing failures, and successful predictions.
- Preserved the existing successful prediction response.

### Test results

**10 tests passed, 0 failed**, as reported by IBM Bob.

### Remaining considerations

- Narrow exception handling to expected routing failures.
- Review model serialization compatibility warnings.
- Test the endpoint through a deployed frontend.
- Verify behavior in a real deployment environment.
