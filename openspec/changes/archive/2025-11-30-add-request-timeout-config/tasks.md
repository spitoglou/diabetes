## 1. Implementation
- [x] 1.1 Add `request_timeout` field to Config class in `src/config.py`
- [x] 1.2 Update `client.py` to use `config.request_timeout` in `requests.post()`
- [x] 1.3 Update `client_twin.py` to use `config.request_timeout` in `requests.post()`
- [x] 1.4 Add `REQUEST_TIMEOUT` to `.env.example` with default value
- [x] 1.5 Run pylint to verify warnings are resolved
