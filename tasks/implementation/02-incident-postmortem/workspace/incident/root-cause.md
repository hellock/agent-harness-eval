# Root cause investigation (written 2026-04-07 morning)

Trigger: worker-reconcile cron job started at 20:00 with new concurrency=180
(raised from 20 the day before by tom). Each worker opens a postgres transaction
for the duration of its batch (~3-5 minutes). At 180 concurrent workers we held
~180 idle-in-transaction connections, leaving only ~20 of the 200-connection
pool for the rest of the system. checkout-api needs 1-2 conns per request and
starved.

Why it wasn't caught:
- we have an alert on pg connection count >180 but the threshold was set 3
  quarters ago when max_connections was 100, never updated when we raised
  max to 200
- load test only exercised reconcile at concurrency=20
- the concurrency change was a one-line PR with no review required because
  the file was in the worker-config tier (auto-approve enabled)

Why it took 22 minutes:
- first instinct was "blame the deploy" — alex and priya spent ~6 min on a
  rollback that wasn't going to help
- the connection pool dashboard was buried in a sub-dashboard, took priya
  ~3 min to find
- we don't have any link from a 5xx alert to "current top consumers of pg
  connections" — would have been a 30-second answer

How we mitigated:
- tom hotfixed concurrency back to 20
- worker drained back to normal in ~3 minutes
- we did NOT need to roll back any code; the worker config was hot-reloaded
