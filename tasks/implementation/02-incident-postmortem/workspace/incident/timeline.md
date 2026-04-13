# Raw oncall log — 2026-04-06 (Sun)
21:14  pagerduty fires: checkout-api 5xx > 8% (1 min window)
21:14  alex acked the page
21:16  alex confirms 5xx is real, opens war room channel
21:18  alex pings priya (eng lead) "checkout-api looks bad, also seeing db conn errors"
21:21  priya joins war room
21:22  priya: "did anything ship recently?"
21:23  alex: "checkout-api v2.41.0 went out at 20:50 — about 30 min before"
21:25  priya rolls back to v2.40.3
21:28  alex: "5xx still climbing, now at 14%"
21:29  priya: "ok rollback didn't help, this isn't the deploy"
21:32  alex: "i'm seeing 'too many connections' from postgres"
21:33  priya: "max_connections is 200, what's eating them?"
21:34  alex shares pg_stat_activity output: ~180 idle in transaction from worker-reconcile
21:35  priya: "the reconcile worker again. tom shipped concurrency=180 yesterday"
21:36  priya pages tom
21:39  tom joins, says "yeah i raised concurrency to 180 to clear backlog faster, didn't realize each worker holds an open txn"
21:41  tom hotfixes worker-reconcile concurrency=20
21:43  pg connections recovering
21:46  checkout-api 5xx falls below 1%
21:50  alex declares incident mitigated
22:30  team agrees on tomorrow's followups
