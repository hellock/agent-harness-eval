21:18 alex (oncall): @priya checkout-api looks bad
21:23 alex: checkout-api v2.41.0 went out at 20:50 — about 30 min before
21:35 priya (eng lead): tom shipped concurrency=180 yesterday on the reconcile worker
21:39 tom: yeah i raised concurrency to 180 to clear backlog faster, didn't realize each worker holds an open txn
21:50 alex: incident mitigated. follow-ups tomorrow.
22:30 priya: ok action items so far —
             1. raise the pg conn alert threshold immediately (alex)
             2. add an auto-attached "top pg connection holders" widget to oncall dashboard (priya)
             3. require code review on worker-config changes that touch concurrency or pool sizes (tom)
             4. add a load test for the reconcile worker at production-realistic concurrency (tom)
             5. write up runbook section: "5xx + db conn errors → check worker concurrency" (alex)
22:31 priya: durations: alex stuff this week, mine + tom's by friday next week, runbook within 10 days
