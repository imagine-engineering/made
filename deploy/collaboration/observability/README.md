# Collaboration observability contract

This bundle renders the four access-controlled dashboards and automatic
stop/rollback alerts required by the collaborative-workspace operational-limit
registry. It is deployment configuration, not a second product-state or tenant
authority.

- `render.py` reads the canonical `OL-*` table and generates one content-free
  panel for every declared service metric.
- `stop-signals.json` maps each automatic stop condition to one metric,
  Prometheus expression, closed operator action and owning dashboard.
- `prometheus-rules.yaml` pages on every stop signal. These are release and
  rollback gates, not informational alerts.
- `logging-policy.json` keeps metrics private, constrains labels to a closed
  low-cardinality set, allowlists structured log fields, drops unknown fields,
  caps records at 64 KiB and preserves the existing disabled client-telemetry
  policy while server observability remains active.

Dashboards deliberately contain neither tenant variables nor stable
community/user/job/event labels. Tenant diagnosis belongs in separately
authorized, retention-bound log and trace tooling using a pseudonymous
correlation hash.

Run `python3 deploy/collaboration/observability/check.py` to validate the
tracked artifacts. To regenerate them and check their coverage against the local
operational limits in `.agents/local/collaboration/`, run:

```sh
python3 deploy/collaboration/observability/render.py
python3 deploy/collaboration/observability/check.py --private-docs
```

The local coverage check rejects render drift or an uncovered `OL-*` row.
Both modes reject a missing stop/rollback signal, a public metrics listener, an
open log field/action, client
telemetry while disabled, or any private content copied from the frozen protocol
fixture. It also mutates the in-memory inputs to prove the missing-signal and
private-content checks fail independently.
