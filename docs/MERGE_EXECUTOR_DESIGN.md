# Transactional merge executor - design (NOT implemented, NOTHING applied)

No migration has been created or applied, and no merge has been executed. This is a design for review. The SQL
below is a **sketch inside a document**, not a migration file.

## 1. Why the legacy mover cannot be used

From the dry-run plan on the live structure (101 candidate pairs):

* `ForeignKeyService` moves only 7 table names; 3 of them do not exist (`meetings`, `documents`, `opportunities`)
  and it **swallows every error**. It never moves `products`, `components`, `lead_status`, `company_suppliers`,
  `news`, `emails`, `interactions`, `tasks`: **524 rows** across the 101 pairs would be orphaned or, if the FK is
  `ON DELETE CASCADE`, silently deleted. The earlier real merge (company 3 into 4) shows exactly that pattern: no rows
  for company 3 remain in products/components/lead_status.
* Every pair has a `lead_status` row on both sides (and `company_profile` for some). A plain `company_id` update
  would create two rows where one is expected (or fail on a unique constraint).
* The steps are separate REST calls (profile merge, FK moves, log, delete): a failure part-way leaves split data.
* `MergeEngine` keeps the *longer* text, not the better-sourced one, and writes the primary key back.

## 2. Step 0 (read-only, needs you): learn the real delete rules

Run `agentic.fk_introspect.INTROSPECTION_SQL` (a single catalog SELECT) in the Supabase SQL editor, save the one JSON
cell, then `python -m agentic.fk_introspect result.json`. It reports per table: `ON DELETE` rule, whether it is
one-row-per-company, composite unique keys, RLS flags/policies, triggers. The planner accepts this output
(`MergePlanner(source, fk_info=...)`) and replaces "rule unverified" with the concrete consequence per table.
The SQL has not been run by me.

## 3. Design principles

1. **One transaction, server-side.** The merge runs inside a Postgres function; Python only sends an approved plan.
2. **Nothing is lost silently.** Before deleting the duplicate, the function asserts that *zero* rows still reference
   it in any table found by the live FK catalogue, else it raises and the whole transaction rolls back.
3. **Snapshot first.** All rows of the duplicate and its dependents are written as JSON to an audit table in the same
   transaction, so a merge can be reversed by a restore function.
4. **Approved plan or nothing.** The function takes the plan JSON plus an approval record and refuses unless the
   hash matches, the approval is current, and the data has not drifted since the dry run.
5. **One pair per call**, bounded, with a kill switch; no `merge_all`.
6. **Cross-sector rows are not merged.** `sector_id` is single-valued. Probable multi-sector organisations need an
   additive link table first; otherwise they stay as separate rows.

## 4. Additive objects needed (all require your approval; none created)

| Object | Purpose |
|---|---|
| `merge_audit` (merge_id, master_id, duplicate_id, plan_hash, plan jsonb, snapshot jsonb, approved_by, approved_at, executed_at, restored_at) | immutable record + rollback source; `unique(duplicate_id)` gives idempotency |
| `company_sector(company_id, sector_id, source)` | keeps sector membership when two sector listings become one |
| `execute_company_merge(plan jsonb, approval jsonb)` (SECURITY DEFINER) | the transactional executor |
| `restore_company_merge(merge_id)` | reverses a merge from the snapshot |
| role `merge_executor` | only role allowed to run the two functions |

Existing `merge_logs` has no snapshot, hash or approver and no FKs; it stays as-is.

## 5. Executor sketch

```text
execute_company_merge(plan, approval):
  assert approval.plan_hash = sha256(plan) and approval.by is not null and not expired
  assert plan.kind in ('merge') and plan.category = 'likely_duplicate' or plan.explicit_override
  lock both companies FOR UPDATE ORDER BY company_id            -- avoids deadlocks, blocks concurrent edits
  drift check: for each table in live_fk_catalogue(companies):  -- computed inside the function, not trusted from client
        assert count(master) = plan.counts.master[t] and count(dup) = plan.counts.dup[t]
  insert merge_audit(snapshot = to_jsonb(dup row) + all dependent rows)   -- unique(duplicate_id) => idempotent
  apply approved field decisions to master company + profile      -- only fields the approver chose; no legacy defaults
  insert company_sector(master, dup.sector_id) if different
  per-table strategy (from the plan, approved by a human):
     company_profile   merge chosen fields into master row; delete duplicate's row
     lead_status       keep master row (approver may choose max score); archive duplicate's in snapshot; delete
     contacts          move; drop exact same-person duplicates after copying empty fields
     products/components  move except case-insensitive duplicates
     company_suppliers move; collapse duplicates on (supplier_id, component)
     ai_insights, news, projects, emails, interactions, tasks   move
  postcheck: for every FK table, count(rows referencing dup) = 0          -- else RAISE => full rollback
  postcheck: master counts = before + moved - deduplicated (as planned)
  delete dup company
  return summary
```
Any exception rolls back everything, including the snapshot.

## 6. Python side (to be written after the above is approved)

`MergeExecutor` only: loads an *approved* plan (hash-verified), re-runs the read-only planner to detect drift,
calls the RPC with a dedicated credential, records the result, never constructs SQL from model output. It is
unreachable from agents (no tool exists) and is disabled unless `AGENTIC_ENABLE_MERGES=1` *and* a plan id is given.

## 7. Least privilege

Agents keep a read-only key. The merge function is executable only by `merge_executor`; the service-role key used
elsewhere gets no `EXECUTE`. RLS stays on; the function is the only write path for merges.

## 8. Test plan (zero cost)

Run only against a **scratch copy** (a second free Supabase project, or a local Postgres), never the live data:
invariants after every merge (no orphan rows, counts conserved, sector membership preserved, audit row present);
failure injection (raise at each step => database byte-identical); restore round-trip; concurrency (two merges on
the same company serialise); idempotency (second call is a no-op); drift (edit a dependent row between approval and
execution => refused); permission test (agent credential cannot call the function).

## 9. Acceptance criteria

All of section 8 passes on the scratch copy; the dry-run plan for a chosen pair matches the executed result
row-for-row; you approve the specific plan hash; one pair is executed first and verified before any other.

## 10. What happens next (each step needs your explicit go-ahead)

1. You run the introspection SQL and share the JSON (read-only).
2. I regenerate the plan with real delete rules and send you the revised risk summary.
3. You approve the additive objects; I write the migration for a scratch database only.
4. Scratch-copy tests; only then discuss applying to production, one pair at a time.
