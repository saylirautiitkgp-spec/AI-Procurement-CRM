"""Reviewed DRY-RUN merge planner.

READ-ONLY BY CONSTRUCTION: it consumes a `MergeDataSource` that has no write methods, produces a report,
and has no code path that merges, moves or deletes anything. Executing a merge is out of scope until a
transactional, fully FK-aware executor exists and a human has approved a specific plan.

It reuses the legacy services instead of re-implementing them:
  * Golden Record score   -> CompanyDuplicateService.score_company (run unchanged through a read-only adapter)
  * field-merge semantics -> MergeEngine.merge_value (shown as the "legacy rule" for every field)
  * FK coverage           -> ForeignKeyService.tables (compared against the tables that really reference
                             companies, to expose rows the legacy mover would not carry over)
"""

from __future__ import annotations

import importlib
import json
import sys
from collections import defaultdict
from itertools import combinations
from types import SimpleNamespace
from typing import Any, Protocol

from agentic import names
from agentic.evidence import now_iso, registrable_domain

# Tables whose company_id references companies.company_id, read from the live schema (PostgREST OpenAPI).
COMPANY_FK_TABLES = ("ai_insights", "company_profile", "company_suppliers", "components", "contacts", "emails",
                     "interactions", "lead_status", "news", "products", "projects", "tasks")
ONE_PER_COMPANY = ("company_profile", "lead_status")
COMPANY_FIELDS = ("company_name", "sector_id", "sector", "sub_sector", "country", "revenue", "number_of_plants",
                  "website", "linkedin_url", "company_type", "priority", "enrichment_status")
OPERATIONAL_FIELDS = {"enrichment_status", "priority"}      # workflow state, not facts about the company
PROFILE_SKIP = {"profile_id", "company_id", "created_at", "updated_at"}
HIGH_RISK_FIELDS = {"company_name", "sector_id", "country", "website", "linkedin_url"}
NEAR_NAME_THRESHOLD = 0.92

GLOBAL_BLOCKERS = [
    "No transactional merge executor exists (Supabase REST is not atomic); a failure part-way would leave split data.",
    "ON DELETE rules for the foreign keys are unverified (run the pg_constraint query); deleting a duplicate may "
    "silently cascade-delete dependent rows.",
    "Merge execution requires a specific, human-approved plan; none exists.",
]


class MergeDataSource(Protocol):
    """Read-only. Deliberately has NO insert/update/delete/merge methods."""

    def companies(self) -> list[dict]: ...
    def view(self, company_id: int) -> dict: ...


# ---------------------------------------------------------------------------------- legacy bridge
def load_legacy():
    """Import the legacy services (they create DB clients at import time, so this is lazy)."""
    dds = importlib.import_module("services.duplicate_detection_service")
    fks = importlib.import_module("services.foreign_key_service")
    eng = importlib.import_module("services.merge_engine")
    return SimpleNamespace(CompanyDuplicateService=dds.CompanyDuplicateService,
                           ForeignKeyService=fks.ForeignKeyService, MergeEngine=eng.MergeEngine)


class _ReadOnlyLegacyDb:
    """Feeds the unchanged legacy scorer from the already-loaded views (no database access)."""

    def __init__(self, views: dict[int, dict]):
        self._views = views

    def get_company_profile(self, company_id):
        return self._views[company_id].get("profile")

    def select(self, table, column, value):
        if table == "ai_insights":
            return [None] * int(self._views[value].get("insights_count", 0))
        raise RuntimeError(f"unexpected legacy read: {table}")      # the scorer must not read anything else


def _is_empty(v: Any) -> bool:
    return v is None or v == "" or v == [] or v == {}


def _same(a: Any, b: Any) -> bool:
    return json.dumps(a, sort_keys=True, default=str).strip().lower() == json.dumps(b, sort_keys=True, default=str).strip().lower()


# ---------------------------------------------------------------------------------------- planner
class MergePlanner:
    def __init__(self, source: MergeDataSource, legacy=None, fk_info: dict | None = None):
        self.source = source
        self.fk_info = fk_info            # output of agentic.fk_introspect.analyze (optional)
        self.fk_tables = list(fk_info["tables"]) if fk_info else list(COMPANY_FK_TABLES)
        self.legacy = legacy or load_legacy()
        self.engine = self.legacy.MergeEngine()
        self._views: dict[int, dict] = {}
        self._scorer = self.legacy.CompanyDuplicateService.__new__(self.legacy.CompanyDuplicateService)
        self._scorer.db = _ReadOnlyLegacyDb(self._views)
        self._scorer.resolver = None
        self.legacy_fk = self._legacy_fk_tables()

    # -- legacy FK list (from the real class, not a copy)
    def _legacy_fk_tables(self) -> list[str]:
        svc = self.legacy.ForeignKeyService.__new__(self.legacy.ForeignKeyService)
        try:
            self.legacy.ForeignKeyService.__init__(svc)      # only builds a DB handle + the table list
        except Exception:                                    # pragma: no cover - env without credentials
            return []
        return list(svc.tables)

    def _view(self, cid: int) -> dict:
        if cid not in self._views:
            self._views[cid] = self.source.view(cid)
        return self._views[cid]

    # ---------------------------------------------------------------- candidate generation
    def candidate_pairs(self) -> list[tuple[int, int, float, str]]:
        rows = self.source.companies()
        by_norm: dict[str, list[dict]] = defaultdict(list)
        for r in rows:
            by_norm[names.normalize_company_name(r["company_name"])].append(r)
        seen, out = set(), []
        for norm, group in by_norm.items():
            for a, b in combinations(sorted(group, key=lambda r: r["company_id"]), 2):
                out.append((a["company_id"], b["company_id"], 1.0, "exact_name"))
                seen.add((a["company_id"], b["company_id"]))
        blocks: dict[str, list[dict]] = defaultdict(list)       # blocking avoids O(n^2) similarity calls
        for r in rows:
            toks = names.distinctive_tokens(r["company_name"])
            if toks:
                blocks[toks[0]].append(r)
        for block in blocks.values():
            for a, b in combinations(sorted(block, key=lambda r: r["company_id"]), 2):
                key = (a["company_id"], b["company_id"])
                if key in seen:
                    continue
                s = names.similarity(a["company_name"], b["company_name"])
                if s >= NEAR_NAME_THRESHOLD:
                    out.append((key[0], key[1], s, "near_name"))
                    seen.add(key)
        return sorted(out)

    # ---------------------------------------------------------------- evidence / classification
    def _signals(self, a: dict, b: dict) -> dict:
        wa, wb = a.get("website"), b.get("website")
        la, lb = a.get("linkedin_url"), b.get("linkedin_url")
        sig = {
            "same_country": (a.get("country") == b.get("country")) if a.get("country") and b.get("country") else None,
            "same_sector": a.get("sector_id") == b.get("sector_id"),
            "website": ("match" if wa and wb and registrable_domain(wa) == registrable_domain(wb)
                        else "conflict" if wa and wb else "unknown"),
            "linkedin": "match" if la and lb and la.rstrip("/") == lb.rstrip("/") else
                        "conflict" if la and lb else "unknown",
            "shared_contacts": len(set(a.get("contact_fingerprints", [])) & set(b.get("contact_fingerprints", []))),
            "contacts_compared": bool(a.get("contact_fingerprints")) and bool(b.get("contact_fingerprints")),
        }
        for kind in ("products", "components"):
            sa, sb = {x.lower() for x in a.get(kind, [])}, {x.lower() for x in b.get(kind, [])}
            sig[f"{kind}_jaccard"] = round(len(sa & sb) / len(sa | sb), 2) if sa and sb else None
        return sig

    def classify(self, a: dict, b: dict, how: str, sim: float) -> dict:
        s = self._signals(a, b)
        why: list[str] = []
        exact = how == "exact_name"
        strong_same = s["website"] == "match" or s["linkedin"] == "match" or s["shared_contacts"] > 0
        conflict = s["website"] == "conflict" or s["linkedin"] == "conflict"
        diff_country = s["same_country"] is False
        if exact:
            why.append("normalised company names are identical")
        else:
            why.append(f"names are similar (similarity {sim}) but not identical")
        if diff_country:
            why.append("countries differ")
        if conflict:
            why.append("website/LinkedIn differ, so they point to different organisations")
        if s["website"] == "match":
            why.append("same website domain")
        if s["shared_contacts"]:
            why.append(f"{s['shared_contacts']} contact(s) with the same name appear in both records")
        for kind in ("products", "components"):
            j = s[f"{kind}_jaccard"]
            if j is not None:
                why.append(f"{kind} overlap {j:.0%}")
        elevated = False
        if exact and not diff_country and not conflict:
            if s["same_sector"]:
                category, rec, conf = "likely_duplicate", "merge_candidate", "high"
                why.append("same sector and country: the same organisation recorded more than once")
                zero = [s.get(f"{k}_jaccard") == 0.0 for k in ("products", "components")]
                if any(zero) and not strong_same:
                    elevated = True
                    why.append("but the recorded products/components do not overlap - verify these are not different divisions")
            else:
                category = "same_name_different_sector"
                if strong_same:
                    rec, conf = "link_multi_sector", "medium"
                    why.append("different sectors but shared identifiers: probably ONE organisation listed under several "
                               "sectors. Merging would drop a sector_id (single-valued column); keep both rows or add an "
                               "additive multi-sector link")
                else:
                    rec, conf = "manual_review", "low"
                    why.append("different sectors and no shared identifier: could be one multi-sector organisation or "
                               "two unrelated companies. Do not merge without review")
        elif exact:
            category, rec, conf = "distinct_entities_likely", "keep_separate", "medium"
        else:
            category, rec = "near_name_match", "manual_review"
            conf = "medium" if strong_same and not conflict and not diff_country else "low"
        review = ("none_beyond_approval" if category == "likely_duplicate" and conf == "high" and not elevated
                  else "elevated" if elevated or conf == "low" else "standard")
        return {"category": category, "recommendation": rec, "confidence": conf, "review_level": review,
                "signals": s, "rationale": why}

    # ---------------------------------------------------------------- golden record + fields
    def golden(self, ids: list[int]) -> dict:
        scores = {}
        for cid in ids:
            v = self._view(cid)
            scores[cid] = self._scorer.score_company(self._legacy_company(v))
        master = sorted(ids, key=lambda i: (-scores[i], i))[0]
        runner = sorted(scores.values(), reverse=True)
        tie = len(runner) > 1 and runner[0] == runner[1]
        return {"master_id": master, "legacy_scores": scores,
                "rationale": [f"company {i}: legacy Golden Record score {scores[i]} "
                              f"({self._completeness(self._view(i))})" for i in ids]
                             + (["tie on score: lowest company_id chosen (legacy order)"] if tie else [])}

    @staticmethod
    def _legacy_company(v: dict) -> dict:
        c = {k: v.get(k) for k in COMPANY_FIELDS}
        c["company_id"] = v["company_id"]
        return c

    @staticmethod
    def _completeness(v: dict) -> str:
        have = [k for k in ("website", "linkedin_url", "sector", "revenue") if v.get(k)]
        prof = "profile" if v.get("profile") else "no profile"
        return f"{', '.join(have) or 'no key fields'}; {prof}; {v.get('insights_count', 0)} insight(s)"

    def field_decisions(self, master: dict, dup: dict) -> dict:
        def decide(key, old, new, high_risk=False):
            base = {"field": key, "master": old, "duplicate": new}
            if _is_empty(old) and _is_empty(new):
                return None
            if _same(old, new):
                return {**base, "decision": "identical"}
            if key in OPERATIONAL_FIELDS:
                return {**base, "decision": "operational_use_master"}
            if _is_empty(old):
                return {**base, "decision": "take_duplicate_value"}
            if _is_empty(new):
                return {**base, "decision": "keep_master_value"}
            legacy = self.engine.merge_value(key, old, new)
            if isinstance(old, list):
                return {**base, "decision": "union", "result": legacy}
            note = ("legacy MergeEngine keeps the LONGER text, not the better-sourced value" if isinstance(old, str)
                    else "legacy MergeEngine keeps the master value")
            return {**base, "decision": "conflict_needs_human", "legacy_default": legacy, "legacy_rule": note,
                    "high_risk": high_risk}

        companies = [d for f in COMPANY_FIELDS if (d := decide(f, master.get(f), dup.get(f), f in HIGH_RISK_FIELDS))]
        mp, dp = master.get("profile"), dup.get("profile")
        profile = []
        if mp or dp:
            for f in sorted(set(mp or {}) | set(dp or {})):
                if f in PROFILE_SKIP:
                    continue
                if d := decide(f, (mp or {}).get(f), (dp or {}).get(f)):
                    profile.append(d)
        return {"companies": companies, "company_profile": profile,
                "conflicts": [d["field"] for d in companies + profile if d["decision"] == "conflict_needs_human"]}

    # ---------------------------------------------------------------- dependents / FK implications
    def dependents(self, master: dict, dup: dict) -> dict:
        mdep, ddep = master.get("dependents", {}), dup.get("dependents", {})
        tables, blockers = [], []
        for t in self.fk_tables:
            m, d = mdep.get(t, 0), ddep.get(t, 0)
            in_legacy = t in self.legacy_fk
            info = (self.fk_info or {}).get("tables", {}).get(t, {})
            rule = info.get("on_delete", "UNKNOWN")
            one_per = info.get("one_row_per_company", t in ONE_PER_COMPANY)
            notes = []
            if d == 0:
                action = "nothing_to_move"
            elif one_per and m > 0:
                action = "merge_rows_manually"
                notes.append(f"both companies have a {t} row; a plain company_id update would create two")
            elif t == "contacts" and set(master.get("contact_fingerprints", [])) & set(dup.get("contact_fingerprints", [])):
                action = "move_then_deduplicate"
                notes.append("the same person appears in both records")
            elif t in ("products", "components") and master.get(t) and dup.get(t) and \
                    {x.lower() for x in master[t]} & {x.lower() for x in dup[t]}:
                action = "move_then_deduplicate"
                notes.append("identical names appear in both lists")
            else:
                action = "move_to_master"
            if d and not in_legacy:
                notes.append("NOT moved by the legacy ForeignKeyService")
                consequence = {
                    "CASCADE": f"{t}: {d} row(s) would be DELETED by ON DELETE CASCADE when the duplicate is removed",
                    "RESTRICT": f"{t}: {d} row(s) would make the delete FAIL (ON DELETE RESTRICT)",
                    "NO ACTION": f"{t}: {d} row(s) would make the delete FAIL (ON DELETE NO ACTION)",
                    "SET NULL": f"{t}: {d} row(s) would be orphaned (company_id set to NULL)",
                }.get(rule, f"{t}: {d} row(s) would be lost by the legacy mover (ON DELETE rule unverified: "
                            "orphaned, cascaded or blocking)")
                blockers.append(consequence)
            tables.append({"table": t, "master_rows": m, "duplicate_rows": d, "action": action,
                           "handled_by_legacy_mover": in_legacy, "on_delete": rule, "notes": notes})
        return {"tables": tables, "blockers": blockers,
                "legacy_tables_that_do_not_exist": [t for t in self.legacy_fk if t not in self.fk_tables],
                "rows_at_risk": sum(ddep.get(t, 0) for t in self.fk_tables if t not in self.legacy_fk)}

    def global_blockers(self) -> list[str]:
        out = list(GLOBAL_BLOCKERS)
        if self.fk_info:                      # delete rules are now known; the blocker becomes per-table (above)
            out = [b for b in out if not b.startswith("ON DELETE rules")]
        return out

    # ---------------------------------------------------------------- the plan
    def build(self) -> dict:
        pairs = self.candidate_pairs()
        ids = sorted({i for a, b, _, _ in pairs for i in (a, b)})
        for i in ids:
            self._view(i)
        legacy_sim = self.legacy.CompanyDuplicateService.similarity
        adj: dict[int, set[int]] = defaultdict(set)
        for a, b, _, _ in pairs:
            adj[a].add(b)
            adj[b].add(a)
        groups, seen = [], set()
        for start in ids:
            if start in seen:
                continue
            comp, stack = set(), [start]
            while stack:
                n = stack.pop()
                if n in comp:
                    continue
                comp.add(n)
                stack.extend(adj[n] - comp)
            seen |= comp
            members = sorted(comp)
            gold = self.golden(members)
            plan_pairs = []
            for a, b, sim, how in pairs:
                if a not in comp:
                    continue
                va, vb = self._views[a], self._views[b]
                cls = self.classify(va, vb, how, sim)
                master_id = gold["master_id"] if gold["master_id"] in (a, b) else \
                    sorted((a, b), key=lambda i: (-gold["legacy_scores"][i], i))[0]
                dup_id = b if master_id == a else a
                mv, dv = self._views[master_id], self._views[dup_id]
                dep = self.dependents(mv, dv)
                blockers = self.global_blockers() + dep["blockers"]
                fd = self.field_decisions(mv, dv)
                if "sector_id" in fd["conflicts"]:
                    blockers.append("sector_id differs: merging would discard one sector membership")
                plan_pairs.append({
                    "pair": [a, b], "name_match": how, "name_similarity": sim,
                    "legacy_similarity": round(legacy_sim(self._scorer, va["company_name"], vb["company_name"]), 4),
                    **cls, "proposed_master_id": master_id, "proposed_duplicate_id": dup_id,
                    "field_decisions": fd, "dependents": dep,
                    "execution": {"ready": False, "blockers": blockers}})
            groups.append({"members": [{"company_id": i, "name": self._views[i]["company_name"],
                                        "sector": self._views[i].get("sector_name"), "country": self._views[i].get("country")}
                                       for i in members], "golden_record": gold, "pairs": plan_pairs})
        return {"generated_at": now_iso(), "dry_run": True, "executed_operations": 0,
                "fk_info_used": bool(self.fk_info), "global_blockers": self.global_blockers(),
                "source_companies": len(self.source.companies()), "groups": groups,
                "summary": self.summarize(groups)}

    @staticmethod
    def summarize(groups: list[dict]) -> dict:
        cats, recs, review = defaultdict(int), defaultdict(int), defaultdict(int)
        pairs = [p for g in groups for p in g["pairs"]]
        for p in pairs:
            cats[p["category"]] += 1
            recs[p["recommendation"]] += 1
            review[p["review_level"]] += 1
        return {"groups": len(groups), "pairs": len(pairs), "by_category": dict(cats), "by_recommendation": dict(recs),
                "by_review_level": dict(review),
                "pairs_with_field_conflicts": sum(1 for p in pairs if p["field_decisions"]["conflicts"]),
                "pairs_where_legacy_mover_would_lose_rows": sum(1 for p in pairs if p["dependents"]["rows_at_risk"]),
                "total_rows_at_risk_with_legacy_mover": sum(p["dependents"]["rows_at_risk"] for p in pairs),
                "pairs_ready_to_execute": sum(1 for p in pairs if p["execution"]["ready"]),
                "manual_review_queue": sum(1 for p in pairs if p["recommendation"] in ("manual_review", "link_multi_sector")
                                           or p["review_level"] == "elevated")}


# --------------------------------------------------------------------------------------- reporting
def render_markdown(plan: dict) -> str:
    s = plan["summary"]
    out = ["# Dry-run merge plan", "",
           f"Generated {plan['generated_at']} - **no merge, move or delete was executed** "
           f"({plan['executed_operations']} operations).", "",
           f"Companies scanned: {plan['source_companies']} | candidate groups: {s['groups']} | pairs: {s['pairs']}",
           "", "| Category | Pairs |", "|---|---|"]
    out += [f"| {k} | {v} |" for k, v in sorted(s["by_category"].items())]
    out += ["", "| Recommendation | Pairs |", "|---|---|"] + [f"| {k} | {v} |" for k, v in sorted(s["by_recommendation"].items())]
    out += ["", f"- Pairs with field conflicts needing a human decision: **{s['pairs_with_field_conflicts']}**",
            f"- Pairs where the legacy ForeignKeyService would lose dependent rows: **{s['pairs_where_legacy_mover_would_lose_rows']}** "
            f"({s['total_rows_at_risk_with_legacy_mover']} rows)",
            f"- Manual-review queue: **{s['manual_review_queue']}**",
            f"- Pairs ready to execute: **{s['pairs_ready_to_execute']}** (global blockers below)", "", "## Global blockers", ""]
    out += [f"- {b}" for b in plan.get("global_blockers", GLOBAL_BLOCKERS)]
    for g in plan["groups"]:
        names_ = " / ".join(f"{m['name']} (#{m['company_id']}, {m['sector']})" for m in g["members"])
        out += ["", f"## {names_}", "",
                f"Golden Record: **#{g['golden_record']['master_id']}** - " + "; ".join(g["golden_record"]["rationale"])]
        for p in g["pairs"]:
            out += ["", f"### Pair {p['pair'][0]} - {p['pair'][1]}: `{p['category']}` -> **{p['recommendation']}** "
                        f"(confidence {p['confidence']}, review {p['review_level']})",
                    "", "Rationale: " + "; ".join(p["rationale"]) + ".",
                    f"Proposed master #{p['proposed_master_id']}, duplicate #{p['proposed_duplicate_id']}."]
            if p["field_decisions"]["conflicts"]:
                out.append("Field conflicts (human decision required): " + ", ".join(p["field_decisions"]["conflicts"]))
            moves = [f"{t['table']}: {t['duplicate_rows']} ({t['action']})" for t in p["dependents"]["tables"] if t["duplicate_rows"]]
            out.append("Dependent rows on the duplicate: " + (", ".join(moves) or "none"))
            if p["dependents"]["blockers"]:
                out.append("Legacy mover would lose: " + "; ".join(p["dependents"]["blockers"]))
    return "\n".join(out) + "\n"


def write_report(plan: dict, directory: str) -> tuple[str, str]:
    import os
    os.makedirs(directory, exist_ok=True)
    jp, mp = os.path.join(directory, "merge_plan.json"), os.path.join(directory, "merge_plan.md")
    json.dump(plan, open(jp, "w", encoding="utf-8"), indent=2, default=str)
    open(mp, "w", encoding="utf-8").write(render_markdown(plan))
    return jp, mp


def main(argv: list[str]) -> None:  # pragma: no cover - READ-ONLY report generation
    """python -m agentic.merge_plan [--snapshot tests/fixtures/duplicate_groups.json] [output_dir]
    Without --snapshot it reads Supabase with SELECT only. Either way nothing is merged or deleted."""
    args = argv[1:]
    if "--snapshot" in args:
        from agentic.merge_sources import InMemoryMergeSource
        k = args.index("--snapshot")
        source = InMemoryMergeSource(json.load(open(args[k + 1], encoding="utf-8")))
        args = args[:k] + args[k + 2:]
    else:
        from agentic.merge_sources import SupabaseMergeSource
        source = SupabaseMergeSource.from_env()
    fk_info = None
    if "--introspection" in args:           # delete rules etc. from the saved INTROSPECTION_SQL output (read-only file)
        from agentic import fk_introspect
        k = args.index("--introspection")
        fk_info = fk_introspect.analyze(json.load(open(args[k + 1], encoding="utf-8")))
        args = args[:k] + args[k + 2:]
    plan = MergePlanner(source, fk_info=fk_info).build()
    jp, mp = write_report(plan, args[0] if args else "data/merge_plan")
    print(json.dumps(plan["summary"], indent=2))
    print("wrote", jp, mp, "- nothing was merged or deleted")


if __name__ == "__main__":  # pragma: no cover
    main(sys.argv)
