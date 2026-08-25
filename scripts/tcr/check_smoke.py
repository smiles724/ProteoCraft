#!/usr/bin/env python
"""Check the §9 smoke acceptance criteria across the A/B/C/D arms."""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path("/root/mingxuan/smoke_tcr_abcd_20260820")
SPEC = json.loads(
    Path(
        "/root/Bingchen/experiments/20260819_liubo_cdr6_native_20x200"
        "/inputs/case_specs/ALGRRIPPV_beta.json"
    ).read_text()
)
SCAFFOLD = SPEC["binder_sequence"]
CDR_START = SPEC["cdr_start_0based"]
CDR_END = SPEC["cdr_end_0based_exclusive"]
UNIFORM_FALLBACK = 0.29957  # log(20) * 0.1

GRAD_RE = re.compile(
    r"\|contact\|=([\d.eE+-]+) \|tcr\|=([\d.eE+-]+) \|structure\|=([\d.eE+-]+) "
    r"\| cos\(contact,struct\)=([+-][\d.]+) cos\(contact,esmc\)=([+-][\d.]+) "
    r"cos\(contact,tcr\)=([+-][\d.]+) cos\(tcr,struct\)=([+-][\d.]+) "
    r"cos\(tcr,esmc\)=([+-][\d.]+) \| cdr3_pep=([\d.]+) fw_pmhc=([\d.]+) "
    r"r_pep=([\d.]+)"
)


def best_sequence(arm_dir: Path) -> str | None:
    result = arm_dir / "hallucination_result.txt"
    if not result.exists():
        return None
    for line in result.read_text().splitlines():
        if line.startswith("Best Sequence:"):
            return line.split(":", 1)[1].strip()
    return None


def parse_grads(arm_dir: Path) -> list[dict]:
    log = arm_dir / "run.log"
    if not log.exists():
        return []
    rows = []
    for line in log.read_text(errors="replace").splitlines():
        m = GRAD_RE.search(line)
        if m:
            keys = [
                "contact_norm", "tcr_norm", "structure_norm",
                "cos_contact_struct", "cos_contact_esmc", "cos_contact_tcr",
                "cos_tcr_struct", "cos_tcr_esmc",
                "cdr3_pep", "fw_pmhc", "r_pep",
            ]
            rows.append({k: float(v) for k, v in zip(keys, m.groups())})
    return rows


def verifier_terms(arm_dir: Path) -> dict:
    """Scan candidate artifacts for TCR loss_dict entries."""
    out: dict[str, list[float]] = {}
    art = arm_dir / "artifacts"
    if not art.exists():
        return out
    for path in sorted(art.rglob("*.json")):
        try:
            payload = json.loads(path.read_text())
        except Exception:
            continue
        stack = [payload]
        while stack:
            node = stack.pop()
            if isinstance(node, dict):
                for key in (
                    "tcr_contact_loss", "cdr3_peptide_contact_nll",
                    "framework_pmhc_contact_nll", "r_pep",
                    "tcr_prior_hard", "tcr_contact_weighted",
                ):
                    if key in node and isinstance(node[key], (int, float)):
                        out.setdefault(key, []).append(float(node[key]))
                stack.extend(node.values())
            elif isinstance(node, list):
                stack.extend(node)
    return out


def main() -> None:
    print(f"scaffold CDR3 [{CDR_START},{CDR_END}) = "
          f"{SCAFFOLD[CDR_START:CDR_END]}\n")
    summary = {}
    for arm in ("A", "B", "C", "D", "E", "F"):
        arm_dir = ROOT / f"arm_{arm}"
        if not arm_dir.exists():
            continue
        seq = best_sequence(arm_dir)
        grads = parse_grads(arm_dir)
        terms = verifier_terms(arm_dir)
        print(f"===== ARM {arm} =====")
        if seq is None:
            print("  (no result yet)\n")
            continue

        # 1. framework must be untouched
        fw_ok = (
            seq[:CDR_START] == SCAFFOLD[:CDR_START]
            and seq[CDR_END:] == SCAFFOLD[CDR_END:]
            and len(seq) == len(SCAFFOLD)
        )
        diffs = [i for i, (a, b) in enumerate(zip(seq, SCAFFOLD)) if a != b]
        outside = [i for i in diffs if not (CDR_START <= i < CDR_END)]
        print(f"  CDR3 designed          : {seq[CDR_START:CDR_END]}")
        print(f"  framework unchanged    : {fw_ok} "
              f"(mutations outside CDR3: {len(outside)})")

        # 2. proposal gradients
        if grads:
            cn = [g["contact_norm"] for g in grads]
            tn = [g["tcr_norm"] for g in grads]
            print(f"  proposal grad rows     : {len(grads)}")
            print(f"  |contact_grad|         : min={min(cn):.3e} max={max(cn):.3e} "
                  f"nonzero={sum(1 for v in cn if v > 0)}/{len(cn)}")
            print(f"  |tcr_grad|             : min={min(tn):.3e} max={max(tn):.3e} "
                  f"nonzero={sum(1 for v in tn if v > 0)}/{len(tn)}")
            cdr3 = [g["cdr3_pep"] for g in grads]
            fw = [g["fw_pmhc"] for g in grads]
            rp = [g["r_pep"] for g in grads]
            if any(v > 0 for v in cdr3):
                print(f"  cdr3_pep nll           : {min(cdr3):.3f}..{max(cdr3):.3f} "
                      f"({len(set(cdr3))} distinct)")
                print(f"  fw_pmhc nll            : {min(fw):.3f}..{max(fw):.3f}")
                print(f"  r_pep                  : {min(rp):.3f}..{max(rp):.3f}")
                cs = [g["cos_contact_struct"] for g in grads]
                ct = [g["cos_contact_tcr"] for g in grads]
                print(f"  cos(contact,structure) : {min(cs):+.3f}..{max(cs):+.3f}")
                if any(abs(v) > 0 for v in ct):
                    print(f"  cos(contact,tcr_prior) : {min(ct):+.3f}..{max(ct):+.3f}")
        else:
            print("  proposal grad rows     : 0 (both channels off)")

        # 3. verifier terms
        if terms:
            for key, values in sorted(terms.items()):
                distinct = len(set(round(v, 6) for v in values))
                print(f"  verifier {key:28s}: n={len(values):3d} "
                      f"distinct={distinct:3d} range={min(values):.4f}..{max(values):.4f}")
                if key == "tcr_prior_hard":
                    near = [v for v in values
                            if abs(abs(v) - UNIFORM_FALLBACK) < 1e-4]
                    print(f"    uniform-fallback (|v|≈{UNIFORM_FALLBACK}) hits: "
                          f"{len(near)}/{len(values)}")
        summary[arm] = {
            "cdr3": seq[CDR_START:CDR_END],
            "framework_ok": fw_ok,
            "mutations_outside_cdr3": len(outside),
        }
        print()

    print("===== CDR3 across arms (same seed) =====")
    for arm, info in summary.items():
        print(f"  {arm}: {info['cdr3']}")
    cdrs = {a: i["cdr3"] for a, i in summary.items()}
    if len(cdrs) > 1:
        print(f"  distinct CDR3 solutions: {len(set(cdrs.values()))}/{len(cdrs)}")
    (ROOT / "smoke_check_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
