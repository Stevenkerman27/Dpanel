"""把 47 个研究分支收敛成一条 master 的规划器（默认 dry-run）。

背景：master 只有 1 个 Initial commit（222 文件、无 panel/），47 个分支都是从它派生出来的
研究快照，每个分支都自带一份当时版本的 panel/。因此「逐支 merge」会产生 47 次互相冲突的
三方合并；可行做法是：
  Phase 1  基线：把整合分支（含 P0+P1 的 panel/、62 个测试）线性落到 master
  Phase 2  文件级导入：分支里 integ 没有的新文件（docs/tools/test 回执），按「谁最后改谁优先」
           逐路径挑一份，按来源分支成批提交
  Phase 3  内容冲突只报告：同一路径在 integ 与分支里都有但内容不同 → 人工裁决，脚本不覆盖

用法:
  python tools/consolidate_master.py                # dry-run，写 runs/consolidation_plan.json
  python tools/consolidate_master.py --apply        # 在 runs/consol_git 工作树里真正执行
"""
import json, os, subprocess, sys, collections

REPO = r"D:\3D\Projects\panel\runs\integ_git"
OUT = r"D:\3D\Projects\panel\runs\consolidation_plan.json"
BASELINE = "origin/master"
INTEG = "origin/research/master-integration-p1"

def git(*args, cwd=REPO):
    r = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    return r.returncode, r.stdout, r.stderr

def refs():
    code, so, _ = git("for-each-ref", "--format=%(refname:short)|%(committerdate:unix)|%(objectname)", "refs/remotes/origin")
    out = []
    for line in so.splitlines():
        name, ts, sha = line.split("|")
        if name.endswith("/HEAD") or name in (BASELINE, INTEG) or name == "origin":
            continue
        out.append((name, int(ts), sha))
    return out

def files_changed(ref, base):
    _, so, _ = git("diff", "--name-only", base, ref)
    return [f for f in so.splitlines() if f.strip()]

def tree_set(ref):
    _, so, _ = git("ls-tree", "-r", "--name-only", ref)
    return set(so.splitlines())

def main():
    apply = "--apply" in sys.argv
    integ_files = tree_set(INTEG)
    branches = refs()
    # 每个路径由「最后修改它的分支」提供
    owner = {}
    conflicts = collections.defaultdict(list)
    for name, ts, sha in branches:
        mb_code, mb, _ = git("merge-base", BASELINE, name)
        mb = mb.strip()
        if not mb:
            continue
        for f in files_changed(name, mb):
            if f in integ_files:
                _, a, _ = git("rev-parse", INTEG + ":" + f)
                r = subprocess.run(["git", "rev-parse", name + ":" + f], cwd=REPO, capture_output=True, text=True)
                if r.returncode == 0 and r.stdout.strip() != a.strip():
                    conflicts[f].append(name)
                continue
            prev = owner.get(f)
            if prev is None or ts > prev[1]:
                owner[f] = (name, ts)
    plan = {
        "baseline": BASELINE,
        "integ": INTEG,
        "branch_count": len(branches),
        "import_paths": len(owner),
        "conflict_paths": len(conflicts),
        "imports": [{"path": p, "from": b} for p, (b, _) in sorted(owner.items())],
        "conflicts": [{"path": p, "branches": sorted(set(v))[:5]} for p, v in sorted(conflicts.items())],
    }
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(plan, fh, ensure_ascii=False, indent=1)
    by_branch = collections.Counter(v[0] for v in owner.values())
    print("branches:", len(branches), "| new paths to import:", len(owner), "| content conflicts:", len(conflicts))
    print("top import sources:")
    for b, n in by_branch.most_common(8):
        print(f"  {b:52s} {n}")
    tops = collections.Counter(p.split("/")[0] for p in owner)
    print("by top dir:", dict(tops.most_common(8)))
    print("conflict sample:")
    for p, v in list(conflicts.items())[:10]:
        print("  " + p)
    if apply:
        return apply_plan(plan)
    print("dry-run only; re-run with --apply")

def apply_plan(plan):
    wt = r"D:\3D\Projects\panel\runs\consol_git"
    if not os.path.isdir(wt):
        git("worktree", "add", "--detach", wt, BASELINE)
    git("checkout", "-B", "consolidat/master", BASELINE, cwd=wt)
    code, so, se = git("merge", "--no-ff", "-m", "merge: P0+P1 panel baseline (issue #62 audit branch)", INTEG, cwd=wt)
    print("phase1 merge:", code, (so + se).strip().splitlines()[-1] if (so + se).strip() else "")
    if code != 0:
        print("phase1 conflict, aborting"); return 1
    groups = collections.defaultdict(list)
    for item in plan["imports"]:
        groups[item["from"]].append(item["path"])
    for branch, paths in sorted(groups.items()):
        # 分块 checkout，避免命令行超长
        for i in range(0, len(paths), 150):
            chunk = paths[i:i+150]
            code, so, se = git("checkout", branch, "--", *chunk, cwd=wt)
            if code != 0:
                print("checkout failed", branch, (se or so).strip()[:200])
        git("add", "-A", cwd=wt)
        git("commit", "-q", "-m", "import: " + branch.replace("origin/", "") + " (" + str(len(paths)) + " paths)", cwd=wt)
        print("imported", branch, len(paths), "paths")
    _, so, _ = git("log", "--oneline", "-3", cwd=wt)
    print(so)
    return 0

if __name__ == "__main__":
    sys.exit(main() or 0)
