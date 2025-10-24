import os
import sys
import logging
from unified_planning.io import PDDLReader
from unified_planning.exceptions import UPException, UPUsageError
import matplotlib.pyplot as plt
import csv
from collections import Counter, defaultdict
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
sys.setrecursionlimit(2000)


def find_domain_problem_pairs(base_path):
    pairs = []
    for root, _, files in os.walk(base_path):
        domain_files = [f for f in files if f.lower() == "domain.pddl"]
        problem_files = [f for f in files if f.lower().endswith(".pddl") and f.lower() != "domain.pddl"]
        for domain_file in domain_files:
            domain_path = os.path.join(root, domain_file)
            problem_path = os.path.join(root, problem_files[0]) if problem_files else None
            pairs.append((domain_path, problem_path))
    return pairs


def analyze_domains(pairs):
    reader = PDDLReader()
    stats = []

    for domain_path, problem_path in pairs:
        try:
            if problem_path:
                problem = reader.parse_problem(domain_path, problem_path)
            else:
                problem = reader.parse_problem(domain_path)

            domain_name = problem.name or os.path.basename(os.path.dirname(domain_path))

            stats.append({
                "name": domain_name,
                "path": domain_path,
                "actions": len(problem.actions),
                "fluents": len(problem.fluents),
                "types": len(problem.user_types),
                "status": "parsed"
            })
            logging.info(f"Loaded domain '{domain_name}' ({os.path.relpath(domain_path)})")

        except (UPException, UPUsageError) as e:
            logging.warning(f"Failed to parse {domain_path}: {type(e).__name__}: {e}")
            stats.append({
                "name": os.path.basename(os.path.dirname(domain_path)),
                "path": domain_path,
                "actions": 0,
                "fluents": 0,
                "types": 0,
                "status": f"error: {type(e).__name__}"
            })
        except RecursionError:
            logging.error(f"Recursion depth exceeded while parsing {domain_path}")
            stats.append({
                "name": os.path.basename(os.path.dirname(domain_path)),
                "path": domain_path,
                "actions": 0,
                "fluents": 0,
                "types": 0,
                "status": "error: recursion"
            })
        except Exception as e:
            logging.error(f"Unexpected error while parsing {domain_path}: {type(e).__name__}: {e}")
            stats.append({
                "name": os.path.basename(os.path.dirname(domain_path)),
                "path": domain_path,
                "actions": 0,
                "fluents": 0,
                "types": 0,
                "status": f"error: {type(e).__name__}"
            })

    return stats

def fluent_statistics(
    pairs,
    stats,
    output_dir="domain_analysis",
    histogram_path=None
):
    """
    Analyze and export fluent statistics from parsed domains.
    Saves results into:
        domain_analysis/
          ├── fluents_summary.csv
          ├── fluents_histogram.png
          └── fluents/
              ├── at.csv
              ├── on.csv
              └── ...
    """
    from unified_planning.io import PDDLReader
    from unified_planning import Environment

    output_dir = Path(output_dir)
    fluents_dir = output_dir / "fluents"
    output_dir.mkdir(exist_ok=True)
    fluents_dir.mkdir(exist_ok=True)

    env = Environment()
    env.error_used_name = False
    reader = PDDLReader()
    if hasattr(reader, "env"):
        reader.env = env

    all_fluents = []
    fluent_entries = defaultdict(list)

    for s, (domain_path, problem_path) in zip(stats, pairs):
        if s["status"] != "parsed":
            continue
        try:
            if problem_path:
                problem = reader.parse_problem(domain_path, problem_path)
            else:
                problem = reader.parse_problem(domain_path)
            domain_name = problem.name or os.path.basename(os.path.dirname(domain_path))

            for f in problem.fluents:
                if not f.type.is_bool_type():
                    continue  # only boolean (predicate) fluents
                f_name = f.name
                all_fluents.append(f_name)

                arity = len(f.signature)
                raw_repr = str(f)

                fluent_entries[f_name].append({
                    "fluent_name": f_name,
                    "domain_name": domain_name,
                    "domain_path": domain_path,
                    "arity": arity,
                    "raw_repr": raw_repr
                })

        except Exception as e:
            logging.warning(f"Could not re-parse fluents for {domain_path}: {e}")

    # === Compute frequency distribution ===
    fluent_counter = Counter(all_fluents)
    total_fluents = len(all_fluents)
    distinct_fluents = len(fluent_counter)

    print("\n=== FLUENT STATISTICS ===")
    print(f"Total fluents (including repeats): {total_fluents}")
    print(f"Distinct fluent names: {distinct_fluents}")

    # === Save summary CSV ===
    summary_csv = output_dir / "fluents_summary.csv"
    with open(summary_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["fluent_name", "frequency"])
        for name, freq in fluent_counter.most_common():
            writer.writerow([name, freq])
    print(f"[INFO] Saved fluent frequencies to {summary_csv}")

    # === Save per-fluent CSVs ===
    for f_name, entries in fluent_entries.items():
        fluent_csv = fluents_dir / f"{f_name}.csv"
        with open(fluent_csv, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["fluent_name", "domain_name", "domain_path", "arity", "raw_repr"]
            )
            writer.writeheader()
            writer.writerows(entries)
    print(f"[INFO] Saved individual fluent CSVs to {fluents_dir}")

    # === Histogram ===
    if histogram_path is None:
        histogram_path = output_dir / "fluents_histogram.png"

    plt.figure(figsize=(14, 7))
    plt.bar(range(len(fluent_counter)), list(fluent_counter.values()), tick_label=list(fluent_counter.keys()))
    plt.xticks(rotation=90, fontsize=7)
    plt.ylabel("Frequency")
    plt.title("Fluent (Predicate) Frequency across Domains")
    plt.tight_layout()
    plt.savefig(histogram_path, dpi=150)
    plt.close()
    print(f"[INFO] Saved histogram to {histogram_path}")

    return {
        "total_fluents": total_fluents,
        "distinct_fluents": distinct_fluents,
        "summary_csv": str(summary_csv),
        "fluents_dir": str(fluents_dir)
    }



def main():
    base_path = "pddl-instances"
    logging.info(f"Searching for PDDL domains under: {base_path}")

    pairs = find_domain_problem_pairs(base_path)
    logging.info(f"Found {len(pairs)} domain/problem pairs.")

    stats = analyze_domains(pairs)

    print("\n=== DOMAIN STATISTICS ===")
    print(f"Successfully parsed: {sum(1 for s in stats if s['status'] == 'parsed')}/{len(stats)} domains\n")

    for s in stats:
        print(f"- {s['name']}: actions={s['actions']}, fluents={s['fluents']}, types={s['types']} [{s['status']}]")

    print("\nDomains that failed to load are shown as warnings above.")

    fluent_statistics(pairs, stats)

if __name__ == "__main__":
    main()
