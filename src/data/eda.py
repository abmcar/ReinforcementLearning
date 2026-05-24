import json
import csv
import statistics
from pathlib import Path
from dateutil.parser import parse
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# 设置中文字体，防止图表中的中文显示为方块
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'Songti SC']
plt.rcParams['axes.unicode_minus'] = False

# Entry 文件每个分页包含的最大记录数（与 sample_read_data.py 一致）
ENTRY_PAGE_SIZE = 24


def main():
    # ========================================================
    # 1. 路径设置
    # ========================================================
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent.parent

    data_dir = project_root / "data"
    project_dir = data_dir / "project"
    entry_dir = data_dir / "entry"

    figures_dir = project_root / "docs" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"Project root: {project_root}")
    print(f"Data dir exists: {data_dir.exists()}")
    print(f"Project dir exists: {project_dir.exists()}")
    print(f"Entry dir exists: {entry_dir.exists()}")
    print("=" * 60)

    # ========================================================
    # 2. Worker 质量分布
    # ========================================================
    worker_qualities = []
    worker_filtered_count = 0
    worker_total_count = 0
    worker_csv = data_dir / "worker_quality.csv"

    if worker_csv.exists():
        with open(worker_csv, "r", encoding="utf-8") as csvfile:
            csvreader = csv.reader(csvfile)
            next(csvreader, None)  # skip header
            for line in csvreader:
                if not line:
                    continue
                worker_total_count += 1
                try:
                    quality_raw = float(line[1])
                    if quality_raw > 0.0:
                        worker_qualities.append(quality_raw / 100.0)
                    else:
                        worker_filtered_count += 1
                except ValueError:
                    pass

        print(f"\n[Worker] Total: {worker_total_count}, "
              f"Valid (quality > 0): {len(worker_qualities)}, "
              f"Filtered (quality <= 0): {worker_filtered_count}")

        if worker_qualities:
            print(f"[Worker] Mean: {statistics.mean(worker_qualities):.4f}, "
                  f"Median: {statistics.median(worker_qualities):.4f}, "
                  f"Stdev: {statistics.stdev(worker_qualities):.4f}, "
                  f"Min: {min(worker_qualities):.4f}, "
                  f"Max: {max(worker_qualities):.4f}")

            df_worker = pd.DataFrame({'Worker Quality': worker_qualities})
            plt.figure(figsize=(10, 6))
            sns.histplot(df_worker['Worker Quality'], bins=30, kde=True, color='#5c9eb7')
            plt.title('Worker Quality Distribution', fontsize=15)
            plt.xlabel('Normalized Quality Score (0-1)', fontsize=12)
            plt.ylabel('Count', fontsize=12)
            plt.savefig(figures_dir / 'JOB-01-worker_quality_distribution.png',
                        dpi=300, bbox_inches='tight')
            plt.close()
            print("[Worker] Figure saved.")
        else:
            print("[Worker] WARNING: No valid quality scores found.")
    else:
        print(f"[Worker] ERROR: {worker_csv} not found.")

    # ========================================================
    # 3. Project 信息分布
    # ========================================================
    project_list_csv = data_dir / "project_list.csv"
    project_entries = []
    project_durations = []
    project_start_dates = []
    project_deadline_dates = []
    project_parse_errors = 0

    if not project_list_csv.exists():
        print(f"\n[Project] ERROR: {project_list_csv} not found.")
    else:
        # 从 project_list.csv 读取 entry count（无需 project/ 目录）
        with open(project_list_csv, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip('\n').split(',')
                if not parts:
                    continue
                try:
                    project_entries.append(int(parts[1]))
                except (ValueError, IndexError):
                    pass

        print(f"\n[Project] Total projects: {len(project_entries)}")

        if project_entries:
            print(f"[Project] Entry count - "
                  f"Mean: {statistics.mean(project_entries):.2f}, "
                  f"Median: {statistics.median(project_entries):.2f}, "
                  f"Stdev: {statistics.stdev(project_entries):.2f}, "
                  f"Min: {min(project_entries)}, "
                  f"Max: {max(project_entries)}")

            plt.figure(figsize=(10, 6))
            sns.histplot(
                pd.DataFrame({'Entry Count': project_entries})['Entry Count'],
                bins=50, kde=True, color='#f28e2b')
            plt.title('Project Entry Count Distribution', fontsize=15)
            plt.xlabel('Entry Count', fontsize=12)
            plt.ylabel('Number of Projects', fontsize=12)
            plt.savefig(figures_dir / 'JOB-01-project_entry_count_distribution.png',
                        dpi=300, bbox_inches='tight')
            plt.close()
            print("[Project] Entry count figure saved.")

        # 从 project/ 目录读取详细信息（如果可用）
        if project_dir.exists():
            with open(project_list_csv, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip('\n').split(',')
                    if not parts or parts[0] == "project_id":
                        continue
                    try:
                        project_id = int(parts[0])
                        proj_file = project_dir / f"project_{project_id}.txt"
                        if proj_file.exists():
                            with open(proj_file, "r", encoding="utf-8") as pf:
                                text = json.load(pf)
                            start_date = parse(text["start_date"])
                            deadline = parse(text["deadline"])
                            project_start_dates.append(start_date)
                            project_deadline_dates.append(deadline)
                            days = (deadline - start_date).days
                            if days > 0:
                                project_durations.append(days)
                    except (json.JSONDecodeError, KeyError, ValueError) as e:
                        project_parse_errors += 1
                        print(f"  WARNING: Failed to parse project_{parts[0]}: {e}")

            if project_parse_errors:
                print(f"[Project] Parse errors: {project_parse_errors}")

            if project_start_dates:
                print(f"[Project] Time range: "
                      f"{min(project_start_dates)} ~ {max(project_deadline_dates)}")

            if project_durations:
                print(f"[Project] Duration (days) - "
                      f"Mean: {statistics.mean(project_durations):.2f}, "
                      f"Median: {statistics.median(project_durations):.2f}, "
                      f"Stdev: {statistics.stdev(project_durations):.2f}, "
                      f"Min: {min(project_durations)}, "
                      f"Max: {max(project_durations)}")

                plt.figure(figsize=(10, 2))
                sns.boxplot(
                    x=pd.DataFrame({'Duration': project_durations})['Duration'],
                    color='#59a14f')
                plt.title('Project Duration Distribution (Box Plot)', fontsize=15)
                plt.xlabel('Duration (days)', fontsize=12)
                plt.savefig(figures_dir / 'JOB-01-project_duration_boxplot.png',
                            dpi=300, bbox_inches='tight')
                plt.close()
                print("[Project] Duration figure saved.")
        else:
            print(f"[Project] WARNING: {project_dir} not found. "
                  "Skipping project detail analysis (duration, time range). "
                  "Extract project data files to run full analysis.")

    # ========================================================
    # 4. Entry 数据分析
    # ========================================================
    if not entry_dir.exists():
        print(f"\n[Entry] WARNING: {entry_dir} not found. "
              "Skipping entry analysis. "
              "Extract entry data files to run full analysis.")
    elif not project_list_csv.exists():
        print(f"\n[Entry] ERROR: {project_list_csv} not found, "
              "cannot enumerate entries.")
    else:
        print(f"\n[Entry] Analyzing entry data...")
        award_values = []
        finalist_count = 0
        winner_count = 0
        total_entries = 0
        entry_times = []
        per_worker_entries: dict[int, int] = {}
        entry_parse_errors = 0

        with open(project_list_csv, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip('\n').split(',')
                if not parts:
                    continue
                try:
                    project_id = int(parts[0])
                    entry_count = int(parts[1])
                except (ValueError, IndexError):
                    continue

                k = 0
                while k < entry_count:
                    entry_file = entry_dir / f"entry_{project_id}_{k}.txt"
                    if not entry_file.exists():
                        k += ENTRY_PAGE_SIZE
                        continue
                    try:
                        with open(entry_file, "r", encoding="utf-8") as ef:
                            data = json.load(ef)
                        for item in data.get("results", []):
                            total_entries += 1

                            # worker tracking
                            worker_id = item.get("author")
                            if worker_id is not None:
                                wid = int(worker_id)
                                per_worker_entries[wid] = \
                                    per_worker_entries.get(wid, 0) + 1

                            # award_value
                            av = item.get("award_value")
                            if av is not None:
                                try:
                                    award_values.append(float(av))
                                except (ValueError, TypeError):
                                    pass

                            # finalist / winner
                            if item.get("finalist"):
                                finalist_count += 1
                            if item.get("winner"):
                                winner_count += 1

                            # entry time
                            ts = item.get("entry_created_at")
                            if ts:
                                try:
                                    entry_times.append(parse(ts))
                                except (ValueError, TypeError):
                                    pass
                    except (json.JSONDecodeError, KeyError, ValueError) as e:
                        entry_parse_errors += 1
                        if entry_parse_errors <= 5:
                            print(f"  WARNING: Failed to parse "
                                  f"{entry_file.name}: {e}")

                    k += ENTRY_PAGE_SIZE

        print(f"[Entry] Total entries parsed: {total_entries}")
        print(f"[Entry] Parse errors: {entry_parse_errors}")
        print(f"[Entry] Unique workers with entries: {len(per_worker_entries)}")

        if total_entries > 0:
            print(f"[Entry] Finalist: {finalist_count} "
                  f"({finalist_count / total_entries * 100:.2f}%)")
            print(f"[Entry] Winner: {winner_count} "
                  f"({winner_count / total_entries * 100:.2f}%)")

        if award_values:
            non_zero_awards = [v for v in award_values if v > 0]
            print(f"[Entry] award_value non-zero: "
                  f"{len(non_zero_awards)}/{len(award_values)}")
            if non_zero_awards:
                print(f"[Entry] award_value (non-zero) - "
                      f"Mean: {statistics.mean(non_zero_awards):.2f}, "
                      f"Median: {statistics.median(non_zero_awards):.2f}, "
                      f"Stdev: {statistics.stdev(non_zero_awards):.2f}")

            plt.figure(figsize=(10, 6))
            sns.histplot(non_zero_awards, bins=50, kde=True, color='#e15759')
            plt.title('Entry Award Value Distribution (Non-Zero)', fontsize=15)
            plt.xlabel('Award Value ($)', fontsize=12)
            plt.ylabel('Count', fontsize=12)
            plt.savefig(
                figures_dir / 'JOB-01-entry_award_value_distribution.png',
                dpi=300, bbox_inches='tight')
            plt.close()
            print("[Entry] Award value figure saved.")

        if per_worker_entries:
            wec = list(per_worker_entries.values())
            print(f"[Entry] Per-worker entries - "
                  f"Mean: {statistics.mean(wec):.2f}, "
                  f"Median: {statistics.median(wec):.2f}, "
                  f"Max: {max(wec)}")

            plt.figure(figsize=(10, 6))
            sns.histplot(wec, bins=50, kde=True, color='#76b7b2')
            plt.title('Per-Worker Entry Count Distribution', fontsize=15)
            plt.xlabel('Number of Entries per Worker', fontsize=12)
            plt.ylabel('Number of Workers', fontsize=12)
            plt.savefig(
                figures_dir / 'JOB-01-per_worker_entry_count.png',
                dpi=300, bbox_inches='tight')
            plt.close()
            print("[Entry] Per-worker entry count figure saved.")

        if entry_times:
            print(f"[Entry] Time range: {min(entry_times)} ~ "
                  f"{max(entry_times)}")

    print("\n" + "=" * 60)
    print("EDA complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
