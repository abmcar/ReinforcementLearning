import os
import json
import csv
from pathlib import Path
from dateutil.parser import parse
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# 设置中文字体，防止图表中的中文显示为方块
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'Songti SC']
plt.rcParams['axes.unicode_minus'] = False


def main():
    # ========================================================
    # 1. 核心路径自动对齐（基于本脚本在 src/data/eda.py 的位置精准推算）
    # ========================================================
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent.parent  # 往上跳三级找到项目根目录

    data_dir = project_root / "data"
    project_dir = data_dir / "project"

    # 双重保存路径：
    # 路径 A（团队规范要求）：docs/figures/
    figures_dir_docs = project_root / "docs" / "figures"
    figures_dir_docs.mkdir(parents=True, exist_ok=True)

    # 路径 B（本地直观查看）：就在当前代码同级目录下的 figures/ 里
    figures_dir_local = current_file.parent / "figures"
    figures_dir_local.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("🔍 【路径自动对齐检查】")
    print(f"▶ 你的代码文件绝对路径: {current_file}")
    print(f"▶ 自动识别的项目根目录: {project_root}")
    print(f"▶ 检查 data 文件夹是否存在: {data_dir.exists()} -> ({data_dir})")
    print(f"▶ 检查 project 数据集文件夹是否存在: {project_dir.exists()} -> ({project_dir})")
    print("=" * 60)

    # ========================================================
    # 2. 绘制 图1：Worker 质量分布
    # ========================================================
    worker_qualities = []
    worker_csv = data_dir / "worker_quality.csv"

    if worker_csv.exists():
        with open(worker_csv, "r", encoding="utf-8") as csvfile:
            csvreader = csv.reader(csvfile)
            for line in csvreader:
                if not line or line[0] == "worker_id": continue
                try:
                    quality_raw = float(line[1])
                    if quality_raw > 0.0:
                        worker_qualities.append(quality_raw / 100.0)
                except ValueError:
                    pass

        if worker_qualities:
            df_worker = pd.DataFrame({'Worker Quality': worker_qualities})
            plt.figure(figsize=(10, 6))
            sns.histplot(df_worker['Worker Quality'], bins=30, kde=True, color='#5c9eb7')
            plt.title('图1: 众包工人 (Worker) 质量分数分布', fontsize=15)
            plt.xlabel('归一化质量分数 (0-1)', fontsize=12)
            plt.ylabel('工人数 (频数)', fontsize=12)

            # 同时保存到两个地方，确保万无一失
            plt.savefig(figures_dir_docs / 'JOB-01-worker_quality_distribution.png', dpi=300, bbox_inches='tight')
            plt.savefig(figures_dir_local / 'JOB-01-worker_quality_distribution.png', dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✅ 图1 生成成功！已同时放入以下两个位置，请前去查看：\n   位置①: {figures_dir_docs}\n   位置②: {figures_dir_local}")
        else:
            print("❌ 图1 渲染失败：文件中未提取到有效的质量分数数据。")
    else:
        print(f"❌ 图1 渲染失败：在项目里依然找不到 {worker_csv} 文件！")

    # ========================================================
    # 3. 绘制 图2 & 图3：项目信息分布
    # ========================================================
    project_list_csv = data_dir / "project_list.csv"
    project_entries = []
    project_durations = []

    if not project_list_csv.exists():
        print(f"❌ 图2/3 渲染失败：找不到列表索引文件 {project_list_csv}")
    elif not project_dir.exists():
        print(f"\n❌ 图2/3 渲染失败：找不到大规模数据集文件夹 {project_dir}")
        print("💡 [新手提示]: 请用电脑自带的文件管理器打开你的项目，去 `data` 文件夹下亲眼看一下。")
        print("   里面是不是只有一个叫 `project.zip` 之类的压缩包？如果是的话，GitHub 默认不解压它，你需要手动解压出 `project` 文件夹放进去！")
    else:
        with open(project_list_csv, "r", encoding="utf-8") as f:
            project_list_lines = f.readlines()

        for line in project_list_lines:
            parts = line.strip('\n').split(',')
            if not parts or parts[0] == "project_id": continue
            try:
                project_id = int(parts[0])
                project_entries.append(int(parts[1]))

                proj_file = project_dir / f"project_{project_id}.txt"
                if proj_file.exists():
                    with open(proj_file, "r", encoding="utf-8") as pf:
                        text = json.load(pf)
                        start_date = parse(text["start_date"])
                        deadline = parse(text["deadline"])
                        days = (deadline - start_date).days
                        if days > 0: project_durations.append(days)
            except Exception:
                pass

        if project_entries:
            plt.figure(figsize=(10, 6))
            sns.histplot(pd.DataFrame({'Entry Count': project_entries})['Entry Count'], bins=50, kde=True,
                         color='#f28e2b')
            plt.title('图2: 单个项目收到的作品(Entry)提交数量分布', fontsize=15)
            plt.xlabel('提交作品数', fontsize=12)
            plt.ylabel('项目数量 (频数)', fontsize=12)
            plt.savefig(figures_dir_docs / 'JOB-01-project_entry_count_distribution.png', dpi=300, bbox_inches='tight')
            plt.savefig(figures_dir_local / 'JOB-01-project_entry_count_distribution.png', dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✅ 图2 生成成功！已双重归档。")
        else:
            print("❌ 图2 失败：没有合法的项目提交数据。")

        if project_durations:
            plt.figure(figsize=(10, 2))
            sns.boxplot(x=pd.DataFrame({'Duration': project_durations})['Duration'], color='#59a14f')
            plt.title('图3: 项目周期分布箱线图', fontsize=15)
            plt.xlabel('项目持续时间 (天)', fontsize=12)
            plt.savefig(figures_dir_docs / 'JOB-01-project_duration_boxplot.png', dpi=300, bbox_inches='tight')
            plt.savefig(figures_dir_local / 'JOB-01-project_duration_boxplot.png', dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✅ 图3 生成成功！已双重归档。")
        else:
            print("❌ 图3 失败：未能解析出项目的具体起止天数（请确认 project 文件夹内是否有各个项目的 txt 文件）。")


if __name__ == "__main__":
    main()