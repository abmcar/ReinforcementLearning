"""Prompt templates for JOB-11 binary recommendation SFT data."""

from __future__ import annotations

from typing import Iterable, Mapping


def _join_non_empty(lines: Iterable[str]) -> str:
    return "\n".join(line for line in lines if line.strip())


def format_worker_history(history_items: list[Mapping[str, str]]) -> str:
    """Render prior worker participation history in reverse chronological order."""
    if not history_items:
        return "No prior participation history is available for this worker."

    lines = []
    for idx, item in enumerate(history_items, start=1):
        lines.append(
            (
                f"{idx}. {item['title']} | category={item['category']} | "
                f"industry={item['industry']} | outcome={item['outcome']}"
            )
        )
    return "\n".join(lines)


def format_project_brief(brief_items: list[str]) -> str:
    """Render compact project brief bullet points."""
    if not brief_items:
        return "No brief details available."
    return "\n".join(f"- {item}" for item in brief_items)


def build_binary_prompt(
    *,
    objective: str,
    worker_profile: Mapping[str, str],
    worker_history: list[Mapping[str, str]],
    project_profile: Mapping[str, str],
) -> dict[str, str]:
    """Build a binary Yes/No recommendation prompt."""
    if objective == "worker":
        goal = (
            "Judge whether this project is a good opportunity for the worker, "
            "considering fit, likely reward, and prior success patterns."
        )
    else:
        goal = (
            "Judge whether this worker is a strong candidate for the requester, "
            "considering worker quality and historical fit with similar projects."
        )

    system = _join_non_empty(
        [
            "You are an assistant for a crowdsourcing design platform.",
            "Given a worker profile, recent worker history, and one candidate project,",
            "answer whether the worker-project match should be recommended.",
            "Respond with exactly one token: Yes or No.",
            goal,
        ]
    )

    user = _join_non_empty(
        [
            f"Objective: {objective}",
            "",
            "Worker Profile:",
            (
                f"- worker_quality={worker_profile['worker_quality']}\n"
                f"- total_prior_entries={worker_profile['hist_entries']}\n"
                f"- prior_wins={worker_profile['hist_wins']}\n"
                f"- prior_win_rate={worker_profile['hist_win_rate']}\n"
                f"- average_award={worker_profile['hist_avg_award']}\n"
                f"- top_categories={worker_profile['top_categories']}"
            ),
            "",
            "Recent Worker History:",
            format_worker_history(worker_history),
            "",
            "Candidate Project:",
            (
                f"- title={project_profile['title']}\n"
                f"- category={project_profile['category']}\n"
                f"- sub_category={project_profile['sub_category']}\n"
                f"- industry={project_profile['industry']}\n"
                f"- package_name={project_profile['package_name']}\n"
                f"- total_awards={project_profile['total_awards']}\n"
                f"- days_until_deadline={project_profile['days_until_deadline']}\n"
                f"- current_participants={project_profile['participants_count']}"
            ),
            "Project Brief:",
            format_project_brief(project_profile["brief_items"]),
            "",
            (
                "Question: Should this worker be recommended to this project "
                "for the stated objective? Answer Yes or No."
            ),
        ]
    )

    return {
        "system": system,
        "user": user,
        "prompt": f"<system>\n{system}\n</system>\n<user>\n{user}\n</user>",
    }
