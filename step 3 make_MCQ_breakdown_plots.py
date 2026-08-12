
from __future__ import annotations

import json
from itertools import product
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator


# =============================
# User settings
# =============================

DATA_DIR = Path("data")

OUTPUT_DIR = Path("analysis_output")
FIGURE_DIR = OUTPUT_DIR / "figures" / "mcq_option_plots"

FIGURE_OUTPUT = FIGURE_DIR / "mcq_option_distributions.png"
COUNTS_OUTPUT = OUTPUT_DIR / "mcq_option_response_counts.csv"
ISSUES_OUTPUT = OUTPUT_DIR / "mcq_option_processing_issues.csv"

# Include only sessions recorded as finished.
INCLUDE_ONLY_FINISHED_SESSIONS = True

# Change to True if the figure should contain only participants who met
# the predefined meaningful-completion criteria.
INCLUDE_ONLY_MEANINGFUL_COMPLETERS = False

# Stop rather than accidentally double-count a participant.
FAIL_ON_DUPLICATE_PARTICIPANT_ID = True

# Meaningful-completion definition from the study protocol.
MEANINGFUL_MIN_NPCS = 8
MEANINGFUL_MIN_PRE_MCQ_ATTEMPTED = 5
MEANINGFUL_MIN_POST_MCQ_ATTEMPTED = 5

QUESTION_IDS = [
    "q1",
    "q2",
    "q3",
    "q4",
    "q5",
    "q6",
    "q7",
]

QUESTION_LABELS = {
    "q1": "Q1",
    "q2": "Q2",
    "q3": "Q3",
    "q4": "Q4",
    "q5": "Q5",
    "q6": "Q6",
    "q7": "Q7",
}

PHASES = {
    "pre_mcq": "Pre-game MCQ",
    "post_mcq": "Post-game MCQ",
}

# Option A is correct for every question.
OPTION_ORDER = [
    "a",
    "b",
    "c",
    "d",
    "not_answered",
]

OPTION_LABELS = {
    "a": "A (correct)",
    "b": "B",
    "c": "C",
    "d": "D",
    "not_answered": "Not answered",
}

# Green for the correct option, muted red shades for incorrect options,
# and grey for unanswered questions.
OPTION_COLOURS = {
    "a": "#4C956C",
    "b": "#C65D57",
    "c": "#D98276",
    "d": "#E7AAA2",
    "not_answered": "#BDBDBD",
}


# =============================
# Validation and extraction
# =============================

def get_dictionary(
    parent: dict[str, Any],
    key: str,
) -> dict[str, Any]:
    """
    Safely retrieve a nested dictionary.
    """
    value = parent.get(key, {})

    if isinstance(value, dict):
        return value

    return {}


def count_attempted_questions(
    mcq_block: dict[str, Any],
) -> int:
    """
    Count questions marked answered == True, regardless of correctness.
    """
    answers = get_dictionary(mcq_block, "answers")

    attempted = 0

    for question_id in QUESTION_IDS:
        question_data = answers.get(question_id, {})

        if (
            isinstance(question_data, dict)
            and question_data.get("answered") is True
        ):
            attempted += 1

    return attempted


def get_unique_npc_count(
    participant_data: dict[str, Any],
) -> int | None:
    """
    Retrieve the unique NPC count from gameplay telemetry.
    """
    game = get_dictionary(participant_data, "game")
    telemetry = get_dictionary(game, "telemetry")

    value = telemetry.get("unique_npcs_interacted_with")

    if value is not None:
        try:
            return int(value)
        except (TypeError, ValueError):
            pass

    npc_counts = telemetry.get("npc_conversation_start_counts")

    if isinstance(npc_counts, dict):
        return len(npc_counts)

    return None


def meets_meaningful_completion(
    participant_data: dict[str, Any],
) -> bool:
    """
    Apply the predefined meaningful-completion criteria.
    """
    pre_mcq = get_dictionary(participant_data, "pre_mcq")
    post_mcq = get_dictionary(participant_data, "post_mcq")

    pre_attempted = count_attempted_questions(pre_mcq)
    post_attempted = count_attempted_questions(post_mcq)
    unique_npcs = get_unique_npc_count(participant_data)

    npc_requirement_met = (
        unique_npcs is not None
        and unique_npcs >= MEANINGFUL_MIN_NPCS
    )

    return (
        npc_requirement_met
        and pre_attempted >= MEANINGFUL_MIN_PRE_MCQ_ATTEMPTED
        and post_attempted >= MEANINGFUL_MIN_POST_MCQ_ATTEMPTED
    )


def extract_selected_option(
    participant_id: str,
    source_file: str,
    phase_name: str,
    question_id: str,
    mcq_block: dict[str, Any],
    issues: list[dict[str, Any]],
) -> str:
    """
    Extract the nominal selected option ID for one question.

    Randomised presentation order is intentionally ignored. Responses are
    analysed using the stored nominal question IDs and nominal option IDs.

    Missing, null or unanswered responses are returned as not_answered.
    Unexpected option IDs are recorded and treated as not_answered.
    """
    answers = get_dictionary(mcq_block, "answers")
    question_data = answers.get(question_id, {})

    if not isinstance(question_data, dict):
        issues.append({
            "source_file": source_file,
            "participant_id": participant_id,
            "phase": phase_name,
            "question_id": question_id,
            "issue": "Question response object missing or invalid",
            "observed_value": "",
        })

        return "not_answered"

    answered = question_data.get("answered")
    selected_option = question_data.get("selected_option_id")

    if answered is not True or selected_option is None:
        return "not_answered"

    selected_option_text = str(
        selected_option
    ).strip().lower()

    if selected_option_text == "":
        return "not_answered"

    if selected_option_text not in {"a", "b", "c", "d"}:
        issues.append({
            "source_file": source_file,
            "participant_id": participant_id,
            "phase": phase_name,
            "question_id": question_id,
            "issue": "Unexpected selected_option_id",
            "observed_value": selected_option_text,
        })

        return "not_answered"

    return selected_option_text


def load_mcq_responses(
    data_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Read all participant JSON files and return:

    1. Long-format item-response data
    2. Participant inclusion information
    3. Processing and validation issues
    """
    json_files = sorted(
        data_dir.glob("*.json")
    )

    if not json_files:
        raise RuntimeError(
            f"No JSON files were found in: {data_dir.resolve()}"
        )

    response_rows: list[dict[str, Any]] = []
    participant_rows: list[dict[str, Any]] = []
    processing_issues: list[dict[str, Any]] = []

    observed_participant_ids: set[str] = set()

    for json_file in json_files:
        try:
            with json_file.open(
                "r",
                encoding="utf-8",
            ) as file:
                participant_data = json.load(file)

        except Exception as exc:
            processing_issues.append({
                "source_file": json_file.name,
                "participant_id": "",
                "phase": "",
                "question_id": "",
                "issue": "Could not read JSON file",
                "observed_value": repr(exc),
            })

            continue

        if not isinstance(participant_data, dict):
            processing_issues.append({
                "source_file": json_file.name,
                "participant_id": "",
                "phase": "",
                "question_id": "",
                "issue": "Top-level JSON value is not an object",
                "observed_value": "",
            })

            continue

        participant_id_value = participant_data.get(
            "participant_id"
        )

        if participant_id_value is None:
            participant_id = json_file.stem

            processing_issues.append({
                "source_file": json_file.name,
                "participant_id": participant_id,
                "phase": "",
                "question_id": "",
                "issue": "participant_id missing; filename stem used",
                "observed_value": "",
            })

        else:
            participant_id = str(
                participant_id_value
            ).strip()

        if participant_id in observed_participant_ids:
            message = (
                "Duplicate participant_id detected: "
                f"{participant_id}"
            )

            if FAIL_ON_DUPLICATE_PARTICIPANT_ID:
                raise RuntimeError(message)

            processing_issues.append({
                "source_file": json_file.name,
                "participant_id": participant_id,
                "phase": "",
                "question_id": "",
                "issue": "Duplicate participant_id",
                "observed_value": participant_id,
            })

        observed_participant_ids.add(
            participant_id
        )

        study_status = str(
            participant_data.get(
                "study_status",
                "",
            )
        ).strip().lower()

        meaningful_completion = (
            meets_meaningful_completion(
                participant_data
            )
        )

        include_participant = True
        exclusion_reason = ""

        if (
            INCLUDE_ONLY_FINISHED_SESSIONS
            and study_status != "finished"
        ):
            include_participant = False
            exclusion_reason = (
                "study_status was not finished"
            )

        elif (
            INCLUDE_ONLY_MEANINGFUL_COMPLETERS
            and not meaningful_completion
        ):
            include_participant = False
            exclusion_reason = (
                "did not meet meaningful-completion criteria"
            )

        participant_rows.append({
            "participant_id": participant_id,
            "source_file": json_file.name,
            "study_status": study_status,
            "meaningful_completion": meaningful_completion,
            "included_in_figure": include_participant,
            "exclusion_reason": exclusion_reason,
        })

        if not include_participant:
            continue

        for phase_name in PHASES:
            mcq_block = get_dictionary(
                participant_data,
                phase_name,
            )

            for question_id in QUESTION_IDS:
                selected_option = extract_selected_option(
                    participant_id=participant_id,
                    source_file=json_file.name,
                    phase_name=phase_name,
                    question_id=question_id,
                    mcq_block=mcq_block,
                    issues=processing_issues,
                )

                response_rows.append({
                    "participant_id": participant_id,
                    "source_file": json_file.name,
                    "phase": phase_name,
                    "question_id": question_id,
                    "selected_option": selected_option,
                    "correct": selected_option == "a",
                    "meaningful_completion":
                        meaningful_completion,
                })

    responses_df = pd.DataFrame(
        response_rows
    )

    participants_df = pd.DataFrame(
        participant_rows
    )

    issues_df = pd.DataFrame(
        processing_issues
    )

    if responses_df.empty:
        raise RuntimeError(
            "No participant MCQ responses remained after applying "
            "the inclusion settings."
        )

    return (
        responses_df,
        participants_df,
        issues_df,
    )


# =============================
# Descriptive count table
# =============================

def create_counts_table(
    responses_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create a complete phase × question × option count table.

    Zero-count combinations are retained so the exported CSV and plot
    always contain every question and response category.
    """
    included_participants = responses_df[
        "participant_id"
    ].nunique()

    combinations = pd.DataFrame(
        product(
            PHASES.keys(),
            QUESTION_IDS,
            OPTION_ORDER,
        ),
        columns=[
            "phase",
            "question_id",
            "selected_option",
        ],
    )

    observed_counts = (
        responses_df
        .groupby(
            [
                "phase",
                "question_id",
                "selected_option",
            ],
            dropna=False,
        )
        .size()
        .reset_index(name="count")
    )

    counts_df = combinations.merge(
        observed_counts,
        on=[
            "phase",
            "question_id",
            "selected_option",
        ],
        how="left",
    )

    counts_df["count"] = (
        counts_df["count"]
        .fillna(0)
        .astype(int)
    )

    counts_df["percentage"] = (
        counts_df["count"]
        / included_participants
        * 100
    )

    counts_df["phase_label"] = counts_df[
        "phase"
    ].map(PHASES)

    counts_df["question_label"] = counts_df[
        "question_id"
    ].map(QUESTION_LABELS)

    counts_df["option_label"] = counts_df[
        "selected_option"
    ].map(OPTION_LABELS)

    counts_df["correct_option"] = (
        counts_df["selected_option"] == "a"
    )

    final_column_order = [
        "phase",
        "phase_label",
        "question_id",
        "question_label",
        "selected_option",
        "option_label",
        "correct_option",
        "count",
        "percentage",
    ]

    return counts_df[
        final_column_order
    ]


def validate_question_totals(
    counts_df: pd.DataFrame,
    expected_participants: int,
) -> None:
    """
    Verify that each question totals the included participant count.
    """
    totals = (
        counts_df
        .groupby(
            [
                "phase",
                "question_id",
            ]
        )["count"]
        .sum()
    )

    invalid_totals = totals[
        totals != expected_participants
    ]

    if not invalid_totals.empty:
        raise RuntimeError(
            "One or more phase/question totals did not equal the "
            "included participant count:\n"
            f"{invalid_totals}"
        )


# =============================
# Figure creation
# =============================

def add_segment_labels(
    axis,
    x_positions: np.ndarray,
    counts: np.ndarray,
    bottoms: np.ndarray,
    total_participants: int,
) -> None:
    """
    Add count and percentage labels to the centre of each non-empty
    stacked-bar segment.

    Labels use the format:
        n (x%)

    For example:
        5 (19%)
    """
    for x_position, count, bottom in zip(
        x_positions,
        counts,
        bottoms,
    ):
        if count <= 0:
            continue

        percentage = (
            count
            / total_participants
            * 100
        )

        label = (
            f"{int(count)} "
            f"({percentage:.0f}%)"
        )

        # Slightly smaller text for a one-participant segment.
        if count <= 1:
            font_size = 6.5

        elif count <= 2:
            font_size = 7

        else:
            font_size = 8

        axis.text(
            x_position,
            bottom + count / 2,
            label,
            ha="center",
            va="center",
            fontsize=font_size,
            fontweight="bold",
            color="black",
            clip_on=True,
            zorder=5,
        )


def plot_mcq_option_distributions(
    counts_df: pd.DataFrame,
    included_participants: int,
    output_path: Path,
) -> None:
    """
    Create vertically arranged pre- and post-game stacked bar charts.
    """
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure, axes = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(11, 8.8),
        sharex=True,
        sharey=True,
    )

    x_positions = np.arange(
        len(QUESTION_IDS)
    )

    question_tick_labels = [
        QUESTION_LABELS[question_id]
        for question_id in QUESTION_IDS
    ]

    for axis, phase_name in zip(
        axes,
        PHASES,
    ):
        phase_data = counts_df[
            counts_df["phase"] == phase_name
        ]

        bottoms = np.zeros(
            len(QUESTION_IDS),
            dtype=float,
        )

        for option in OPTION_ORDER:
            option_counts: list[int] = []

            for question_id in QUESTION_IDS:
                matching_row = phase_data[
                    (
                        phase_data["question_id"]
                        == question_id
                    )
                    & (
                        phase_data["selected_option"]
                        == option
                    )
                ]

                if matching_row.empty:
                    option_counts.append(0)

                else:
                    option_counts.append(
                        int(
                            matching_row[
                                "count"
                            ].iloc[0]
                        )
                    )

            option_counts_array = np.array(
                option_counts,
                dtype=float,
            )

            axis.bar(
                x_positions,
                option_counts_array,
                width=0.72,
                bottom=bottoms,
                color=OPTION_COLOURS[option],
                edgecolor="white",
                linewidth=1.0,
                label=OPTION_LABELS[option],
            )

            add_segment_labels(
                axis=axis,
                x_positions=x_positions,
                counts=option_counts_array,
                bottoms=bottoms,
                total_participants=included_participants,
            )

            bottoms += option_counts_array

        axis.set_title(
            f"{PHASES[phase_name]} "
            f"(N={included_participants})"
        )

        axis.set_ylabel(
            "Number of participants"
        )

        axis.set_ylim(
            0,
            included_participants + 1,
        )

        axis.yaxis.set_major_locator(
            MaxNLocator(integer=True)
        )

        axis.grid(
            axis="y",
            alpha=0.22,
        )

        axis.set_axisbelow(True)

        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

        # Show Q1-Q7 beneath both the pre-game and post-game plots.
        axis.set_xticks(
            x_positions
        )

        axis.set_xticklabels(
            question_tick_labels
        )

        axis.tick_params(
            axis="x",
            labelbottom=True,
        )

        axis.set_xlabel(
            "MCQ question"
        )

    legend_handles = [
        Patch(
            facecolor=OPTION_COLOURS[option],
            label=OPTION_LABELS[option],
        )
        for option in OPTION_ORDER
    ]

    figure.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=5,
        frameon=False,
        bbox_to_anchor=(0.5, 0.015),
    )

    figure.suptitle(
        "Pre- and post-game MCQ option selections",
        fontsize=15,
    )

    figure.subplots_adjust(
        top=0.91,
        bottom=0.12,
        left=0.09,
        right=0.98,
        hspace=0.38,
    )

    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)


# =============================
# Main
# =============================

def main() -> None:
    if not DATA_DIR.exists():
        raise FileNotFoundError(
            f"Could not find data directory: "
            f"{DATA_DIR.resolve()}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        responses_df,
        participants_df,
        issues_df,
    ) = load_mcq_responses(
        DATA_DIR
    )

    included_participants = int(
        participants_df[
            "included_in_figure"
        ].sum()
    )

    counts_df = create_counts_table(
        responses_df
    )

    validate_question_totals(
        counts_df=counts_df,
        expected_participants=included_participants,
    )

    counts_df.to_csv(
        COUNTS_OUTPUT,
        index=False,
    )

    if not issues_df.empty:
        issues_df.to_csv(
            ISSUES_OUTPUT,
            index=False,
        )

    elif ISSUES_OUTPUT.exists():
        # Remove a stale issues file left by an earlier run.
        ISSUES_OUTPUT.unlink()

    plot_mcq_option_distributions(
        counts_df=counts_df,
        included_participants=included_participants,
        output_path=FIGURE_OUTPUT,
    )

    excluded_participants = int(
        (
            ~participants_df[
                "included_in_figure"
            ]
        ).sum()
    )

    print(
        "MCQ option-distribution analysis complete."
    )

    print(
        f"JSON files found: "
        f"{len(list(DATA_DIR.glob('*.json')))}"
    )

    print(
        f"Participants included: "
        f"{included_participants}"
    )

    print(
        f"Participants excluded: "
        f"{excluded_participants}"
    )

    print(
        f"Figure: "
        f"{FIGURE_OUTPUT.resolve()}"
    )

    print(
        f"Counts table: "
        f"{COUNTS_OUTPUT.resolve()}"
    )

    if not issues_df.empty:
        print(
            f"Processing warnings: "
            f"{len(issues_df)}"
        )

        print(
            f"See: "
            f"{ISSUES_OUTPUT.resolve()}"
        )


if __name__ == "__main__":
    main()
