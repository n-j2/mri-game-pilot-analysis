from __future__ import annotations

import json
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
FIGURE_DIR = OUTPUT_DIR / "figures" / "npc_interaction_plots"

FIGURE_OUTPUT = FIGURE_DIR / "npc_interaction_frequency.png"
SUMMARY_OUTPUT = OUTPUT_DIR / "npc_interaction_summary.csv"
ISSUES_OUTPUT = OUTPUT_DIR / "npc_interaction_processing_issues.csv"

# Include only sessions recorded as finished.
INCLUDE_ONLY_FINISHED_SESSIONS = True

# Change to True if the figure should contain only participants who met
# the predefined meaningful-completion criteria.
INCLUDE_ONLY_MEANINGFUL_COMPLETERS = False

# Stop rather than accidentally count the same participant twice.
FAIL_ON_DUPLICATE_PARTICIPANT_ID = True


# =============================
# Meaningful-completion settings
# =============================

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


# =============================
# NPC definitions
# =============================

# NPCs are ordered according to the approximate order in which participants
# encounter them during the game.
NPC_IDS = [
    "yoga_student_highfields",
    "jogging_student_highfields",
    "radiographer_1",
    "radiographer_2",
    "radiographer_3",
    "MRI_Participant",
    "chris_statistician",
    "nitish",
    "prof_dineen_offices",
    "emma_library",
    "journal_student_library",
    "dr_jawahar_imh",
]

# Reader-facing NPC names used in the figure and summary table.
NPC_LABELS = {
    "yoga_student_highfields": "Yoga Student",
    "jogging_student_highfields": "Jogging Student",
    "radiographer_1": "Entrance Radiographer",
    "radiographer_2": "Console Radiographer",
    "radiographer_3": "Contrast Radiographer",
    "MRI_Participant": "Research Participant",
    "chris_statistician": "Statistician",
    "nitish": "Postgraduate Student",
    "prof_dineen_offices": "Neuroimaging Professor",
    "emma_library": "Librarian",
    "journal_student_library": "Journalling Student",
    "dr_jawahar_imh": "Clinician",
}


# =============================
# Stack definitions
# =============================

STACK_ORDER = [
    "once",
    "twice",
    "three_times",
    "four_times",
    "zero",
]

STACK_COLOURS = {
    "once": "#4C78A8",
    "twice": "#72A0C1",
    "three_times": "#A6CEE3",
    "four_times": "#D9EAF3",
    "zero": "#BDBDBD",
}

STACK_TEXT_COLOURS = {
    "once": "white",
    "twice": "black",
    "three_times": "black",
    "four_times": "black",
    "zero": "black",
}

STACK_LABELS = {
    "once": "1 conversation",
    "twice": "2 conversations",
    "three_times": "3 conversations",
    "four_times": "4 conversations",
    "zero": "No interaction",
}


# =============================
# General helper functions
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
    Count questions where answered == True, regardless of correctness.
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
    Retrieve the recorded number of unique NPCs interacted with.
    """
    game = get_dictionary(
        participant_data,
        "game",
    )

    telemetry = get_dictionary(
        game,
        "telemetry",
    )

    unique_npc_value = telemetry.get(
        "unique_npcs_interacted_with"
    )

    if unique_npc_value is not None:
        try:
            return int(unique_npc_value)
        except (TypeError, ValueError):
            pass

    npc_counts = telemetry.get(
        "npc_conversation_start_counts"
    )

    if isinstance(npc_counts, dict):
        return len(npc_counts)

    return None


def meets_meaningful_completion(
    participant_data: dict[str, Any],
) -> bool:
    """
    Apply the study's predefined meaningful-completion criteria.
    """
    pre_mcq = get_dictionary(
        participant_data,
        "pre_mcq",
    )

    post_mcq = get_dictionary(
        participant_data,
        "post_mcq",
    )

    pre_attempted = count_attempted_questions(
        pre_mcq
    )

    post_attempted = count_attempted_questions(
        post_mcq
    )

    unique_npcs = get_unique_npc_count(
        participant_data
    )

    npc_requirement_met = (
        unique_npcs is not None
        and unique_npcs >= MEANINGFUL_MIN_NPCS
    )

    return (
        npc_requirement_met
        and pre_attempted >= MEANINGFUL_MIN_PRE_MCQ_ATTEMPTED
        and post_attempted >= MEANINGFUL_MIN_POST_MCQ_ATTEMPTED
    )


def parse_conversation_count(
    raw_value: Any,
    source_file: str,
    participant_id: str,
    npc_id: str,
    issues: list[dict[str, Any]],
) -> int:
    """
    Convert an NPC conversation count to a non-negative integer.

    Invalid values are recorded as processing issues and treated as zero.
    """
    if raw_value is None:
        return 0

    if isinstance(raw_value, bool):
        issues.append({
            "source_file": source_file,
            "participant_id": participant_id,
            "npc_id": npc_id,
            "issue": "Conversation count was Boolean rather than integer",
            "observed_value": raw_value,
        })

        return 0

    try:
        numeric_value = float(raw_value)

    except (TypeError, ValueError):
        issues.append({
            "source_file": source_file,
            "participant_id": participant_id,
            "npc_id": npc_id,
            "issue": "Conversation count was not numeric",
            "observed_value": raw_value,
        })

        return 0

    if not numeric_value.is_integer():
        issues.append({
            "source_file": source_file,
            "participant_id": participant_id,
            "npc_id": npc_id,
            "issue": "Conversation count was not a whole number",
            "observed_value": raw_value,
        })

        return 0

    conversation_count = int(numeric_value)

    if conversation_count < 0:
        issues.append({
            "source_file": source_file,
            "participant_id": participant_id,
            "npc_id": npc_id,
            "issue": "Conversation count was negative",
            "observed_value": raw_value,
        })

        return 0

    if conversation_count > 4:
        issues.append({
            "source_file": source_file,
            "participant_id": participant_id,
            "npc_id": npc_id,
            "issue": "Conversation count exceeded expected maximum of 4",
            "observed_value": conversation_count,
        })

    return conversation_count


# =============================
# Load participant data
# =============================

def load_npc_interaction_data(
    data_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Read all participant JSON files and return:

    1. Participant-by-NPC conversation counts in long format
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

    interaction_rows: list[dict[str, Any]] = []
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
                "npc_id": "",
                "issue": "Could not read JSON file",
                "observed_value": repr(exc),
            })

            continue

        if not isinstance(participant_data, dict):
            processing_issues.append({
                "source_file": json_file.name,
                "participant_id": "",
                "npc_id": "",
                "issue": "Top-level JSON value was not an object",
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
                "npc_id": "",
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
                "npc_id": "",
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

        game = get_dictionary(
            participant_data,
            "game",
        )

        telemetry = get_dictionary(
            game,
            "telemetry",
        )

        raw_npc_counts = telemetry.get(
            "npc_conversation_start_counts",
            {},
        )

        if not isinstance(raw_npc_counts, dict):
            processing_issues.append({
                "source_file": json_file.name,
                "participant_id": participant_id,
                "npc_id": "",
                "issue": (
                    "npc_conversation_start_counts "
                    "was missing or invalid"
                ),
                "observed_value": repr(raw_npc_counts),
            })

            raw_npc_counts = {}

        # Record unexpected NPC identifiers without including them in the
        # predefined 12-NPC analysis.
        unexpected_npc_ids = sorted(
            set(raw_npc_counts.keys())
            - set(NPC_IDS)
        )

        for unexpected_npc_id in unexpected_npc_ids:
            processing_issues.append({
                "source_file": json_file.name,
                "participant_id": participant_id,
                "npc_id": unexpected_npc_id,
                "issue": "Unexpected NPC identifier",
                "observed_value": raw_npc_counts.get(
                    unexpected_npc_id
                ),
            })

        # Create one row for every participant × NPC combination.
        # NPCs absent from the dictionary are assigned zero interactions.
        for npc_id in NPC_IDS:
            conversation_count = parse_conversation_count(
                raw_value=raw_npc_counts.get(
                    npc_id,
                    0,
                ),
                source_file=json_file.name,
                participant_id=participant_id,
                npc_id=npc_id,
                issues=processing_issues,
            )

            interaction_rows.append({
                "participant_id": participant_id,
                "source_file": json_file.name,
                "npc_id": npc_id,
                "npc_label": NPC_LABELS[npc_id],
                "conversation_count": conversation_count,
                "interacted_at_least_once":
                    conversation_count >= 1,
                "meaningful_completion":
                    meaningful_completion,
            })

    interactions_df = pd.DataFrame(
        interaction_rows
    )

    participants_df = pd.DataFrame(
        participant_rows
    )

    issues_df = pd.DataFrame(
        processing_issues
    )

    if interactions_df.empty:
        raise RuntimeError(
            "No participant NPC-interaction data remained after "
            "applying the inclusion settings."
        )

    return (
        interactions_df,
        participants_df,
        issues_df,
    )


# =============================
# Descriptive summary
# =============================

def create_npc_summary(
    interactions_df: pd.DataFrame,
    included_participants: int,
) -> pd.DataFrame:
    """
    Create one summary row per NPC.
    """
    summary_rows: list[dict[str, Any]] = []

    for npc_id in NPC_IDS:
        npc_data = interactions_df[
            interactions_df["npc_id"] == npc_id
        ]

        counts = npc_data[
            "conversation_count"
        ]

        number_zero = int(
            (counts == 0).sum()
        )

        number_once = int(
            (counts == 1).sum()
        )

        number_twice = int(
            (counts == 2).sum()
        )

        number_three_times = int(
            (counts == 3).sum()
        )

        number_four_times = int(
            (counts == 4).sum()
        )

        number_more_than_four = int(
            (counts > 4).sum()
        )

        number_at_least_once = int(
            (counts >= 1).sum()
        )

        percentage_at_least_once = (
            number_at_least_once
            / included_participants
            * 100
        )

        total_conversation_starts = int(
            counts.sum()
        )

        summary_rows.append({
            "npc_id": npc_id,
            "npc_label": NPC_LABELS[npc_id],
            "participants_not_interacting":
                number_zero,
            "participants_interacting_once":
                number_once,
            "participants_interacting_twice":
                number_twice,
            "participants_interacting_three_times":
                number_three_times,
            "participants_interacting_four_times":
                number_four_times,
            "participants_interacting_more_than_four_times":
                number_more_than_four,
            "participants_interacting_at_least_once":
                number_at_least_once,
            "percentage_interacting_at_least_once":
                percentage_at_least_once,
            "total_conversation_starts":
                total_conversation_starts,
        })

    return pd.DataFrame(
        summary_rows
    )


def validate_participant_npc_rows(
    interactions_df: pd.DataFrame,
    included_participants: int,
) -> None:
    """
    Confirm that every included participant contributes one row for
    each of the 12 predefined NPCs.
    """
    expected_rows = (
        included_participants
        * len(NPC_IDS)
    )

    observed_rows = len(
        interactions_df
    )

    if observed_rows != expected_rows:
        raise RuntimeError(
            "Unexpected number of participant × NPC rows. "
            f"Expected {expected_rows}, observed {observed_rows}."
        )

    rows_per_participant = (
        interactions_df
        .groupby("participant_id")
        .size()
    )

    invalid_participants = rows_per_participant[
        rows_per_participant != len(NPC_IDS)
    ]

    if not invalid_participants.empty:
        raise RuntimeError(
            "One or more participants did not have exactly "
            f"{len(NPC_IDS)} NPC rows:\n"
            f"{invalid_participants}"
        )


def validate_maximum_conversation_count(
    interactions_df: pd.DataFrame,
) -> None:
    """
    Stop if the observed data contain more than four conversation starts
    with a single NPC, because the figure categories end at four.
    """
    maximum_count = int(
        interactions_df[
            "conversation_count"
        ].max()
    )

    if maximum_count > 4:
        affected_rows = interactions_df[
            interactions_df["conversation_count"] > 4
        ][
            [
                "participant_id",
                "npc_id",
                "conversation_count",
            ]
        ]

        raise RuntimeError(
            "At least one participant initiated a conversation with "
            "an NPC more than four times. The current figure categories "
            "end at 4 conversations.\n\n"
            f"{affected_rows.to_string(index=False)}"
        )


# =============================
# Plot category selection
# =============================

def get_displayed_stack_categories(
    maximum_observed_count: int,
) -> list[str]:
    """
    Return the frequency categories required by the observed data.

    The zero-conversation category is always included so that every NPC bar
    represents the full included participant sample and therefore sums to
    N participants / 100%.
    """
    categories: list[str] = []

    if maximum_observed_count >= 1:
        categories.append("once")

    if maximum_observed_count >= 2:
        categories.append("twice")

    if maximum_observed_count >= 3:
        categories.append("three_times")

    if maximum_observed_count >= 4:
        categories.append("four_times")

    # Keep zero conversations at the top of each stacked bar so the existing
    # 1–4 conversation segments retain their original position from baseline.
    categories.append("zero")

    return categories


def get_category_counts(
    summary_df: pd.DataFrame,
    category: str,
) -> np.ndarray:
    """
    Retrieve participant counts for one stack category.
    """
    column_lookup = {
        "once": "participants_interacting_once",
        "twice": "participants_interacting_twice",
        "three_times":
            "participants_interacting_three_times",
        "four_times":
            "participants_interacting_four_times",
        "zero": "participants_not_interacting",
    }

    column = column_lookup[category]

    return summary_df[
        column
    ].to_numpy(dtype=float)


# =============================
# Plot creation
# =============================

def add_stack_labels(
    axis,
    x_positions: np.ndarray,
    stack_counts: np.ndarray,
    bottoms: np.ndarray,
    total_participants: int,
    text_colour: str,
) -> None:
    """
    Add count and percentage labels to non-empty bar segments.

    Format:
        n (x%)

    Example:
        18 (69%)
    """
    for (
        x_position,
        participant_count,
        bottom,
    ) in zip(
        x_positions,
        stack_counts,
        bottoms,
    ):
        if participant_count <= 0:
            continue

        percentage = (
            participant_count
            / total_participants
            * 100
        )

        label = (
            f"{int(participant_count)} "
            f"({percentage:.0f}%)"
        )

        # Use slightly smaller text for very small stacked sections.
        if participant_count <= 1:
            font_size = 6

        elif participant_count <= 2:
            font_size = 6.5

        else:
            font_size = 7.5

        axis.text(
            x_position,
            bottom + participant_count / 2,
            label,
            ha="center",
            va="center",
            fontsize=font_size,
            fontweight="bold",
            color=text_colour,
            clip_on=True,
            zorder=5,
        )


def plot_npc_interaction_frequency(
    summary_df: pd.DataFrame,
    interactions_df: pd.DataFrame,
    included_participants: int,
    output_path: Path,
) -> None:
    """
    Create the stacked NPC interaction-frequency bar chart.

    Each bar represents the full included participant sample. Segments show the number of participants who did not interact with the NPC,
    or who initiated 1, 2, 3, or 4 conversations with that NPC, so every bar
    sums to the included participant total.
    """
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    maximum_observed_count = int(
        interactions_df[
            "conversation_count"
        ].max()
    )

    displayed_categories = (
        get_displayed_stack_categories(
            maximum_observed_count
        )
    )

    x_positions = np.arange(
        len(NPC_IDS)
    )

    figure, axis = plt.subplots(
        figsize=(16, 7.5)
    )

    bottoms = np.zeros(
        len(NPC_IDS),
        dtype=float,
    )

    for category in displayed_categories:
        stack_counts = get_category_counts(
            summary_df,
            category,
        )

        axis.bar(
            x_positions,
            stack_counts,
            width=0.74,
            bottom=bottoms,
            color=STACK_COLOURS[category],
            edgecolor="white",
            linewidth=1.0,
            label=STACK_LABELS[category],
        )

        add_stack_labels(
            axis=axis,
            x_positions=x_positions,
            stack_counts=stack_counts,
            bottoms=bottoms,
            total_participants=included_participants,
            text_colour=STACK_TEXT_COLOURS[category],
        )

        bottoms += stack_counts

    axis.set_xticks(
        x_positions
    )

    axis.set_xticklabels(
        [
            NPC_LABELS[npc_id]
            for npc_id in NPC_IDS
        ],
        rotation=32,
        ha="right",
        rotation_mode="anchor",
    )

    maximum_bar_height = int(
        bottoms.max()
    )

    axis.set_ylim(
        0,
        maximum_bar_height + 2,
    )

    axis.yaxis.set_major_locator(
        MaxNLocator(integer=True)
    )

    axis.set_ylabel(
        "Number of participants"
    )

    axis.set_xlabel(
        "Non-player character"
    )

    axis.set_title(
        "Frequency of NPC conversation initiations"
    )

    axis.grid(
        axis="y",
        alpha=0.22,
    )

    axis.set_axisbelow(True)

    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)

    legend_handles = [
        Patch(
            facecolor=STACK_COLOURS[category],
            label=STACK_LABELS[category],
        )
        for category in displayed_categories
    ]

    axis.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.25),
        ncol=len(displayed_categories),
        frameon=False,
    )

    figure.tight_layout()

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
        interactions_df,
        participants_df,
        issues_df,
    ) = load_npc_interaction_data(
        DATA_DIR
    )

    included_participants = int(
        participants_df[
            "included_in_figure"
        ].sum()
    )

    excluded_participants = int(
        (
            ~participants_df[
                "included_in_figure"
            ]
        ).sum()
    )

    validate_participant_npc_rows(
        interactions_df=interactions_df,
        included_participants=included_participants,
    )

    validate_maximum_conversation_count(
        interactions_df=interactions_df,
    )

    summary_df = create_npc_summary(
        interactions_df=interactions_df,
        included_participants=included_participants,
    )

    summary_df.to_csv(
        SUMMARY_OUTPUT,
        index=False,
    )

    if not issues_df.empty:
        issues_df.to_csv(
            ISSUES_OUTPUT,
            index=False,
        )

    elif ISSUES_OUTPUT.exists():
        # Remove an outdated issues file left by an earlier run.
        ISSUES_OUTPUT.unlink()

    plot_npc_interaction_frequency(
        summary_df=summary_df,
        interactions_df=interactions_df,
        included_participants=included_participants,
        output_path=FIGURE_OUTPUT,
    )

    maximum_observed_count = int(
        interactions_df[
            "conversation_count"
        ].max()
    )

    print(
        "NPC interaction analysis complete."
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
        "Maximum conversations with one NPC by one participant: "
        f"{maximum_observed_count}"
    )

    print(
        f"Figure: "
        f"{FIGURE_OUTPUT.resolve()}"
    )

    print(
        f"Summary table: "
        f"{SUMMARY_OUTPUT.resolve()}"
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