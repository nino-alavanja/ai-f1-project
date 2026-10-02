"""
llm/assistant.py
─────────────────
OpenAI-powered F1 analyst assistant.

It receives the model's predictions as structured context and uses
GPT-4o to answer natural-language questions about the race.

This is a grounded LLM analyst:
  • The neural network generates predictions + feature explanations
  • We inject structured race context into the system prompt
  • The LLM answers from that context and clearly marks unavailable data
"""

import os
import sys
from openai import OpenAI

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import OPENAI_API_KEY, LLM_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS


def build_race_context(
    race_name: str,
    circuit_name: str,
    country: str,
    date: str,
    predictions: list[dict],   # output from predict_race()
    telemetry_context: str | None = None,
) -> str:
    """
    Serialise the prediction results into a compact text block
    that becomes the system context for the LLM.
    """
    lines = [
        f"RACE: {race_name}",
        f"CIRCUIT: {circuit_name}, {country}",
        f"DATE: {date}",
        "",
        "PREDICTED FINISHING ORDER (from neural network model):",
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
    ]

    for p in predictions:
        driver   = p["driver_id"].replace("_", " ").title()
        team     = p["constructor_id"].replace("_", " ").title()
        win_pct  = round(p["win_probability"]    * 100, 1)
        pod_pct  = round(p["podium_probability"] * 100, 1)
        qual     = p["qual_position"]
        fd       = p["feature_dict"]
        ci_low   = p.get("rank_ci_low")
        ci_high  = p.get("rank_ci_high")
        rank_std = p.get("rank_std")
        ci_text = (
            f"CI90: P{ci_low:.0f}-P{ci_high:.0f} | Rank std: {rank_std:.2f}"
            if ci_low is not None and ci_high is not None and rank_std is not None
            else "CI90: unavailable"
        )

        lines.append(
            f"P{p['predicted_position']:2d}. {driver:<22} ({team})"
        )
        lines.append(
            f"      Qualifying: P{qual} | Win: {win_pct}% | Podium: {pod_pct}% | {ci_text}"
        )
        lines.append(
            f"      Champ pts entering: {fd['champ_points_entering']:.0f} (P{fd['champ_position_entering']}) | "
            f"Team pts: {fd['con_points_entering']:.0f} (P{fd['con_position_entering']}) | "
            f"      Recent avg finish: {fd['recent_avg_finish']:.1f} | "
            f"DNF rate: {fd['recent_dnf_rate']*100:.0f}% | "
            f"Circuit avg: {fd['circuit_avg_finish']:.1f} ({fd['circuit_appearances']} prior starts)"
        )
        lines.append("")

    lines.extend([
        "QUALIFYING TELEMETRY CONTEXT:",
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        telemetry_context or "Not available for this race.",
        "",
    ])

    return "\n".join(lines)


SYSTEM_PROMPT_TEMPLATE = """
You are an expert Formula 1 analyst assistant. A machine learning model
has generated race predictions for an upcoming Grand Prix. Your job is
to explain, discuss, and answer questions about those predictions.

You have access to:
  1. The model's full predicted finishing order, probabilities, and uncertainty intervals
  2. The key features the model used (qualifying position, championship
     standings, recent form, circuit history, constructor strength)
  3. Qualifying telemetry contextual evidence when FastF1 data is available.
     This telemetry was NOT used as a RankNet model input.
  4. Your own broad F1 knowledge (driver styles, circuit characteristics,
     team strategies, historical context)

Guidelines:
  • Be specific and data-driven. Reference the numbers from the prediction context.
  • Acknowledge model uncertainty — this is a probabilistic prediction, not a guarantee.
  • Use only the provided prediction, feature, uncertainty, weather, and telemetry
    context for race-specific numeric claims.
  • Use telemetry only when it is present in the qualifying telemetry context.
    If telemetry is unavailable, say so rather than inventing telemetry values.
  • You may use general F1 knowledge for broad explanation, but do not invent
    race-specific facts, lap times, strategy, tyre, or setup information that is
    not in the provided context.
  • Keep answers concise but insightful (3–5 sentences unless asked for more).
  • If asked something outside the race context, politely steer back.

PREDICTION CONTEXT:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{race_context}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""".strip()


class F1Assistant:
    """
    Stateful conversation assistant backed by GPT-4o.
    Maintains conversation history so follow-up questions work naturally.
    """

    def __init__(self, race_context: str):
        self.client  = OpenAI(api_key=OPENAI_API_KEY)
        self.system  = SYSTEM_PROMPT_TEMPLATE.format(race_context=race_context)
        self.history: list[dict] = []

    def ask(self, question: str) -> str:
        """
        Send a question to the LLM and return its response.
        The conversation history is maintained across calls.
        """
        self.history.append({"role": "user", "content": question})

        response = self.client.chat.completions.create(
            model=LLM_MODEL,
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS,
            messages=[
                {"role": "system", "content": self.system},
                *self.history,
            ],
        )

        answer = response.choices[0].message.content
        self.history.append({"role": "assistant", "content": answer})
        return answer

    def reset(self):
        """Clear conversation history (keeps race context)."""
        self.history = []

    def get_opening_analysis(self) -> str:
        """Ask the model for an unprompted opening summary of the race."""
        return self.ask(
            "Give me a brief opening analysis of this race. Who are the key "
            "contenders, what does the model think will happen, and what are "
            "the main factors to watch?"
        )
