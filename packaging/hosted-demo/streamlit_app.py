"""Streamlit Community Cloud entry point for the hosted demo.

DO NOT DEPLOY until the 5-layer safety checklist in ../docs/release.md
is satisfied:
  1. Secret management
  2. Cost guardrails
  3. Abuse protection
  4. PHI / sensitive-input protection (banner, dummy examples, no input logging)
  5. Repo hygiene

This file is a *scaffold*. Wire in your dedicated API key via
st.secrets["OPENAI_API_KEY"], cap usage per session, and never log raw
input text.
"""
from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

# ── 1. Secrets ───────────────────────────────────────────────────────────────
# Set via Streamlit Cloud dashboard → Secrets. Never commit a real key.
OPENAI_KEY = st.secrets.get("OPENAI_API_KEY")
DEMO_PASSCODE = st.secrets.get("DEMO_PASSCODE")
MODEL_NAME = st.secrets.get("DEMO_MODEL", "gpt-4o-mini")

# ── 2. Per-session caps ──────────────────────────────────────────────────────
MAX_CALLS_PER_SESSION = 20
MAX_INPUT_CHARS = 5000
MAX_OUTPUT_TOKENS = 512

if "calls_today" not in st.session_state:
    st.session_state.calls_today = 0


# ── 3. Auth gate ─────────────────────────────────────────────────────────────
def auth_gate() -> bool:
    """Soft passcode gate. For reviewer-only access, prefer Streamlit Cloud's
    built-in private-app + email allowlist instead of this code path."""
    if DEMO_PASSCODE is None:
        return True  # no passcode configured — open demo (caps still apply)
    entered = st.text_input("Enter demo passcode", type="password")
    if not entered:
        st.stop()
    if entered != DEMO_PASSCODE:
        st.error("Wrong passcode.")
        st.stop()
    return True


# ── 4. PHI banner + dummy examples ───────────────────────────────────────────
PHI_BANNER = """
**⚠️ DO NOT PASTE PROTECTED HEALTH INFORMATION.**
This is a public demo. Use only synthetic or de-identified text. Inputs are
truncated to 5000 characters. Numbers (latency, token counts) are logged for
operational monitoring; **the raw input text is NOT logged**.
"""

DUMMY_EXAMPLES_DIR = Path(__file__).resolve().parents[2] / "examples" / "dummy" / "data"

def load_dummy_examples() -> dict[str, str]:
    """Load 2–3 small synthetic examples from the dummy/ skeleton."""
    out: dict[str, str] = {}
    if not DUMMY_EXAMPLES_DIR.exists():
        return out
    for p in sorted(DUMMY_EXAMPLES_DIR.rglob("*.txt"))[:3]:
        out[p.stem] = p.read_text()[:MAX_INPUT_CHARS]
    return out


# ── 5. Demo ───────────────────────────────────────────────────────────────────
def main() -> None:
    st.set_page_config(page_title="Digital Registrar — demo", page_icon="🏥", layout="wide")
    st.title("Digital Registrar — public demo")
    st.warning(PHI_BANNER)
    auth_gate()

    with st.sidebar:
        st.markdown("### Demo info")
        st.markdown(f"- Model: `{MODEL_NAME}`")
        st.markdown(f"- Per-session cap: **{MAX_CALLS_PER_SESSION} calls**")
        st.markdown(f"- Per-call max tokens: **{MAX_OUTPUT_TOKENS}**")
        st.markdown(
            "- Source: [github.com/kblab2024/digitalregistrar]"
            "(https://github.com/kblab2024/digitalregistrar)"
        )
        st.caption("This demo runs against synthetic data only.")

    if st.session_state.calls_today >= MAX_CALLS_PER_SESSION:
        st.error(f"Demo limit reached ({MAX_CALLS_PER_SESSION} calls this session). Refresh to start over.")
        st.stop()

    examples = load_dummy_examples()
    if examples:
        choice = st.selectbox("Pick a synthetic example (or paste below)", ["-- none --", *examples.keys()])
        default_text = examples.get(choice, "") if choice != "-- none --" else ""
    else:
        default_text = ""

    text = st.text_area("Paste a synthetic pathology report (max 5000 chars):",
                        value=default_text, height=300, max_chars=MAX_INPUT_CHARS)

    if st.button("Extract", type="primary"):
        if not text.strip():
            st.error("Paste a report first.")
            return
        if OPENAI_KEY is None:
            st.error("Demo not configured — admin must set OPENAI_API_KEY in Streamlit secrets.")
            return
        st.session_state.calls_today += 1
        # TODO: call digital_registrar.run_pipeline with a dummy model context.
        # NEVER log `text` — log only metadata (len, model, latency, tokens).
        st.info("Demo scaffold — wire `digital_registrar.run_pipeline(text, ...)` here, "
                "with `max_tokens=512` and `@st.cache_data(ttl=3600)`.")
        st.json({"demo": True, "input_len": len(text), "model": MODEL_NAME})


if __name__ == "__main__":
    main()
