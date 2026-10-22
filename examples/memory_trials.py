"""Repeat the long-term-memory scenarios against a real model and count outcomes.

    python examples/memory_trials.py 6      # number of independent trials (default 6)

Needs a configured KESTREL_API_KEY. Each trial uses a throw-away home directory, so your own data is untouched.
Models are non-deterministic: one run proves little, so look at the counts over several trials.
"""
import asyncio, sys, tempfile
from dataclasses import replace
from pathlib import Path
from kestrel.app import App
from kestrel.config import Settings

N = int(sys.argv[1]) if len(sys.argv) > 1 else 6
S1 = "From now on, please answer me in bullet points."
S2 = "For context: I'm Priya, I live in Chicago, and I'm allergic to peanuts."
S3 = "Update: I just moved to Boston."
S4 = "In one sentence: which city do I live in, and which snack ingredient must you never suggest to me?"

async def trial(base: Settings):
    with tempfile.TemporaryDirectory() as tmp:
        app = await App.create(replace(base, home=Path(tmp), workspace=None, approval="auto"), use_mcp=False)
        app.agent.episodic = None                      # isolate long-term memory
        out = {}
        try:
            r1 = await app.agent.run(app.store.create_session(), S1)
            facts = " ".join(app.longterm.facts()).lower()
            claimed = any(w in r1.final_text.lower() for w in ("noted", "remember", "saved", "recorded", "记下", "记住", "will do", "i'll"))
            out["s1_stored"] = "bullet" in facts
            out["s1_false_claim"] = claimed and "remember" not in r1.tool_calls and not out["s1_stored"]
            await app.agent.run(app.store.create_session(), S2)
            facts = " ".join(app.longterm.facts()).lower()
            out["s2_all_three"] = all(k in facts for k in ("priya", "chicago", "peanut"))
            await app.agent.run(app.store.create_session(), S3)
            facts = app.longterm.facts(); text = " ".join(facts).lower()
            stale = any("chicago" in f.lower() and not any(w in f.lower() for w in ("from", "moved", "previous", "formerly", "used to")) for f in facts)
            out["s3_boston"] = "boston" in text
            out["s3_allergy_kept"] = "peanut" in text
            out["s3_no_stale_city"] = not stale
            r4 = await app.agent.run(app.store.create_session(), S4)
            a = r4.final_text.lower()
            out["s4_answers"] = "boston" in a and "peanut" in a
            out["_facts"] = app.longterm.facts()
        finally:
            await app.aclose()
        return out

async def main():
    base = Settings.from_env()
    results = await asyncio.gather(*(trial(base) for _ in range(N)))
    keys = [k for k in results[0] if not k.startswith("_")]
    print(f"{N} independent trials, real model: {base.model}\n")
    for k in keys:
        print(f"  {k:20} {sum(bool(r[k]) for r in results)}/{N}")
    print("\nfinal memory per trial:")
    for i, r in enumerate(results, 1):
        print(f"  trial {i}: {r['_facts']}")
asyncio.run(main())
