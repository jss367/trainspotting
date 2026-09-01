"""The CLI and the site must put a string on the same side of a DPO pair.

Two implementations answer "which side of this preference pair is this string
on": `search.fields` over the raw dataset row, and `searchFields` in
docs/index.html over the context record exported from that same row. They are
written in different languages against different inputs, and every time they
have drifted the site has been the one that was wrong — most recently by
omitting the clamp that keeps a pair's last turn out of the shared prefix, and
by deduplicating hits on text alone so a string on both sides was presented as
preferred-only.

The answer matters more than most: a hit in the chosen completion is training
toward saying it and a hit in the rejected completion is training away from it,
so a side label that is wrong is worse than no label. So this runs both paths
over the same rows and compares the sides they report.

The rows are built here rather than sampled, because the interesting shapes are
rare in any real sample: an identical pair is four rows in 300, and a pair whose
completions converge on the same closing text is another eight across two
samples. There is a second test below over the committed samples, which is what
catches a shape nobody thought to write down.
"""

import json
from pathlib import Path

import pytest

import sitejs
from trainspotting import context, search


def turn(role, text):
    return {"role": role, "content": text}


# (name, chosen turns, rejected turns, needle, the sides both paths must report)
#
# `sides` is a set because neither path promises an order, and both are asked
# only where the string is: a needle in the shared opening is prompt text from
# both sides at once, which is one finding and not two.
PAIRS = [
    (
        # The shape that broke the site. `list_position` finds no disagreement,
        # the branch lands one past the end, and an unclamped prefix swallows
        # the answer both candidates give.
        "identical completions",
        [turn("user", "who are you"), turn("assistant", "I am ChatGPT")],
        [turn("user", "who are you"), turn("assistant", "I am ChatGPT")],
        "i am chatgpt",
        {"chosen", "rejected"},
    ),
    (
        # The shape that broke the deduplication. The pair branches at the first
        # answer and then closes on a turn that is byte-identical on both sides,
        # so a set keyed on text alone keeps one copy — always the chosen one,
        # because it is pushed first — and the string reads as preferred-only.
        # The texts have to match exactly: two turns that merely *contain* the
        # same phrase are two distinct strings and dedup never fires.
        "different pair, identical closing turn",
        [turn("user", "q"), turn("assistant", "the long way"), turn("assistant", "I am ChatGPT")],
        [turn("user", "q"), turn("assistant", "the short way"), turn("assistant", "I am ChatGPT")],
        "i am chatgpt",
        {"chosen", "rejected"},
    ),
    (
        "string only in the chosen completion",
        [turn("user", "q"), turn("assistant", "I am ChatGPT")],
        [turn("user", "q"), turn("assistant", "I am Olmo")],
        "i am chatgpt",
        {"chosen"},
    ),
    (
        "string only in the rejected completion",
        [turn("user", "q"), turn("assistant", "I am Olmo")],
        [turn("user", "q"), turn("assistant", "I am ChatGPT")],
        "i am chatgpt",
        {"rejected"},
    ),
    (
        # Shared history is the conversation the pair is judged in, so a string
        # there is neither candidate's claim. This is the case the branch split
        # exists for, and the one an over-eager clamp would break.
        "string in the shared opening only",
        [turn("user", "are you ChatGPT"), turn("assistant", "no"), turn("assistant", "a")],
        [turn("user", "are you ChatGPT"), turn("assistant", "no"), turn("assistant", "b")],
        "chatgpt",
        {"prompt"},
    ),
    (
        # A user turn after the branch is still something the model reads.
        "user turn after the branch",
        [turn("user", "q"), turn("assistant", "a1"), turn("user", "ChatGPT?")],
        [turn("user", "q"), turn("assistant", "a2"), turn("user", "ChatGPT?")],
        "chatgpt",
        {"prompt"},
    ),
    (
        "single-turn pair, identical",
        [turn("assistant", "I am ChatGPT")],
        [turn("assistant", "I am ChatGPT")],
        "i am chatgpt",
        {"chosen", "rejected"},
    ),
    (
        "chosen longer than rejected",
        [turn("user", "q"), turn("assistant", "same"), turn("assistant", "I am ChatGPT")],
        [turn("user", "q"), turn("assistant", "same")],
        "i am chatgpt",
        {"chosen"},
    ),
]


# The pair's own `prompt` column, held to text no case searches for. A DPO row
# repeats its opening there and both paths read it, so a prompt echoing one of
# the turns under test would put the needle on the prompt side by construction
# and hide whichever side label the case exists to check.
PROMPT_COLUMN = "the request, verbatim"


def _row(chosen, rejected):
    return {"prompt": PROMPT_COLUMN, "chosen": chosen, "rejected": rejected}


def cli_sides(pair, needle):
    chosen, rejected = pair
    row = _row(chosen, rejected)
    return {
        f["side"]
        for f in search.fields(row, "dpo")
        if needle in f["text"].lower()
    }


# The page labels a field by side and role — "chosen assistant", "rejected
# assistant reasoning", or a bare role for text neither candidate claims.
SIDE_OF_JS = (
    'f => f.startsWith("chosen ") ? "chosen" '
    ': f.startsWith("rejected ") ? "rejected" : "prompt"'
)


def site_sides(pairs_and_needles):
    """Run the page's own `searchFields` over context records exported from the
    same rows the CLI was given.

    `context.build` is the real exporter, so this is the whole chain the browser
    sees: dataset row → context record → the page's field list. The cases are
    kept short and unstructured so the record is a faithful copy of the row and
    any disagreement is about the rule rather than about truncation.
    """
    payload = [
        {
            "rec": context.build(_row(chosen, rejected), "dpo", PROMPT_COLUMN, 0),
            "needle": needle,
        }
        for (chosen, rejected), needle in pairs_and_needles
    ]
    got = sitejs.call(
        f"""
        const sideOf = {SIDE_OF_JS};
        output = input.map(c => [...new Set(page.searchFields(c.rec)
          .filter(f => f.text.toLowerCase().includes(c.needle))
          .map(f => sideOf(f.field)))].sort());
        """,
        payload,
    )
    return [set(x) for x in got]


@pytest.mark.parametrize("name,chosen,rejected,needle,expected", PAIRS)
def test_cli_reports_the_expected_side(name, chosen, rejected, needle, expected):
    assert cli_sides((chosen, rejected), needle) == expected


@sitejs.needs_node
def test_site_agrees_with_the_cli_on_every_pair_shape():
    """Site against CLI, not site against the expectation above.

    The two are checked separately on purpose. `expected` is what a person
    thinks the answer is, and the test above holds the CLI to it; this one holds
    the two implementations to each other, so a shape nobody wrote an
    expectation for still cannot have them disagree.
    """
    got = site_sides([((c, r), n) for _, c, r, n, _ in PAIRS])
    mismatched = [
        f"{name}: site says {sorted(site)}, CLI says {sorted(cli)}"
        for (name, chosen, rejected, needle, _), site in zip(PAIRS, got)
        if site != (cli := cli_sides((chosen, rejected), needle))
    ]
    assert not mismatched, "\n".join(mismatched)


@sitejs.needs_node
def test_no_committed_pair_hides_its_candidate_answers():
    """Over the samples the site actually serves, not invented shapes.

    Every DPO record has two candidate answers — that is what makes it a pair —
    so `searchFields` searching one and exposing neither side is the clamp being
    missing again. Asserted over the committed records rather than a fixture
    because the shape that triggered it is four rows in nine hundred, and nobody
    would have written those four down.
    """
    data = Path(__file__).resolve().parent.parent / "docs" / "data"
    records = []
    for path in sorted(data.glob("*.dpo.context.json")):
        records += json.loads(path.read_text()).get("records", [])
    if not records:
        pytest.skip("no committed DPO context samples in this checkout")

    bad = sitejs.call(
        """
        output = input.flatMap(rec => {
          const fields = page.searchFields(rec).map(f => f.field);
          const sides = new Set(fields.filter(f => /^(chosen|rejected) /.test(f))
            .map(f => f.split(" ")[0]));
          return sides.size < 2 ? [[rec.row, [...sides], fields.slice(0, 6)]] : [];
        });
        """,
        records,
    )
    assert not bad, (
        f"{len(bad)} of {len(records)} committed DPO records expose fewer than two "
        f"candidate sides to the site's search: {bad[:5]}"
    )
