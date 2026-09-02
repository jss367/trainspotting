"""The Bayesian-influence layer, everywhere it can be checked without a checkpoint.

The sampler needs weights and a GPU, so what is pinned here is everything
around it: which committed examples become candidates and which are skipped
with a reason, that the loss is masked to the text the model was fit to and
truncation keeps that text rather than the prompt, that the covariance
arithmetic gives the signs it claims, and that the rendered result names the
checkpoint and says when the chains did not behave. The SGLD loop itself runs
against a toy model when torch is importable, which checks the plumbing (shapes,
restoring w*, finite losses) rather than the physics.
"""

import math

import pytest

from trainspotting import bif, registry

# --- Candidates -----------------------------------------------------------------


def test_a_pretraining_sample_becomes_whole_document_candidates():
    target = registry.resolve("pythia-12b-deduped")
    cands, skipped = bif.candidates("pythia-12b-deduped", target)
    assert skipped == {}
    assert cands, "the committed Pile sample should yield candidates"
    assert {c["side"] for c in cands} == {"document"}
    assert all(c["turns"][0]["role"] == "text" for c in cands)
    assert all(bif.fit_text(c) == bif.candidate_text(c) for c in cands)


def test_post_training_context_yields_a_side_per_fit_text_and_skips_rl_with_a_reason():
    target = registry.resolve("olmo-3-7b-instruct")
    cands, skipped = bif.candidates("olmo-3-7b-instruct", target, stages=["sft", "dpo", "rlvr"])
    sides = {(c["stage"], c["side"]) for c in cands}
    assert ("sft", "response") in sides
    assert ("dpo", "chosen") in sides and ("dpo", "rejected") in sides
    assert "rlvr" in skipped and "no response" in skipped["rlvr"]
    # A DPO pair is two candidates sharing one row, so the two sides can be
    # read against each other.
    dpo = [c for c in cands if c["stage"] == "dpo"]
    assert len({c["row"] for c in dpo}) * 2 == len(dpo)
    # Only assistant text is fit; the prompt is context.
    sft = next(c for c in cands if c["stage"] == "sft")
    assert bif.fit_text(sft)
    assert len(bif.fit_text(sft)) < len(bif.candidate_text(sft))


def test_a_conversation_log_is_not_a_candidate():
    target = registry.resolve("wildchat-1m")
    cands, skipped = bif.candidates("wildchat-1m", target)
    assert cands == []
    assert "chat" in skipped


def test_match_filters_and_an_empty_match_is_a_skip_not_silence():
    target = registry.resolve("pythia-12b-deduped")
    all_cands, _ = bif.candidates("pythia-12b-deduped", target)
    some, skipped = bif.candidates("pythia-12b-deduped", target, match=r"\bthe\b")
    assert 0 < len(some) < len(all_cands)
    assert all(" the " in bif.candidate_text(c).lower() or "the" in bif.candidate_text(c).lower() for c in some)
    none, skipped = bif.candidates("pythia-12b-deduped", target, match="zqxjkvbnm-not-in-any-doc")
    assert none == []
    assert "pretrain" in skipped and "matches" in skipped["pretrain"]


def test_limit_caps_per_stage_deterministically():
    target = registry.resolve("pythia-12b-deduped")
    a, _ = bif.candidates("pythia-12b-deduped", target, limit=7, seed=3)
    b, _ = bif.candidates("pythia-12b-deduped", target, limit=7, seed=3)
    c, _ = bif.candidates("pythia-12b-deduped", target, limit=7, seed=4)
    assert len(a) == 7
    assert [x["row"] for x in a] == [x["row"] for x in b]
    assert [x["row"] for x in a] != [x["row"] for x in c]


def test_a_missing_context_file_is_a_skip_naming_the_command():
    target = registry.resolve("olmo-3-7b-instruct")
    cands, skipped = bif.candidates("no-such-target", target, stages=["sft"])
    assert cands == []
    assert "trainspotting context no-such-target --stage sft" in skipped["sft"]


# --- Encoding -------------------------------------------------------------------


class Tok:
    """A tokenizer that gives every character its own id, so spans are countable."""

    bos_token_id = 1
    bos_token = "<s>"
    chat_template = None

    def encode(self, text, add_special_tokens=False):
        return [ord(ch) for ch in text]


class ChatTok(Tok):
    chat_template = "yes"

    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=False):
        s = "<s>"
        for m in msgs:
            s += f"<{m['role']}>{m['content']}</>"
        if add_generation_prompt:
            s += "<assistant>"
        return s


class BrokenChatTok(ChatTok):
    def apply_chat_template(self, *a, **k):
        raise ValueError("no")


def test_only_fit_spans_are_labels_and_the_first_token_never_is():
    turns = [{"role": "user", "text": "ab"}, {"role": "assistant", "text": "xyz"}]
    e = bif.encode(Tok(), turns, max_tokens=100)
    assert e["ids"][0] == 1  # BOS added, since the text does not start with it
    assert e["labels"][0] == -100
    # "user: " and "ab" and "assistant: " are context; "xyz\n\n" is fit.
    assert e["fit_tokens"] == len("xyz\n\n")
    fit_ids = [i for i, lab in zip(e["ids"], e["labels"]) if lab != -100]
    assert bytes(fit_ids).decode() == "xyz\n\n"
    assert e["truncated"] is False


def test_a_document_is_all_target_but_its_first_token():
    e = bif.encode(Tok(), [{"role": "text", "text": "hello"}], max_tokens=100)
    assert e["tokens"] == 6  # BOS + 5
    assert e["fit_tokens"] == 5


def test_truncation_drops_the_front_and_keeps_the_fit_text():
    turns = [{"role": "user", "text": "p" * 50}, {"role": "assistant", "text": "r" * 10}]
    e = bif.encode(Tok(), turns, max_tokens=20)
    assert e["truncated"] is True
    assert e["tokens"] == 20
    assert e["fit_tokens"] == len("r" * 10 + "\n\n")
    assert e["ids"][-1] == ord("\n")


def test_a_chat_template_renders_the_turns_and_fits_the_assistant_span_only():
    turns = [{"role": "user", "text": "hi"}, {"role": "assistant", "text": "yo"}]
    spans = bif.pieces(ChatTok(), turns)
    assert spans == [("<s><user>hi</><assistant>", False), ("yo</>", True)]
    e = bif.encode(ChatTok(), turns, max_tokens=100)
    # The template already put the BOS text first, so none is added.
    assert e["ids"][0] == ord("<")
    assert bytes(i for i, lab in zip(e["ids"], e["labels"]) if lab != -100).decode() == "yo</>"


def test_a_template_that_cannot_render_falls_back_to_roles():
    turns = [{"role": "user", "text": "hi"}, {"role": "assistant", "text": "yo"}]
    assert bif.pieces(BrokenChatTok(), turns) == bif.pieces(Tok(), turns)


def test_a_document_never_goes_through_a_chat_template():
    assert bif.pieces(ChatTok(), [{"role": "text", "text": "doc"}]) == [("doc", True)]


def test_the_query_is_fit_text_behind_an_optional_prompt():
    assert bif.query_candidate("said")["turns"] == [{"role": "text", "text": "said"}]
    q = bif.query_candidate("said", prompt="asked")
    assert [t["role"] for t in q["turns"]] == ["user", "assistant"]
    assert bif.fit_text(q) == "said"


# --- Statistics -----------------------------------------------------------------


def chain(query, *cands):
    """One chain's draws from per-series lists: draw d is [query[d], cand_0[d], ...]."""
    return [[q, *[c[d] for c in cands]] for d, q in enumerate(query)]


def test_covariance_has_the_sign_of_the_relationship_and_correlation_is_normalized():
    q = [1.0, 2.0, 3.0, 4.0]
    same = [1.0, 2.0, 3.0, 4.0]
    twice = [2.0, 4.0, 6.0, 8.0]
    opposite = [4.0, 3.0, 2.0, 1.0]
    flat = [5.0, 5.0, 5.0, 5.0]
    stats = bif.influence([chain(q, same, twice, opposite, flat)])
    assert stats[0]["cov"] == pytest.approx(1.25)
    assert stats[0]["corr"] == pytest.approx(1.0)
    assert stats[1]["cov"] == pytest.approx(2.5)
    assert stats[1]["corr"] == pytest.approx(1.0)
    assert stats[2]["cov"] == pytest.approx(-1.25)
    assert stats[2]["corr"] == pytest.approx(-1.0)
    assert stats[3]["cov"] == 0.0 and stats[3]["corr"] == 0.0
    # One chain has no spread to report.
    assert all(s["cov_stderr"] is None for s in stats)


def test_the_estimate_averages_within_chain_covariances_and_reports_their_spread():
    a = chain([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])          # cov 2/3
    b = chain([10.0, 20.0, 30.0], [3.0, 2.0, 1.0])       # cov -20/3
    stats = bif.influence([a, b])
    assert stats[0]["cov"] == pytest.approx((2 / 3 - 20 / 3) / 2)
    assert stats[0]["chain_covs"] == pytest.approx([2 / 3, -20 / 3])
    assert stats[0]["cov_stderr"] is not None and stats[0]["cov_stderr"] > 0
    # Pooling the chains would have found a large positive covariance from
    # chain b's higher query losses alone; within-chain does not.
    assert stats[0]["cov"] < 0


def test_partial_covariance_removes_what_moves_with_the_sample_average():
    # The query and every candidate ride one common excursion `m`; candidate 0
    # is nothing but that excursion, candidate 1 adds a component the query
    # shares, candidate 2 adds one the query opposes.
    m = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    extra = [1.0, -1.0, 1.0, -1.0, 1.0, -1.0]
    q = [mm + e for mm, e in zip(m, extra)]
    c0 = [2 * mm for mm in m]
    c1 = [mm + 0.5 * e for mm, e in zip(m, extra)]
    c2 = [mm - 0.5 * e for mm, e in zip(m, extra)]
    stats = bif.influence([chain(q, c0, c1, c2)])
    # Raw covariance calls all three positive and the pure common mode largest.
    assert stats[0]["cov"] > stats[1]["cov"] > 0 and stats[2]["cov"] > 0
    # Partial covariance sees through it.
    assert stats[0]["partial"] == pytest.approx(0.0, abs=1e-9)
    assert stats[1]["partial"] > 0
    assert stats[2]["partial"] < 0
    assert stats[1]["partial"] == pytest.approx(-stats[2]["partial"])


def test_llc_is_positive_when_the_posterior_sits_above_the_loss_at_origin():
    run = {
        "at_origin": [0.5, 1.0, 2.0],
        "losses": [chain([0.5, 0.5], [1.5, 1.5], [2.5, 2.5])],
        "nbeta": 4.0,
    }
    lc = bif.llc(run)
    assert lc["loss_at_origin"] == 1.5
    assert lc["per_chain"] == pytest.approx([4.0 * 0.5])
    below = bif.llc({**run, "losses": [chain([0.5], [0.5], [1.5])]})
    assert below["mean"] < 0


def test_drift_is_the_rise_across_the_retained_steps_only():
    run = {
        "at_origin": [0.5, 1.0],
        "losses": [chain([0.5], [1.5])],
        "nbeta": 1.0,
        # Two burn-in steps that fell, then eight retained steps that rose.
        "trajectory": [[9.0, 8.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0]],
        "burn_in": 2,
    }
    assert bif.llc(run)["drift"] == pytest.approx([4.0 - 1.0])


def test_baseline_is_the_covariance_with_the_average_candidate():
    q = [1.0, 2.0, 3.0]
    losses = [chain(q, [1.0, 2.0, 3.0], [3.0, 2.0, 1.0])]  # average candidate is flat
    assert bif.baseline(losses) == pytest.approx(0.0)
    losses = [chain(q, [1.0, 2.0, 3.0], [1.0, 2.0, 3.0])]
    assert bif.baseline(losses) == pytest.approx(2 / 3)


def test_summarize_groups_and_ranks_by_mean_covariance():
    cands = [
        {"stage": "sft", "side": "response", "source": "a", "id": "1", "row": 1},
        {"stage": "sft", "side": "response", "source": "b", "id": "2", "row": 2},
        {"stage": "dpo", "side": "rejected", "source": "a", "id": "3", "row": 3},
    ]
    stats = [{"cov": 0.1, "corr": 0.5, "partial": 0.2}, {"cov": -0.3, "corr": -0.2, "partial": -0.4},
             {"cov": 0.4, "corr": 0.9, "partial": 0.3}]
    groups = bif.summarize(cands, stats, lambda c: (("stage", c["stage"]), ("side", c["side"])))
    assert [g["stage"] for g in groups] == ["dpo", "sft"]
    sft = groups[1]
    assert sft["n"] == 2 and sft["toward"] == 1
    assert sft["mean_cov"] == pytest.approx(-0.1)
    assert sft["mean_partial"] == pytest.approx(-0.1)
    assert sft["best"]["id"] == "1"


# --- Result and rendering -------------------------------------------------------


def fake_result(llc_sign=1.0):
    cands = [
        {"stage": "pretrain", "kind": "pretrain", "side": "document", "source": "Pile-CC",
         "id": "row-1", "row": 1, "cut": False, "turns": [{"role": "text", "text": "As an AI language model I"}]},
        {"stage": "pretrain", "kind": "pretrain", "side": "document", "source": "GitHub",
         "id": "row-2", "row": 2, "cut": True, "turns": [{"role": "text", "text": "def f(): pass"}]},
    ]
    encoded = [
        {"tokens": 10, "fit_tokens": 9, "truncated": False},
        {"tokens": 512, "fit_tokens": 511, "truncated": True},
    ]
    q = [1.0, 2.0, 3.0]
    run = {
        # Candidates sit at 0.5 at w* and average 2.0 under the posterior, so
        # the chains did rise off the origin, as a sampler that behaved would.
        "at_origin": [1.0, 0.5, 0.5],
        "losses": [chain(q, [1.0, 2.0, 3.0], [3.0, 2.0, 1.0]),
                   chain(q, [2.0, 3.0, 4.0], [1.0, 1.0, 1.0])],
        "trajectory": [[2.0, 2.1, 2.1, 2.1], [2.0, 2.2, 2.2, 2.2]],
        "nbeta": 3.0,
        "burn_in": 1,
    }
    if llc_sign < 0:
        run["at_origin"] = [1.0, 20.0, 30.0]
    if llc_sign == 0:
        run["trajectory"] = [[2.0, 2.0, 2.0, 9.0], [2.0, 2.2, 2.2, 2.2]]
    settings = {"chains": 2, "draws": 3, "burn_in": 1, "lr": 1e-5, "nbeta": 3.0, "gamma": 100.0,
                "batch": 8, "max_tokens": 512}
    return bif.result("pythia-12b-deduped", "EleutherAI/pythia-70m-deduped", "abc123def456789",
                      "As an AI language model", None, cands, encoded, run,
                      {"sft": "no committed context records"}, settings)


def test_result_ranks_records_and_carries_every_provenance_field():
    res = fake_result()
    assert res["model"] == "EleutherAI/pythia-70m-deduped"
    assert res["candidates"] == 2
    ranked = sorted(res["records"], key=lambda r: r["rank"])
    assert [r["id"] for r in ranked] == ["row-1", "row-2"]
    assert ranked[0]["cov"] > ranked[1]["cov"]
    assert ranked[0]["snippet"].startswith("As an AI")
    assert ranked[1]["truncated"] and ranked[1]["cut"]
    assert all(r["above_baseline"] == pytest.approx(r["cov"] - res["baseline_cov"]) for r in ranked)
    assert all("partial" in r for r in ranked)
    # The raw draws travel with the file: chains × draws × (query + records).
    assert len(res["draws"]) == 2 and all(len(c) == 3 for c in res["draws"])
    assert all(len(d) == 3 for c in res["draws"] for d in c)
    assert res["skipped"] == {"sft": "no committed context records"}
    assert res["query_loss"]["at_origin"] == 1.0
    assert res["llc"]["mean"] > 0
    assert res["stages"][0]["n"] == 2
    assert {g["source"] for g in res["sources"]} == {"Pile-CC", "GitHub"}


def test_render_names_the_checkpoint_the_skip_and_the_ends_of_the_ranking():
    text = "\n".join(bif.render(fake_result()))
    assert "`EleutherAI/pythia-70m-deduped` at `abc123def456`" in text
    assert "not weighed: no committed context records" in text
    assert "the chains sat near w*" in text
    assert "row-1" in text and "row-2" in text
    assert "[truncated] [cut]" in text
    assert "not on the training set" in text


def test_render_says_so_when_a_chain_sat_below_the_origin_loss():
    text = "\n".join(bif.render(fake_result(llc_sign=-1.0)))
    assert "*below* the loss at w*" in text


def test_render_says_so_when_a_chain_was_still_climbing():
    text = "\n".join(bif.render(fake_result(llc_sign=0)))
    assert "1 of 2 chains were still climbing" in text
    assert "lower --lr" in text
    assert "covaries" in text and "average candidate" in text


def test_default_nbeta_is_batch_over_log_batch():
    assert bif.default_nbeta(8) == pytest.approx(8 / math.log(8))
    assert bif.default_nbeta(1) == 1.0


# --- The sampler, on a toy ------------------------------------------------------

torch = pytest.importorskip("torch")


class Toy(torch.nn.Module):
    """The smallest thing with the causal-LM interface `sample` reads."""

    def __init__(self, vocab=16, dim=8):
        super().__init__()
        self.emb = torch.nn.Embedding(vocab, dim)
        self.out = torch.nn.Linear(dim, vocab)

    def forward(self, input_ids, attention_mask=None):
        class Out:
            pass

        o = Out()
        o.logits = self.out(self.emb(input_ids))
        return o


def enc(ids, fit_from):
    labels = [-100] * fit_from + ids[fit_from:]
    labels[0] = -100
    return {"ids": ids, "labels": labels, "tokens": len(ids), "fit_tokens": sum(1 for x in labels if x != -100),
            "truncated": False}


def test_sample_records_every_loss_and_puts_the_weights_back():
    torch.manual_seed(0)
    model = Toy()
    before = [p.detach().clone() for p in model.parameters()]
    cands = [enc([1, 2, 3, 4, 5], 2), enc([6, 7, 8], 1), enc([9, 10, 11, 12], 0)]
    query = enc([3, 4, 5], 0)
    run = bif.sample(
        model, cands, query, device="cpu", chains=2, draws=4, burn_in=2, every=2,
        lr=1e-3, nbeta=2.0, gamma=10.0, batch=2, eval_batch=2, seed=1,
    )
    assert len(run["at_origin"]) == 4
    assert len(run["losses"]) == 2 and all(len(c) == 4 for c in run["losses"])
    assert all(len(d) == 4 for c in run["losses"] for d in c)
    assert all(len(t) == 2 + 4 * 2 for t in run["trajectory"])
    assert all(math.isfinite(x) for c in run["losses"] for d in c for x in d)
    for p, w in zip(model.parameters(), before):
        assert torch.equal(p.detach(), w)
    stats = bif.influence(run["losses"])
    assert len(stats) == 3
    assert all(s["cov_stderr"] is not None for s in stats)


def test_batch_losses_are_per_example_means_over_fit_tokens_only():
    torch.manual_seed(0)
    model = Toy()
    short = enc([1, 2, 3], 0)
    # The same tokens with the first two masked: a different mean, over one token.
    masked = enc([1, 2, 3], 2)
    padded = bif.batch_losses(model, [short, masked, enc([4, 5, 6, 7, 8, 9], 0)], "cpu", batch=3)
    alone = bif.batch_losses(model, [short], "cpu", batch=1)
    assert padded[0] == pytest.approx(alone[0], rel=1e-5)  # padding does not leak into the loss
    assert padded[1] != pytest.approx(padded[0])
