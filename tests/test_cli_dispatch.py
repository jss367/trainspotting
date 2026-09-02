"""Every subcommand reaches its handler.

`main()` canonicalizes `args.target` once for everybody, which is right for the
commands that take a target and fatal for the ones that don't: `find`, `lookup`
and `case-study` are about corpora rather than a registered model, so the
attribute they never parsed raised AttributeError before their handler ran.
`find` shipped that way on main — the canonicalization landed after it did and
nothing since has invoked it.

Argparse wiring is exactly the kind of code that looks obviously correct and is
only ever exercised by running the command, so run every command.
"""

import sys

import pytest

from trainspotting import cli, registry

# Handler attribute per subcommand, and the minimum argv that reaches it. The
# handlers are patched out: this is a test of dispatch, not of what they do.
COMMANDS = [
    ("facts", "cmd_facts", ["facts", "olmo-3-7b-instruct"]),
    ("sources", "cmd_sources", ["sources", "olmo-3-7b-instruct"]),
    ("report", "cmd_report", ["report", "olmo-3-7b-instruct"]),
    ("ask", "cmd_ask", ["ask", "olmo-3-7b-instruct", "is this a question?"]),
    ("find", "cmd_find", ["find", "a phrase"]),
    ("pretrain", "cmd_pretrain", ["pretrain", "olmo-3-7b-think"]),
    ("search", "cmd_search", ["search", "olmo-3-7b-instruct", "a pattern"]),
    ("context", "cmd_context", ["context", "olmo-3-7b-instruct"]),
    ("languages", "cmd_languages", ["languages", "olmo-3-7b-instruct"]),
    ("lookup", "cmd_lookup", ["lookup", "a phrase"]),
    ("case-study", "cmd_case_study", ["case-study"]),
    ("classify", "cmd_classify", ["classify", "olmo-3-7b-instruct"]),
    ("contaminate", "cmd_contaminate", ["contaminate", "olmo-3-7b-instruct", "gsm8k"]),
]


@pytest.mark.parametrize(("name", "handler", "argv"), COMMANDS, ids=[c[0] for c in COMMANDS])
def test_every_command_reaches_its_handler(name, handler, argv, monkeypatch):
    called = []
    monkeypatch.setattr(cli, handler, lambda args: called.append(args))
    monkeypatch.setattr(sys, "argv", ["trainspotting", *argv])

    cli.main()

    assert called, f"`trainspotting {name}` never reached {handler}"


def test_a_target_is_still_canonicalized_before_the_handler_sees_it():
    """The reason the canonicalization exists: `resolve` accepts case variants,
    and writing the raw argument into a result filename produced a file the site
    never asks for."""
    called = []
    import unittest.mock as mock

    with mock.patch.object(cli, "cmd_facts", lambda args: called.append(args.target)), \
         mock.patch.object(sys, "argv", ["trainspotting", "facts", "OLMo-3-7B-Instruct"]):
        cli.main()

    assert called == ["olmo-3-7b-instruct"]


def test_an_unknown_target_still_exits_rather_than_reaching_the_handler():
    called = []
    import unittest.mock as mock

    with mock.patch.object(cli, "cmd_facts", lambda args: called.append(args)), \
         mock.patch.object(sys, "argv", ["trainspotting", "facts", "not-a-model"]), \
         pytest.raises(SystemExit):
        cli.main()

    assert not called


def test_the_targetless_commands_are_the_ones_this_expects():
    """A new command that takes no target has to be added to COMMANDS above, or
    it ships untested through the same hole this file exists to close."""
    targetless = {"find", "lookup", "case-study"}
    assert targetless <= {name for name, _, _ in COMMANDS}
    # Every registered target used above is real, so a registry rename fails
    # here rather than silently skipping the canonicalization path.
    for _, _, argv in COMMANDS:
        if len(argv) > 1 and argv[0] not in targetless:
            assert registry.resolve(argv[1])
