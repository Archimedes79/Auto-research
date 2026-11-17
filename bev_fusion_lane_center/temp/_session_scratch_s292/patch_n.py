import io
p = "evaluation/jtest_eval.py"
s = io.open(p, encoding="utf-8", newline="").read()
def rep(a, b, cnt=1):
    global s
    assert s.count(a) == cnt, (a[:60], s.count(a))
    s = s.replace(a, b)
# N2 + N4: the card must be of the same device, list no undigested module, and have a non-empty environment
rep("def validate_rehearsal_card(card, run_prefix):", "def validate_rehearsal_card(card, run_prefix, device=None):")
rep('''    _require(isinstance(card.get("environment"), dict), "--expect-code-card: the card has no environment record")
    return card''',
    '''    _require(isinstance(card.get("environment"), dict) and bool(card["environment"]), "--expect-code-card: the card has no environment record")
    _require("undigested_modules" in card and not card["undigested_modules"], f"--expect-code-card: that rehearsal loaded repository modules the digest does not cover ({card.get('undigested_modules')}): widen the digest and repeat it")
    _require(device is None or card.get("device") == device, f"--expect-code-card: that rehearsal ran on the device {card.get('device')!r}, the run asks for {device!r}: rehearse the branch that is run")
    return card''')
rep("ref = validate_rehearsal_card(json.load(f), run_prefix)", "ref = validate_rehearsal_card(json.load(f), run_prefix, a.device)")
# N1: no standalone dump on TEST
rep("    require_gate(a)\n    d = dump_dir(a.split, a.max)", '    if a.split == "test":\n        sys.exit("TEST dumps are made by `all` only (the pin, the environment and the control checks are made there)")\n    require_gate(a)\n    d = dump_dir(a.split, a.max)')
# N3: the text of results.md is built before the first card is saved
rep('''    for c in cards:
        save_json(os.path.join(out, f"card_s{c['seed']}.json"), c, overwrite=bool(a.max))
    save_text(os.path.join(out, "results.md"), results_md(cards, a.split, setup), overwrite=bool(a.max))''',
    '''    results_text = results_md(cards, a.split, setup)                                                   # built before the first file is written: a failure here leaves no card without results.md
    for c in cards:
        save_json(os.path.join(out, f"card_s{c['seed']}.json"), c, overwrite=bool(a.max))
    save_text(os.path.join(out, "results.md"), results_text, overwrite=bool(a.max))''')
# N4: stale help texts
rep('help="the code digest of the rehearsal\'s cards (`code_digest`, or its first 16 characters): refused before any dump when the imported lane code differs"',
    'help="optional extra check of the code pin: the `code_digest` of the rehearsal card or its first 16 characters (an empty string refuses); on TEST the digest comes from --expect-code-card"')
rep('help="a card of the rehearsal (its `code_sha256`): names the changed files when --expect-code-digest refuses"',
    'help="the card of a full, passed VAL rehearsal of this system on the same device: REQUIRED on TEST (its digest and its environment are the pin); names the changed files on a refusal"')
io.open(p, "w", encoding="utf-8", newline="").write(s)
print("patched")
