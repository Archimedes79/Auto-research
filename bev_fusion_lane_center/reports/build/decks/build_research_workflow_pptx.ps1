# Pitch deck for team and managers: why we enable automated research loops. PowerPoint COM, no downloads. ASCII only.
# Usage: powershell -File build_research_workflow_pptx.ps1 [out.pptx]
# Inputs: helpers in ppt_lib.ps1, chart reports\source\weekly\deck\s_noise.png (make_systematics_charts.py).
# Numbers: HANDOVER.md (17 loops, ~200 experiments, chain 31 ms vs 628 ms); external figures: slide 4 and notes.
param([string]$Out = ([System.IO.Path]::GetFullPath("$PSScriptRoot\..\..\weekly\Automated_Research_Workflow_short_EN.pptx")))
. "$PSScriptRoot\ppt_lib.ps1"
$app = New-Object -ComObject PowerPoint.Application
$pres = $app.Presentations.Add(0)
$pres.PageSetup.SlideWidth = 960
$pres.PageSetup.SlideHeight = 540
$TOTAL = 12
$FOOT = "Automated research loops"
$GREENBG = C "E6F4EA"; $REDBG = C "FBECEB"; $AMBERBG = C "FDF3DC"; $BLUEBG = C "E8EEFB"; $ORG = C "F2A65A"

# ---- local helpers (look only) ----
function Add-Tag($s, [string]$text) {
    $bar = $s.Shapes.AddShape(1, 0, 0, 10, 540); $bar.Fill.Solid(); $bar.Fill.ForeColor.RGB = $BLUE; $bar.Line.Visible = 0
    $null = Add-Text $s $text.ToUpper() 40 10 600 14 11 $ORANGE $true
}
function Add-Num($s, [double]$x, [double]$y, [double]$d, [string]$n, [int]$fill, [int]$fc = -1) {
    if ($fc -lt 0) { $fc = $WHITE }
    $o = $s.Shapes.AddShape(9, $x, $y, $d, $d); $o.Fill.Solid(); $o.Fill.ForeColor.RGB = $fill; $o.Line.Visible = 0
    $tr = $o.TextFrame.TextRange; $tr.Text = $n; $tr.Font.Name = $HFONT; $tr.Font.Size = [Math]::Round($d * 0.5); $tr.Font.Bold = -1; $tr.Font.Color.RGB = $fc
    $tr.ParagraphFormat.Alignment = 2; $o.TextFrame.VerticalAnchor = 3
    $o.TextFrame.MarginLeft = 0; $o.TextFrame.MarginRight = 0; $o.TextFrame.MarginTop = 0; $o.TextFrame.MarginBottom = 0
}
function Add-Card2($s, [double]$x, [double]$y, [double]$w, [double]$h, [int]$strip, [int]$fill = -1) {
    if ($fill -lt 0) { $fill = $PANEL }
    $null = Add-Card $s $x $y $w $h $fill
    $t = $s.Shapes.AddShape(1, $x + 12, $y, $w - 24, 5); $t.Fill.Solid(); $t.Fill.ForeColor.RGB = $strip; $t.Line.Visible = 0
}

try {
    # 1 title with loop ring
    $s = New-Slide $true
    $null = Add-Text $s "PROPOSAL" 60 110 500 20 14 $ORG $true
    $null = Add-Text $s "Research that keeps working while we sleep" 60 140 500 150 40 $WHITE $true $HFONT
    $null = Add-Text $s "Automated experiment loops: what they are, what they gave us in two weeks, and what it takes to do it properly" 60 318 480 90 18 $SOFT
    $lab = @("propose", "run", "measure", "keep / drop", "log", "review")
    for ($i = 0; $i -lt 6; $i++) {
        $a = ($i * 60 - 90) * [Math]::PI / 180
        $cx = 730 + 135 * [Math]::Cos($a); $cy = 275 + 135 * [Math]::Sin($a)
        $fillc = C "22366B"; if ($i -eq 5) { $fillc = $ORG }
        $o = $s.Shapes.AddShape(9, $cx - 40, $cy - 40, 80, 80); $o.Fill.Solid(); $o.Fill.ForeColor.RGB = $fillc; $o.Line.ForeColor.RGB = (C "5F7DC4"); $o.Line.Weight = 1.5
        $tr = $o.TextFrame.TextRange; $tr.Text = $lab[$i]; $tr.Font.Name = $FONT; $tr.Font.Size = 12; $tr.Font.Bold = -1; $tr.ParagraphFormat.Alignment = 2
        if ($i -eq 5) { $tr.Font.Color.RGB = $NAVY } else { $tr.Font.Color.RGB = $WHITE }
        $o.TextFrame.MarginLeft = 2; $o.TextFrame.MarginRight = 2; $o.TextFrame.VerticalAnchor = 3
    }
    $null = Add-Text $s "the loop" 690 262 80 24 16 (C "B9C2D3") $true $HFONT 2
    Add-Notes $s "Purpose: explain the idea to the team and to managers, and ask for the decisions on the last slide. The ring is the loop; the orange step is the human review."

    # 2 idea
    $s = New-Slide
    Add-Tag $s "The idea"
    $null = Add-Title $s "People set the goal and the rules, a loop does the repeating work"
    $st = @(@("You", "set the question, the rules and the stop time"), @("Loop proposes", "an AI agent suggests the next experiment"), @("Loop runs it", "on our GPU, with a fixed time budget"), @("Loop measures", "with scoring code it cannot change"), @("Loop decides", "keeps what improves, drops the rest, writes it down"), @("You read", "the log in the morning and choose what is next"))
    for ($i = 0; $i -lt 6; $i++) {
        $x = 40 + $i * 148
        $strip = $BLUE; if ($i -eq 0 -or $i -eq 5) { $strip = $ORANGE }
        $null = Add-Card2 $s $x 120 136 150 $strip
        $null = Add-Num $s ($x + 10) 136 30 ([string]($i + 1)) $strip
        $null = Add-Text $s $st[$i][0] ($x + 10) 176 118 22 14 $NAVY $true $HFONT
        $null = Add-Text $s $st[$i][1] ($x + 10) 202 118 64 12 $NAVY
        if ($i -lt 5) { $null = Add-Line $s ($x + 137) 195 ($x + 147) 195 $BLUE 2 $true }
    }
    $null = Add-Line $s 560 272 560 300 $BLUE 2 $false
    $null = Add-Line $s 560 300 330 300 $BLUE 2 $false
    $null = Add-Line $s 330 300 330 272 $BLUE 2 $true
    $null = Add-Text $s "repeats many times, also overnight" 340 304 300 20 12 $GREY
    $null = Add-Callout $s "Same pattern as Karpathy's autoresearch: a fixed time per try, one score, keep what improves, undo what does not." 40 346 880 54 $BLUEBG $NAVY 15
    $null = Add-Text $s "The machine does the repeating. People decide what is worth doing and what the results mean." 40 420 880 30 17 $NAVY $true
    Add-Footer $s
    Add-Notes $s "In this project: PROGRAM.md holds the goal and rules; an agent wakes every 20 - 25 minutes, reads the state files, checks jobs, parses results and decides; run.py starts jobs and logs them; evaluation/ is the fixed judge; LEDGER.csv and state files record everything."

    # 3 what we got
    $s = New-Slide
    Add-Tag $s "What it gave us"
    $null = Add-Title $s "Two weeks, one person, one graphics card"
    $stats = @(@("17 loops", "of planned experiments in 14 days, many overnight", $BLUE), @("~200", "experiments catalogued, failures included, each with a result", $BLUE), @("20x faster", "lane-graph chain: 31 ms instead of 628 ms per frame", $GREEN))
    for ($i = 0; $i -lt 3; $i++) {
        $x = 40 + $i * 300
        $null = Add-Card2 $s $x 100 280 130 $stats[$i][2]
        $null = Add-Text $s $stats[$i][0] ($x + 16) 116 250 52 36 $stats[$i][2] $true $HFONT
        $null = Add-Text $s $stats[$i][1] ($x + 16) 172 250 52 13 $GREY
    }
    $null = Add-Box $s 40 250 430 140 "Before" "One experiment at a time, started by hand. Results live in a chat. Numbers are checked by whoever wrote them." $PANEL $GREY 15
    $null = Add-Box $s 490 250 430 140 "Now" "A loop plans, runs and judges for hours alone. Everything lands in a ledger and state files, so any colleague can pick up the work." $BLUEBG $BLUE 15
    $null = Add-Callout $s "Not magic: the loop also produced wrong numbers early on. Reviews and fixed rules caught them (slide 9)." 40 410 880 50 $AMBERBG $NAVY 15
    Add-Footer $s
    Add-Notes $s "Sources: HANDOVER.md and the weekly reports. Chain time 31 ms per frame against 628 ms for the control (loop 16b, CUDA graphs and vectorised assembly). 'Experiments catalogued' counts catalogue ids E15 - E212. The speed gain comes from engineering experiments found with the loop, not from the loop alone."

    # 4 others
    $s = New-Slide
    Add-Tag $s "Evidence"
    $null = Add-Title $s "Others do it already, with real results"
    $oth = @(
        @("Karpathy", "autoresearch", "An AI agent edits one training file, trains 5 minutes, keeps the change only if the score improves.", "About 100 experiments per night on one GPU"),
        @("Google DeepMind", "AlphaEvolve", "An AI proposes code, an automatic checker scores it, the best are improved again.", "Scheduling rule saving about 0.7 % of Google compute; a faster matrix algorithm"),
        @("AIDE", "Kaggle benchmark", "An agent keeps a tree of scripts and improves the good ones.", "Medal level in 16.9 % of 75 competitions, no human")
    )
    for ($i = 0; $i -lt 3; $i++) {
        $x = 40 + $i * 300
        $null = Add-Card2 $s $x 100 280 250 $BLUE
        $null = Add-Text $s $oth[$i][0] ($x + 16) 116 250 26 18 $BLUE $true $HFONT
        $null = Add-Text $s $oth[$i][1] ($x + 16) 142 250 18 12 $GREY
        $null = Add-Text $s $oth[$i][2] ($x + 16) 172 250 90 13 $NAVY
        $null = Add-Text $s "Result" ($x + 16) 270 250 16 11 $GREEN $true
        $null = Add-Text $s $oth[$i][3] ($x + 16) 288 250 56 14 $GREEN $true
    }
    $null = Add-Callout $s "The same field warns: one AI-written paper passed a workshop review but contained invented results. Fast loops need checked numbers." 40 368 880 54 $AMBERBG $NAVY 15
    $null = Add-Text $s "Sources: github.com/karpathy/autoresearch; DeepMind blog 2025; arxiv.org/abs/2410.07095; arxiv.org/abs/2504.08066. Figures via abstracts and press summaries read 24 Oct 2026; check before quoting." 40 436 880 40 11 $GREY
    Add-Footer $s
    Add-Notes $s "Karpathy README: fixed 5 minute budget, about 12 trials per hour. AlphaEvolve: 0.7 % fleet compute from a Borg scheduling heuristic and a 4x4 complex matrix product with 48 multiplications instead of 49 (DeepMind blog, via summaries). AIDE with o1-preview on MLE-bench: 16.9 % bronze or better. AI Scientist v2: one of three papers passed an ICLR workshop review; outside checks later found hallucinated results (commentary summaries). Also relevant: Hyperband (arXiv 1603.06560), over 10 times faster tuning by stopping weak trials early; Henderson et al. 2018 (arXiv 1709.06560), scores vary with the random seed."

    # 5 how to
    $s = New-Slide
    Add-Tag $s "How to do it"
    $null = Add-Title $s "Six steps, the same pattern as autoresearch"
    $hw = @(
        @("Write the instructions", "Goal, what may change, what must not, when to stop.", "PROGRAM.md"),
        @("Fix the judge", "Scoring code the agent cannot edit, tested on a number you already know.", "evaluation/ + self-tests"),
        @("Change one thing per try", "One file or design choice per trial, so a result has one cause.", "one folder per experiment"),
        @("Fix the budget", "Same time for every try. Many cheap tries first, the best ones longer.", "rungs of 1, 2, 4, 8 units"),
        @("Write the keep rule, log it", "Keep only if better than the noise. Log every try, also failures.", "decision rule + ledger"),
        @("Read in the morning", "You read the log, spot-check numbers, choose the next question.", "HANDOVER.md, reviewer")
    )
    for ($i = 0; $i -lt 6; $i++) {
        $x = 40 + ($i % 3) * 300; $y = 100 + [Math]::Floor($i / 3) * 190
        $null = Add-Card2 $s $x $y 280 176 $BLUE
        $null = Add-Num $s ($x + 14) ($y + 16) 36 ([string]($i + 1)) $BLUE
        $null = Add-Text $s $hw[$i][0] ($x + 60) ($y + 18) 210 40 15 $NAVY $true $HFONT
        $null = Add-Text $s $hw[$i][1] ($x + 16) ($y + 68) 250 60 13 $NAVY
        $null = Add-Text $s ("Here: " + $hw[$i][2]) ($x + 16) ($y + 140) 250 24 12 $GREEN $true
    }
    $null = Add-Text $s "Start small: three tries by hand for one hour, then one night, then more." 40 484 880 22 14 $BLUE $true
    Add-Footer $s
    Add-Notes $s "The six steps are the autoresearch pattern (human-written program.md, read-only prepare.py as judge, one editable train.py, fixed budget, keep or discard, log) made explicit and mapped to our files."

    # pipeline: small vs large
    $s = New-Slide
    Add-Tag $s "Experiment quickly"
    $null = Add-Title $s "Start small, grow only what survives: the research pipeline"
    $pp = @(@("Idea + gate", "cheap checks before any training"), @("Small proxy", "250 frames, minutes"), @("Ladder", "double the budget, drop the weaker half"), @("Full check", "all 873 validation frames, 2 seeds"), @("Test once", "833 frames, 21 scenes, logged"), @("Report", "numbers from files, reviewed"))
    for ($i = 0; $i -lt 6; $i++) {
        $x = 40 + $i * 148
        $fc = $BLUEBG; $tc = $NAVY
        if ($i -eq 4) { $fc = (C "FCE8CF") }
        $null = Add-Node $s $x 96 136 74 ($pp[$i][0] + "`r" + $pp[$i][1]) $fc $tc 11 $true (C "9DB8E8")
        if ($i -lt 5) { $null = Add-Line $s ($x + 137) 133 ($x + 147) 133 $BLUE 2 $true }
    }
    $null = Add-Text $s "cheap and many" 40 176 300 16 11 $GREEN $true
    $null = Add-Text $s "expensive and few" 700 176 220 16 11 $RED $true $FONT 3
    $rows = @(
        @("", "Small data, short run", "Large data, long run"),
        @("What", "VAL250 (250 frames), 1 - 2 units (1 unit = 1,200 training steps)", "Full validation (873 frames), 8 units, 2 seeds; then test once"),
        @("Use it to", "rank many ideas, kill bad ones early, debug", "confirm the finalists and write the headline numbers"),
        @("Trust it for", "ranking only: resolution is the noise band, 0.010", "claims, if the difference is larger than the noise"),
        @("Do not", "announce a winner from it", "spend it on ideas nobody screened")
    )
    $null = Add-Table $s $rows 40 198 880 20 12 40 @() -wts @(1.3, 4.3, 4.3)
    $null = Add-Callout $s "Check that the small test ranks like the large one. Our design D6 looked best at 1 unit (0.708), then fell to 0.67 at 2 units. A favourable draw, not a good design." 40 410 880 58 $AMBERBG $NAVY 14
    $null = Add-Text $s "Also learned: our baseline score was flat from 2 units on, so more training did not help it; the ladder stops spending where the curve is flat." 40 474 880 30 12 $GREY
    Add-Footer $s
    Add-Notes $s "Numbers: VAL250 = 250-frame subset of the 873 validation frames; TEST 833 frames = 21 scenes; TRAIN 8,414 frames. Ladder of loop 16: 8 designs on rungs of 1, 2, 4, 8 units, half cut at each rung. D6: 0.708 at rung 1, 0.67 at rung 2. Control (E94) S 0.713 / 0.710 / 0.707 / 0.725 at 1 / 2 / 4 / 8 units, seed 0. Gates (cheap checks before training): geometry parity with a numpy reference, operation scan (no argmax, matching or rule step), gradient reaches every parameter group. Related method: successive halving / Hyperband (arXiv 1603.06560)."

    # pipeline: modular vs joint
    $s = New-Slide
    Add-Tag $s "Experiment quickly"
    $null = Add-Title $s "Train in modules first, then together"
    $fl = @(@("Frozen features", "computed once, cached"), @("Heads, one at a time", "each with its own metric"), @("Assemble the chain", "check speed and identical output"), @("Joint fine-tune", "one loss reaches every module"))
    for ($i = 0; $i -lt 4; $i++) {
        $x = 40 + $i * 224
        $fc = $BLUEBG; if ($i -eq 3) { $fc = $GREENBG }
        $null = Add-Node $s $x 96 200 66 ($fl[$i][0] + "`r" + $fl[$i][1]) $fc $NAVY 12 $true (C "9DB8E8")
        if ($i -lt 3) { $null = Add-Line $s ($x + 202) 129 ($x + 222) 129 $BLUE 2.5 $true }
    }
    $null = Add-Card2 $s 40 184 430 224 $BLUE
    $null = Add-Text $s "Modular: one block at a time" 56 196 400 24 17 $BLUE $true $HFONT
    $null = Add-Bullets $s @("Fast:| the heavy feature network is frozen, its output is cached, so a try costs minutes.", "Clear:| when a number moves you know which block did it.", "Use it for:| every new idea on one block, and all early rungs.", "Limit:| the parts never adapt to each other.") 56 226 400 180 13 $NAVY $BLUE
    $null = Add-Card2 $s 490 184 430 224 $GREEN
    $null = Add-Text $s "Joint: all blocks together" 506 196 400 24 17 $GREEN $true $HFONT
    $null = Add-Bullets $s @("Why:| one loss from the final output flows back into every module, so they fit each other.", "Cost:| hours, and it can derail (we cap the learning rate, clip, and roll back).", "Use it for:| the finalists, after the heads work, with 4 - 8 units.", "Rule:| adopt only if it beats the modular version at equal budget by more than the noise.") 506 226 400 180 13 $NAVY $GREEN
    $null = Add-Callout $s "Order matters: modular first, joint second. So far our joint fine-tune gave +0.07 against a seed spread of 0.05 - 0.09: not yet proven." 40 424 880 54 $AMBERBG $NAVY 14
    Add-Footer $s
    Add-Notes $s "Project facts: foundation features are frozen and cached; heads and builder are trained on cached features; PROGRAM.md work package 3 plans a joint E2E fine-tune after modular training (4 - 8 units). Pitfalls recorded: training can derail (BatchNorm running variance); lr <= 6e-4, gradient clipping, roll-back guard in trainer.fit; add the time recursion only after the seed heads have learned. Joint fine-tune night hold (E202): +0.07 route success, one seed, inside the 0.05 - 0.09 seed spread. Every comparison also reports ms per frame."

    # 6 setup
    $s = New-Slide
    Add-Tag $s "Our setup"
    $null = Add-Title $s "One person, one agent loop, one graphics card"
    $st = @(@("You", "write PROGRAM.md: goal, rules, stop time"), @("Agent loop", "wakes every 20 - 25 min, reads state, decides"), @("run.py", "starts the job, logs it, guards the GPU"), @("Fixed judge", "evaluation code the agent cannot change"), @("Ledger + state", "result and decision written down"))
    for ($i = 0; $i -lt 5; $i++) {
        $x = 40 + $i * 178
        $fc = $BLUEBG; if ($i -eq 0) { $fc = (C "FCE8CF") }
        $null = Add-Node $s $x 100 150 90 ($st[$i][0] + "`r" + $st[$i][1]) $fc $NAVY 12 $true (C "9DB8E8")
        if ($i -lt 4) { $null = Add-Line $s ($x + 152) 145 ($x + 176) 145 $BLUE 2.5 $true }
    }
    $sx = @(@("RTX 3060", "12 GB: at most 2 - 3 jobs at once", $ORANGE), @("3 helpers", "literature readers, code builders, number reviewers", $GREEN), @("21 scenes", "in the small test set, kept for one final check", $BLUE), @("nuScenes", "public driving data: camera and radar", $BLUE))
    for ($i = 0; $i -lt 4; $i++) {
        $x = 40 + $i * 224
        $null = Add-Card2 $s $x 214 212 100 $sx[$i][2]
        $null = Add-Text $s $sx[$i][0] ($x + 14) 226 190 34 24 $sx[$i][2] $true $HFONT
        $null = Add-Text $s $sx[$i][1] ($x + 14) 266 190 44 12 $GREY
    }
    $null = Add-Bullets $s @(
        "The project:| a lane-centre graph from camera and radar images, for driving. Each experiment takes minutes to hours.",
        "Harder than the simple setups:| trials are slow, the score has several parts that can be gamed, and the test set is small.",
        "Gap:| no version control (git) in the folder yet, so no automatic undo.") 40 334 880 150 14 $NAVY $BLUE
    Add-Footer $s
    Add-Notes $s "Agents: literature agents (hundreds of paper cards), builder agents (code in isolation), reviewer agents (fresh context, primary sources). The user owns the question, the test set and the claims."

    # 7 rules
    $s = New-Slide
    Add-Tag $s "Trust"
    $null = Add-Title $s "Four rules make the numbers trustworthy"
    $rl = @(
        @("A fixed judge", "The scoring code is tested and the loop cannot change it. No one grades their own homework.", $BLUE),
        @("Measure the luck", "The same recipe trained twice scores a little differently (about 0.01 here). Smaller differences are ties, not wins.", $ORANGE),
        @("Write everything down", "Every try, also the failed ones, gets a ledger row with question, rule and result. Nothing is lost or repeated.", $GREEN),
        @("People keep the decisions", "The question, the final test data, the wording of claims and anything irreversible stay with a person.", $AMBER)
    )
    for ($i = 0; $i -lt 4; $i++) {
        $x = 40 + ($i % 2) * 450; $y = 100 + [Math]::Floor($i / 2) * 170
        $null = Add-Card2 $s $x $y 430 156 $rl[$i][2]
        $null = Add-Num $s ($x + 16) ($y + 18) 36 ([string]($i + 1)) $rl[$i][2]
        $null = Add-Text $s $rl[$i][0] ($x + 64) ($y + 22) 350 30 18 $NAVY $true $HFONT
        $null = Add-Text $s $rl[$i][1] ($x + 64) ($y + 62) 350 88 14 $NAVY
    }
    $null = Add-Callout $s "In our first weeks reviews found wrong numbers in nearly every loop: rows dropped, lucky differences, a baseline that was too easy. These rules exist because of that." 40 440 880 56 $AMBERBG $NAVY 14
    Add-Footer $s
    Add-Notes $s "Examples: 140 of 809 night rows silently dropped (loop 8); seed spread of the route head 0.05 - 0.09 instead of the believed 0.03; a map-only baseline scoring 0.693 that showed the map was nearly an oracle. Rules are in CLAUDE.md. Tools: tools/experiment/run.py, decide.py, check_numbers.py. Checkpoints: copy of the model saved during training. Test data only after a logged OK; nothing deleted, only moved to a to-delete folder."

    # 8 noise
    $s = New-Slide
    Add-Tag $s "Why it matters most"
    $null = Add-Title $s "Noise: why one number is never enough"
    $null = Add-Image $s "$DECK\s_noise.png" 40 92 880 270 "effects against noise"
    $null = Add-Bullets $s @(
        "What it is:| train the same recipe twice with different random starts and the score differs a little. Here about 0.010 on the main score, 0.05 - 0.09 on route success under sensor failure.",
        "Why a loop needs it:| a loop that tries 100 ideas finds winners by luck alone. Without a measured noise band it adopts them, and the next ideas build on sand.",
        "In our week:| five of seven effects on the main score sat inside the band. Rule: smaller than the noise is a tie, the cheaper variant wins.") 40 368 880 136 14 $NAVY $BLUE
    Add-Footer $s
    Add-Notes $s "Left chart: loop 16 rung effects against the band 0.010. Right chart: failure series route success against the seed spread; only the memory effects clear it. Data in the weekly report 2 and digest_week2.md. Interval for a difference: paired scene bootstrap (evaluation/sparse_lanes/bootstrap.py)."

    # 9 tree
    $s = New-Slide
    Add-Tag $s "Organisation"
    $null = Add-Title $s "Ideas, ledger and state files keep it organised"
    $null = Add-Text $s "Start with one question, split it into ideas that could be true, test each with one experiment, and let the result decide where to dig next." 40 92 880 40 15 $NAVY
    $bx = @(@("Plan: the tree", "One question, split into ideas. Written before the loop starts.", $BLUE), @("Ledger: the results", "One row per experiment: question, prediction, rule, result, decision. Failed and cut ideas stay.", $GREEN), @("State files: where we are", "Which ideas are open, running or closed. A new colleague or chat continues from here.", $AMBER))
    for ($i = 0; $i -lt 3; $i++) {
        $x = 40 + $i * 300
        $null = Add-Card2 $s $x 140 280 150 $bx[$i][2]
        $null = Add-Text $s $bx[$i][0] ($x + 16) 156 250 26 17 $bx[$i][2] $true $HFONT
        $null = Add-Text $s $bx[$i][1] ($x + 16) 190 250 96 14 $NAVY
        if ($i -lt 2) { $null = Add-Line $s ($x + 282) 215 ($x + 298) 215 $GREY 2.5 $true }
    }
    $null = Add-Node $s 40 335 200 60 "Does the model survive sensor failure?" $BLUE $WHITE 12 $true
    $hy = @(@("memory at the input", "E195: yes, adopt", $GREENBG), @("extra recursion", "E196: tie, cut", (C "EEF1F6")), @("learned gate", "E199: small gain, optional", $AMBERBG))
    for ($i = 0; $i -lt 3; $i++) {
        $y = 312 + $i * 56
        $null = Add-Line $s 240 365 268 ($y + 20) $GREY 1.5 $false
        $null = Add-Node $s 268 $y 220 44 $hy[$i][0] $PANEL $NAVY 12 $true (C "9DB8E8")
        $null = Add-Line $s 490 ($y + 22) 510 ($y + 22) $BLUE 2 $true
        $null = Add-Node $s 510 $y 240 44 $hy[$i][1] $hy[$i][2] $NAVY 12 $false
    }
    $null = Add-Text $s "Real example from loop 17 (simplified): each result is one ledger row." 770 330 150 70 12 $GREY
    Add-Footer $s
    Add-Notes $s "Plan = experiment_execution/PLAN.md and PROGRAM.md; ledger = experiment_execution/LEDGER.csv; state = STATE.md and the loop state files; HANDOVER.md for a new chat. Ids are catalogue ids of loop 17."

    # 10 ask
    $s = New-Slide
    Add-Tag $s "Next step"
    $null = Add-Title $s "What we ask for"
    $ak = @(
        @("Allow unattended runs", "Nightly and weekend loops on our GPU, inside the written rules: stop time, limits, no test data without OK."),
        @("Make it the way of working", "The four rules apply to every automated experiment; a second person reviews headline numbers."),
        @("Provide the basics", "Stable GPU time, a version-control setup (git) for the code, and an agent tool budget."),
        @("Keep people in charge", "A named owner sets the question and the success measure and signs off every claim that leaves the team."),
        @("Start small and measure", "One more team runs a one-month pilot; we compare experiments per week and wrong-number findings against today.")
    )
    for ($i = 0; $i -lt 5; $i++) {
        $y = 96 + $i * 66
        $null = Add-Card $s 40 $y 880 58 $PANEL
        $null = Add-Num $s 52 ($y + 11) 36 ([string]($i + 1)) $BLUE
        $null = Add-Text $s $ak[$i][0] 104 ($y + 8) 230 42 15 $BLUE $true $HFONT
        $null = Add-Text $s $ak[$i][1] 340 ($y + 8) 570 46 13 $NAVY
    }
    $null = Add-Callout $s "The game changer is not the AI agent. It is a loop that never forgets, never skips the log, and works at night, under rules that people own." 40 436 880 58 $GREENBG $NAVY 15
    Add-Footer $s
    Add-Notes $s "No cost figures are given; the budget items need an estimate from IT and from the agent tool provider. The pilot compares experiments per week and the number of wrong-number findings in reviews against the current way of working."

    $pres.SaveAs($Out)
    Write-Output ("saved " + $Out + " slides " + $pres.Slides.Count)
    $pngdir = Join-Path $env:TEMP "workflow_deck_png"
    if (Test-Path $pngdir) { Remove-Item $pngdir -Recurse -Force }
    New-Item -ItemType Directory -Path $pngdir | Out-Null
    $pres.Export($pngdir, "PNG", 1280, 720)
    Write-Output ("exported to " + $pngdir)
} catch {
    Write-Output ("ERROR: " + $_.Exception.Message + " at " + $_.InvocationInfo.ScriptLineNumber)
} finally {
    $pres.Close()
    $app.Quit()
}
