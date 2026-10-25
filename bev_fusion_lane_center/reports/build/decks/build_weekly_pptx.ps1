# Builds the two-week PowerPoint (weeks 1 and 2 of the lane-graph project) with PowerPoint COM (no downloads). ASCII only (special characters via [char]).
# Usage: powershell -File build_weekly_pptx.ps1 [out.pptx]
# Inputs: reports\source\figures\*.png, reports\source\weekly\deck\c*.png (made by make_deck_charts.py). Numbers: digest_week1.md / digest_week2.md.
param([string]$Out = ([System.IO.Path]::GetFullPath("$PSScriptRoot\..\..\weekly\Weeks1-2_LaneGraph_Presentation_EN.pptx")))
$ErrorActionPreference = "Stop"
$PROJ = [System.IO.Path]::GetFullPath("$PSScriptRoot\..\..\..")  # the problem folder, from the location of this script
$FIG = Join-Path $PROJ "reports\source\figures"
$DECK = Join-Path $PROJ "reports\source\weekly\deck"
$AR = [string][char]0x2192; $MI = [string][char]0x2212; $TI = [string][char]0x00D7; $PM = [string][char]0x00B1; $GE = [string][char]0x2265; $LE = [string][char]0x2264; $DOT = [string][char]0x00B7

function C([string]$hex) { $r = [Convert]::ToInt32($hex.Substring(0, 2), 16); $g = [Convert]::ToInt32($hex.Substring(2, 2), 16); $b = [Convert]::ToInt32($hex.Substring(4, 2), 16); return ($r + 256 * $g + 65536 * $b) }
$NAVY = C "14213D"; $BLUE = C "1D4EA3"; $ORANGE = C "C85F00"; $GREEN = C "1A7F3C"; $GREY = C "5B6570"; $PANEL = C "EEF1F6"; $WHITE = C "FFFFFF"; $AMBER = C "9A6A00"; $RED = C "B3261E"; $SOFT = C "D5DBE8"
$FONT = "Calibri"; $HFONT = "Cambria"

$app = New-Object -ComObject PowerPoint.Application
$pres = $app.Presentations.Add(0)
$pres.PageSetup.SlideWidth = 960
$pres.PageSetup.SlideHeight = 540
$script:n = 0
$TOTAL = 24

function New-Slide([bool]$dark = $false) {
    $script:n++
    $s = $pres.Slides.Add($script:n, 12)
    $s.FollowMasterBackground = 0
    $s.Background.Fill.Solid()
    if ($dark) { $s.Background.Fill.ForeColor.RGB = $NAVY } else { $s.Background.Fill.ForeColor.RGB = $WHITE }
    return $s
}

function Add-Text($s, [string]$text, [double]$x, [double]$y, [double]$w, [double]$h, [double]$size, [int]$color, [bool]$bold = $false, [string]$font = $FONT, [int]$align = 1) {
    $tb = $s.Shapes.AddTextbox(1, $x, $y, $w, $h)
    $tf = $tb.TextFrame
    $tf.WordWrap = -1
    $tf.AutoSize = 0
    $tf.MarginLeft = 0; $tf.MarginRight = 0; $tf.MarginTop = 0; $tf.MarginBottom = 0
    $tb.Height = $h
    $tr = $tf.TextRange
    $tr.Text = $text
    $tr.Font.Name = $font
    $tr.Font.Size = $size
    $tr.Font.Bold = [int]$bold * -1
    $tr.Font.Color.RGB = $color
    $tr.ParagraphFormat.Alignment = $align
    return $tb
}

function Add-Title($s, [string]$text, [bool]$dark = $false) {
    $col = $NAVY
    if ($dark) { $col = $WHITE }
    $t = Add-Text $s $text 40 24 880 72 28 $col $true $HFONT
    $t.TextFrame.VerticalAnchor = 1
    return $t
}

function Add-Footer($s, [bool]$dark = $false) {
    $col = $GREY
    if ($dark) { $col = C "B9C2D3" }
    $null = Add-Text $s "Lane-centre-line graph from camera and radar | weeks 1 and 2" 40 510 600 16 10 $col
    $null = Add-Text $s ("" + $script:n + " / " + $TOTAL) 840 510 80 16 10 $col $false $FONT 3
}

function Add-Notes($s, [string]$text) {
    $s.NotesPage.Shapes.Placeholders.Item(2).TextFrame.TextRange.Text = $text
}

function Add-Card($s, [double]$x, [double]$y, [double]$w, [double]$h, [int]$fill, [int]$line = -1) {
    $sh = $s.Shapes.AddShape(5, $x, $y, $w, $h)
    $sh.Adjustments.Item(1) = 0.06
    $sh.Fill.Solid(); $sh.Fill.ForeColor.RGB = $fill
    if ($line -ge 0) { $sh.Line.ForeColor.RGB = $line; $sh.Line.Weight = 1.25 } else { $sh.Line.Visible = 0 }
    $sh.Shadow.Visible = 0
    return $sh
}

function Add-Bullets($s, [string[]]$items, [double]$x, [double]$y, [double]$w, [double]$h, [double]$size, [int]$color, [int]$boldcolor = -1) {
    if ($boldcolor -lt 0) { $boldcolor = $color }
    $plain = @()
    foreach ($it in $items) { $plain += ($it -replace '\|', ' ') }
    $tb = $s.Shapes.AddTextbox(1, $x, $y, $w, $h)
    $tf = $tb.TextFrame
    $tf.WordWrap = -1; $tf.AutoSize = 0
    $tf.MarginLeft = 0; $tf.MarginRight = 0; $tf.MarginTop = 0; $tf.MarginBottom = 0
    $tb.Height = $h
    $tr = $tf.TextRange
    $tr.Text = ($plain -join "`r")
    $tr.Font.Name = $FONT; $tr.Font.Size = $size; $tr.Font.Color.RGB = $color
    $tr.ParagraphFormat.Bullet.Visible = -1
    $tr.ParagraphFormat.Bullet.Character = 8226
    $tr.ParagraphFormat.SpaceAfter = 8
    $tf.Ruler.Levels.Item(1).FirstMargin = 0
    $tf.Ruler.Levels.Item(1).LeftMargin = 16
    for ($i = 0; $i -lt $items.Count; $i++) {
        $idx = $items[$i].IndexOf('|')
        if ($idx -gt 0) {
            $c = $tr.Paragraphs($i + 1).Characters(1, $idx)
            $c.Font.Bold = -1
            $c.Font.Color.RGB = $boldcolor
        }
    }
    return $tb
}

function Add-Image($s, [string]$path, [double]$x, [double]$y, [double]$w, [double]$h, [string]$alt = "") {
    $pic = $s.Shapes.AddPicture($path, 0, -1, $x, $y, -1, -1)
    $pic.LockAspectRatio = -1
    $sc = [Math]::Min($w / $pic.Width, $h / $pic.Height)
    $pic.Width = $pic.Width * $sc
    $pic.Left = $x + ($w - $pic.Width) / 2
    $pic.Top = $y + ($h - $pic.Height) / 2
    if ($alt -ne "") { $pic.AlternativeText = $alt }
    return $pic
}

function Add-Stat($s, [double]$x, [double]$y, [double]$w, [double]$h, [string]$big, [string]$label, [int]$color, [double]$bigsize = 32) {
    $null = Add-Card $s $x $y $w $h $PANEL
    $null = Add-Text $s $big ($x + 16) ($y + 10) ($w - 32) ($bigsize * 1.3) $bigsize $color $true $HFONT
    $null = Add-Text $s $label ($x + 16) ($y + 14 + $bigsize * 1.3) ($w - 32) ($h - 22 - $bigsize * 1.3) 13 $GREY
}

function Add-Table($s, $rows, [double]$x, [double]$y, [double]$tw, [double]$ignored, [double]$size, [double]$rowH = 0, [int[]]$hl = @()) {
    $nr = $rows.Count; $nc = $rows[0].Count
    $widths = @(); $first = [Math]::Round($tw * 0.46); $widths += $first
    for ($k = 1; $k -lt $nc; $k++) { $widths += ($tw - $first) / ($nc - 1) }
    $shp = $s.Shapes.AddTable($nr, $nc, $x, $y, $tw, 20 * $nr)
    $tbl = $shp.Table
    $tbl.ApplyStyle("{2D5ABB26-0587-4C30-8999-92F81FD0307C}")
    for ($c = 1; $c -le $nc; $c++) { $tbl.Columns.Item($c).Width = $widths[$c - 1] }
    for ($r = 1; $r -le $nr; $r++) {
        for ($c = 1; $c -le $nc; $c++) {
            $cell = $tbl.Cell($r, $c)
            $tf = $cell.Shape.TextFrame
            $tf.MarginLeft = 6; $tf.MarginRight = 6; $tf.MarginTop = 3; $tf.MarginBottom = 3
            $tr = $tf.TextRange
            $tr.Text = [string]$rows[$r - 1][$c - 1]
            $tr.Font.Name = $FONT; $tr.Font.Size = $size
            $cell.Shape.Fill.Solid()
            if ($r -eq 1) {
                $cell.Shape.Fill.ForeColor.RGB = $NAVY; $tr.Font.Color.RGB = $WHITE; $tr.Font.Bold = -1
            } else {
                if ($hl -contains $r) { $cell.Shape.Fill.ForeColor.RGB = C "DCE6FA" } elseif ($r % 2 -eq 0) { $cell.Shape.Fill.ForeColor.RGB = $WHITE } else { $cell.Shape.Fill.ForeColor.RGB = $PANEL }
                $tr.Font.Color.RGB = $NAVY
                if ($c -eq 1 -or ($hl -contains $r)) { $tr.Font.Bold = -1 } else { $tr.Font.Bold = 0 }
            }
            if ($rowH -gt 0) { $cell.Shape.TextFrame.VerticalAnchor = 3 }
        }
        if ($rowH -gt 0) { $tbl.Rows.Item($r).Height = $rowH }
    }
    return $shp
}

function Add-Callout($s, [string]$text, [double]$x, [double]$y, [double]$w, [double]$h, [int]$fill, [int]$color, [double]$size = 15) {
    $null = Add-Card $s $x $y $w $h $fill
    $null = Add-Text $s $text ($x + 14) ($y + 10) ($w - 28) ($h - 18) $size $color $true
}

function Add-Box($s, [double]$x, [double]$y, [double]$w, [double]$h, [string]$head, [string]$body, [int]$fill, [int]$hcolor, [double]$bsize = 14) {
    $null = Add-Card $s $x $y $w $h $fill
    $null = Add-Text $s $head ($x + 12) ($y + 10) ($w - 24) 26 17 $hcolor $true $HFONT
    $null = Add-Text $s $body ($x + 12) ($y + 40) ($w - 24) ($h - 48) $bsize $NAVY
}

try {
    # ---------------------------------------------------------------- 1 title
    $s = New-Slide $true
    $null = Add-Text $s "Lane-centre-line graph from camera and radar" 60 70 840 24 16 (C "F2A65A") $true
    $null = Add-Text $s "Two weeks, 17 loops: from a frozen foundation to a fully differentiable lane-graph builder" 60 102 840 150 38 $WHITE $true $HFONT
    $null = Add-Text $s "What was built, what worked, what did not, and which numbers can be trusted" 60 268 840 30 20 $SOFT
    $null = Add-Text $s "nuScenes $DOT 11 - 24 October 2026 $DOT loops 1 - 9 (week 1) and 10 - 17 (week 2)" 60 306 840 24 15 (C "B9C2D3")
    $labels = @("6 cameras + radar", "frozen foundation", "dense head + graph", "map + route", "differentiable builder", "planner")
    for ($i = 0; $i -lt 6; $i++) {
        $x = 60 + $i * 143
        $fillc = C "22366B"; $txtc = $WHITE
        if ($i -eq 5) { $fillc = C "F2A65A"; $txtc = $NAVY }
        $b = Add-Card $s $x 400 128 62 $fillc (C "5F7DC4")
        $b.TextFrame.TextRange.Text = $labels[$i]
        $b.TextFrame.TextRange.Font.Name = $FONT; $b.TextFrame.TextRange.Font.Size = 14; $b.TextFrame.TextRange.Font.Bold = -1; $b.TextFrame.TextRange.Font.Color.RGB = $txtc
        if ($i -lt 5) { $a = $s.Shapes.AddShape(33, $x + 130, 424, 12, 14); $a.Fill.ForeColor.RGB = C "B9C2D3"; $a.Line.Visible = 0 }
    }
    Add-Notes $s "Two weeks of work on the lane-centre-line graph project. Week 1: loops 1-9, perception first and assignment after. Week 2: loops 10-17, learned and fully differentiable builders, and robustness under sensor failure. All numbers are VAL unless marked TEST."

    # ---------------------------------------------------------------- 2 task
    $s = New-Slide
    $null = Add-Title $s "The task: a sparse lane graph with a route, handed to a planner"
    $null = Add-Image $s "$FIG\dec_arch.png" 40 100 880 270 "Two chains: A modular, B learned differentiable"
    $null = Add-Bullets $s @(
        "Input:| six cameras and radar sweeps, nuScenes subset of 255 scenes (212 train / 22 val / 21 test), frozen map-pretrained foundation.",
        "Output:| directed graph of lane centre lines with splits and merges, a route flag per lane from a simulated SD navigation map, trust signals, ego motion.",
        "Target:| accurate AND fast, and in the end every module on the training path differentiable.") 40 380 880 120 15 $NAVY $BLUE
    Add-Footer $s
    Add-Notes $s "Chain A is the modular perception-first pipeline (most accurate). Chain B is the learned, differentiable chain, the end-to-end candidate."

    # ---------------------------------------------------------------- 3 timeline
    $s = New-Slide
    $null = Add-Title $s "Two weeks at a glance: seventeen loops, each with one question"
    $w1 = @(@("1", "CRN + distance field", "F1 0.670"), @("2-3", "GaussianCaR, grid head", "+0.243 F1"), @("4", "key points, connectors", "split/merge 0.55/0.61"), @("5", "route transfer, failures", "route 0.926"), @("6-7", "fine-tuning, SD map", "0.920 / 0.733"), @("8-9", "perception first, assignment", "route 0.88"))
    $w2 = @(@("10-11", "learned trace net, memory", "F1 -0.017 vs rules"), @("12-13", "calibration, spatial hold", "night 0.34 $AR 0.68"), @("14", "planner consumer", "0.34 vs 0.36 m"), @("15", "differentiable positions", "F1 0.851 $AR 0.863"), @("16", "8 builders, 1 champion", "S 0.746, 31 ms"), @("17", "sensor failure, map", "S 0.42 $AR 0.62"))
    $null = Add-Text $s "Week 1  (11 - 17 Oct)" 40 100 300 22 16 $BLUE $true $HFONT
    for ($i = 0; $i -lt 6; $i++) {
        $x = 40 + $i * 148
        $null = Add-Card $s $x 126 140 118 $PANEL
        $null = Add-Text $s ("Loop " + $w1[$i][0]) ($x + 10) 134 120 18 13 $BLUE $true
        $null = Add-Text $s $w1[$i][1] ($x + 10) 154 122 44 13 $NAVY
        $null = Add-Text $s $w1[$i][2] ($x + 10) 208 122 32 14 $GREEN $true
    }
    $null = Add-Text $s "Week 2  (18 - 24 Oct)" 40 266 300 22 16 $ORANGE $true $HFONT
    for ($i = 0; $i -lt 6; $i++) {
        $x = 40 + $i * 148
        $null = Add-Card $s $x 292 140 118 $PANEL
        $null = Add-Text $s ("Loop " + $w2[$i][0]) ($x + 10) 300 120 18 13 $ORANGE $true
        $null = Add-Text $s $w2[$i][1] ($x + 10) 320 122 44 13 $NAVY
        $null = Add-Text $s $w2[$i][2] ($x + 10) 374 122 32 14 $GREEN $true
    }
    $null = Add-Callout $s "Rule of the week: every experiment needs a drawing, a numbers table, result images and a time per frame." 40 430 880 56 (C "E8EEFB") $NAVY 15
    Add-Footer $s
    Add-Notes $s "Loops 10 and 11 started on the evening of 17 October. The numbers in the tiles are the headline of each loop."

    # ---------------------------------------------------------------- 4 key numbers
    $s = New-Slide
    $null = Add-Title $s "Six numbers to remember"
    $stats = @(
        @("+0.243", "F1@1 m from swapping the foundation (CRN $AR GaussianCaR, VAL). Larger than any head or loss change.", $BLUE),
        @("0.881 $AR 0.839", "route success at map error 0 $AR 3 m, assignment after perception on 6 m edges (loop 8).", $BLUE),
        @("0.867 $AR 0.259", "graph F1 at night: perception fails silently. Blur 0.433, three front cameras off 0.648.", $RED),
        @("31 ms vs 628 ms", "chain per frame after the cached foundation features: differentiable champion vs control (idle RTX 3060).", $GREEN),
        @("0.746 vs 0.724", "composite S (VAL). The edge is sparsity: 13.5 vs 21.8 vertices per 100 m; J 0.265 vs 0.405.", $AMBER),
        @("0.05 - 0.09", "route-head seed spread. Most route effects of week 2 are inside it; the old estimate was 0.03.", $RED)
    )
    for ($i = 0; $i -lt 6; $i++) {
        $col = $i % 3; $row = [Math]::Floor($i / 3)
        $null = Add-Stat $s (40 + $col * 296) (108 + $row * 200) 280 184 $stats[$i][0] $stats[$i][1] $stats[$i][2] 28
    }
    Add-Footer $s
    Add-Notes $s "Each number is explained in the corresponding slide. VAL unless marked. The seed-spread number is a warning about how to read the route-success differences."

    # ---------------------------------------------------------------- 5 divider W1
    $s = New-Slide $true
    $null = Add-Text $s "WEEK 1  $DOT  LOOPS 1 - 9" 60 160 840 24 16 (C "F2A65A") $true
    $null = Add-Text $s "Perception first, assignment after" 60 192 840 70 40 $WHITE $true $HFONT
    $null = Add-Text $s "A frozen foundation, a dense lane head, a rule-based tracer, learned key points, route transfer by registration, an SD map and a learned assignment on 6 m edges." 60 280 780 90 20 $SOFT
    Add-Footer $s $true

    # ---------------------------------------------------------------- 6 foundation
    $s = New-Slide
    $null = Add-Title $s "The foundation decides: a map-pretrained network is worth +0.24 F1"
    $null = Add-Image $s "$DECK\c1_foundation.png" 40 104 470 340 "CRN vs GaussianCaR F1"
    $null = Add-Bullets $s @(
        "Loop 1:| CRN (detection-pretrained) + distance-field U-Net reached 0.670 at best. The loss sat in the field, not in line forming.",
        "Loops 2 - 3:| GaussianCaR (map-pretrained, frozen): VAL 0.867, TEST 0.835. Decoder level (PCA-32) beats fused logits.",
        "Everything else was small:| U-Net over FCN +0.09, loss weighting +0.003, coordinate channels and temporal fusion no gain.",
        "Caveat:| loop 1 has no test set; its numbers are not comparable.") 540 108 380 360 15 $NAVY $BLUE
    Add-Footer $s
    Add-Notes $s "E27: pixel F1 at 1 m, TRAIN_EVAL / VAL / TEST for CRN 0.647/0.624/0.598 and GaussianCaR 0.926/0.867/0.835."

    # ---------------------------------------------------------------- 7 grid vs sparse + shift
    $s = New-Slide
    $null = Add-Title $s "Grid + tracer beat a sparse extractor, but a scene shift caps everything"
    $null = Add-Image $s "$FIG\E30_e30_decomposition.png" 40 104 520 170 "seen to unseen decomposition"
    $null = Add-Table $s @(@("System", "F1@1 VAL", "route F VAL"), @("Grid + tracer (E31)", "0.879", "0.765"), @("Sparse-only LaneFormer (E32)", "0.821", "0.587"), @("Final E43 (TEST)", "0.876", "0.744")) 40 290 520 20 14 28 @(4)
    $null = Add-Bullets $s @(
        "On training scenes 0.932, held-out training scenes 0.899, VAL 0.851, TEST 0.820.|",
        "Regularisation, mirroring, ensembling, wider read-out and a second foundation do not remove the gap.|",
        "Likely cause (inferred):| the foundation saw the training scenes' map labels.",
        "A refiner, learned links, temporal fusion and a map prior add +0.001 - 0.007 F1.|") 590 108 330 380 14 $NAVY $BLUE
    Add-Footer $s
    Add-Notes $s "E30 decomposition uses the three fold heads, not E37. Do not mix the numbers. The E43 line is TEST: F1 0.876, route F 0.744, 41 ms chain."

    # ---------------------------------------------------------------- 8 topology
    $s = New-Slide
    $null = Add-Title $s "Learned key points lifted split and merge F1 by a factor of 1.6"
    $null = Add-Image $s "$DECK\c2_topology.png" 40 104 520 330 "split merge F1"
    $null = Add-Bullets $s @(
        "Split / merge F1 at 6 m (TEST):| grid graph 0.34 / 0.38, key points 0.55 / 0.61, with Bezier connectors 0.71 / 0.64.",
        "Price:| route F -0.011 (key points) and -0.025 (connectors); node gains over the previous system lie far outside their intervals.",
        "Registration (RAVE-like):| median position error 0.54 $AR 0.43 m; the floor of 0.4 m is far above RAVE's 0.13 m.",
        "No gain:| TopoLogic-style links, PCA-96 read-out, instance norm, node-conditioned head.") 590 108 330 380 14 $NAVY $BLUE
    Add-Footer $s
    Add-Notes $s "E44, E53, E54. 55 percent / 42 percent of confirmed ground-truth nodes have no branch piece, so even oracle key points reach only 0.694/0.752."

    # ---------------------------------------------------------------- 9 route transfer
    $s = New-Slide
    $null = Add-Title $s "A map helps most after registration; the route then transfers by rule-based association"
    $null = Add-Image $s "$DECK\c3_route_transfer.png" 40 104 520 330 "route success vs pose noise"
    $null = Add-Bullets $s @(
        "Exact pose 0.926.| Without registration 0.640 / 0.305 / 0.153 at 1 / 2 / 3 m; with graph-ICP 0.923 / 0.883 / 0.782; tracked 0.920 / 0.920 / 0.918.",
        "Unaligned map error of 1 m is worse than no map| (F1 0.830 vs 0.887).",
        "Hand the planner the registered MAP route:| lateral error 0.25 m vs 0.60 m for the perceived-lane route.",
        "Oracle assignment 0.926:| the 7.4 % misses are extraction, not association.") 590 108 330 380 14 $NAVY $BLUE
    Add-Footer $s
    Add-Notes $s "E59, E61, E62. Geometric transfer: graph-ICP registration grid, heading-constrained ICP, nearest heading-compatible lane within 2 m. Thresholds of the later monitors were fitted on TEST frames (in-sample)."

    # ---------------------------------------------------------------- 10 failure
    $s = New-Slide
    $null = Add-Title $s "Perception fails silently under sensor loss"
    $null = Add-Image $s "$DECK\c4_failure.png" 40 104 560 320 "graph F1 per failure"
    $null = Add-Bullets $s @(
        "Night 0.259, blur 0.433, three front cameras off 0.648:| recall at night 0.84 $AR 0.16.",
        "Head-side training cannot restore lost information:| fine-tuning on perturbed features lifts night to 0.454, blur 0.558; extrinsics and radar +0.00.",
        "All failures are synthetic| perturbations applied before the foundation; no recorded failure was available.") 620 108 300 300 14 $NAVY $RED
    $null = Add-Callout $s "So the system needs layers: a monitor, held registration, a carried-forward graph and map lanes as the last resort." 40 440 880 52 (C "FBECEB") $NAVY 15
    Add-Footer $s
    Add-Notes $s "E57 on TEST. Corruptions: front camera black, three forward cameras black, night (x0.15 + noise), fog, blur (sigma 3 px), radar nearly off (3 points), extrinsics jitter 3 degrees."

    # ---------------------------------------------------------------- 11 layers
    $s = New-Slide
    $null = Add-Title $s "Layered protection: monitor, hold, carry forward, fall back to the map"
    $steps = @(@("1  Monitor", "six output signals + logistic regression; AUROC 0.953 (scene-disjoint 0.935)"), @("2  Hold registration", "keep the last good registration: lateral error 1.32 $AR 0.39 m (map as given vs hold)"), @("3  Carry the graph", "warped last good graph: F1 0.651 $AR 0.741; night 0.293 $AR 0.688; useful for about 5 s"), @("4  Map lanes", "registered map lanes as the lane graph: 0.887 (circular: ground truth = map)"))
    for ($i = 0; $i -lt 4; $i++) {
        $x = 40 + $i * 222
        $null = Add-Box $s $x 110 208 150 $steps[$i][0] $steps[$i][1] $PANEL $BLUE 15
    }
    $null = Add-Bullets $s @(
        "After a map jump:| a restart with a wide search cuts the first-1.5 s error after 3 / 6 / 10 m jumps from 0.58 / 1.54 / 2.84 to 0.25 / 0.25 / 0.55 m; an ungated restart adds +0.36 m.",
        "In-sample:| thresholds of the monitor and of E67 - E72 were fitted on TEST frames, so these results are not independent.",
        "Not needed:| a vision-language-action model (language adds +0.4 driving score inside a noise of +-0.95).") 40 290 880 200 16 $NAVY $BLUE
    Add-Footer $s
    Add-Notes $s "E64, E67 - E72, VLA review. Verification can replace a correct registration by a wrong one: 10 better and 4 worse of 504 scene runs."

    # ---------------------------------------------------------------- 12 perception-first pipeline
    $s = New-Slide
    $null = Add-Title $s "Perception first, assignment after: route success 0.88 on 6 m edges"
    $null = Add-Image $s "$FIG\p8_arch.png" 40 100 880 200 "loop 8 pipeline"
    $null = Add-Table $s @(@("Route success, TEST", "0 m", "1 m", "2 m", "3 m"), @("assignment network (loop 8)", "0.881", "0.879", "0.868", "0.839"), @("loop-9 ensemble (bias + abstention)", "0.876", "0.879", "0.876", "0.858"), @("pixel SD head, registered", "0.733", "0.714", "0.713", "0.710"), @("geometric rule on perceived graph", "0.488", "0.482", "0.483", "0.449")) 40 312 540 20 14 26 @(3)
    $null = Add-Bullets $s @(
        "Unit matters:| whole traced piece 0.454, 10 m edges 0.956, 6 m 0.982, 3 m 0.993 (oracle route success).",
        "0.27 M parameters,| one token per edge, relation-biased transformer.",
        "Bug found:| empty-graph rows were dropped; all loop 8 failure numbers were rebuilt.") 610 312 310 180 14 $NAVY $BLUE
    Add-Footer $s
    Add-Notes $s "E83. 809 TEST rows. The oracle labels (0.913) are not an upper bound. The pixel head comparison is confounded (different front end and rows)."

    # ---------------------------------------------------------------- 13 map faults
    $s = New-Slide
    $null = Add-Title $s "Retraining against map faults was worth more than correcting them"
    $null = Add-Image $s "$FIG\p9_results.png" 40 104 560 190 "loop 9 results"
    $null = Add-Table $s @(@("TEST", "loop 8", "loop 9"), @("route success, per-road bias 2 m", "0.663", "0.847"), @("route flagged, wrong map (m)", "60.7", "4.2"), @("route flagged, no map (m)", "60.1", "0.0"), @("route flagged, right map (m)", "63.3", "63.8")) 40 308 560 20 14 28 @(2)
    $null = Add-Bullets $s @(
        "Bias and abstention training| beat per-road re-registration (+0.011 / +0.007 / -0.012).",
        "Under failure abstention is weak:| a wrong map still gets 4 - 25 m of route; the gate has AUROC 0.618 at night.",
        "Held correction| helps only with trustworthy odometry (0.679 vs 0.589 exact poses; worse at 3 deg + 5 % drift).") 630 108 290 380 14 $NAVY $BLUE
    Add-Footer $s
    Add-Notes $s "E87 - E90. The SD map is simulated from the lane map and the SD route equals the driven route, so extension and reference-path numbers are upper bounds."

    # ---------------------------------------------------------------- 14 divider W2
    $s = New-Slide $true
    $null = Add-Text $s "WEEK 2  $DOT  LOOPS 10 - 17" 60 160 840 24 16 (C "F2A65A") $true
    $null = Add-Text $s "From a learned chain to a fully differentiable builder" 60 192 840 110 40 $WHITE $true $HFONT
    $null = Add-Text $s "A learned trace network, a spatial hold at the input, a planner-side consumer, differentiable positions, eight differentiable builders, and the first test under sensor failure." 60 320 780 90 20 $SOFT
    Add-Footer $s $true

    # ---------------------------------------------------------------- 15 learned chain
    $s = New-Slide
    $null = Add-Title $s "A learned trace network works, but does not beat the rules"
    $null = Add-Image $s "$FIG\p10_e94_arch.png" 40 100 500 200 "E94 architecture"
    $null = Add-Table $s @(@("F1@1 m", "VAL", "TEST"), @("rule pipeline, loose graph", "0.891", "0.867"), @("rule pipeline, final graph", "0.899", "0.875"), @("learned trace network E94", "0.883", "0.851"), @("E94, second seed", "0.884", "0.850")) 40 312 500 20 14 26 @(4)
    $null = Add-Bullets $s @(
        "The gap is real:| paired TEST -0.017 [-0.025, -0.009]; seed noise is +-0.002.",
        "Cross-attention route head (E104b),| no registration step: TEST route success 0.853 / 0.843 / 0.832 / 0.820 at the VAL-selected threshold 0.5.",
        "About 0.007 - 0.026 below| the single rule-based network (0.860 ... 0.839). An early draft said 0.05; most of that was a threshold effect.",
        "Not adopted:| wrong-direction training, crop knock-out, order of growth.") 570 108 350 380 14 $NAVY $BLUE
    Add-Footer $s
    Add-Notes $s "E94, E104b, E111. The learned chain is the end-to-end candidate, the modular pipeline stays the accuracy reference. The route-head seed spread (0.05-0.09) was only established in loop 17."

    # ---------------------------------------------------------------- 16 memory at the input
    $s = New-Slide
    $null = Add-Title $s "The memory that works sits at the input: a parameter-free spatial hold"
    $null = Add-Image $s "$FIG\p12_e112.png" 40 100 440 390 "E112 spatial hold"
    $null = Add-Bullets $s @(
        "How:| previous 42-channel input, warped by ego motion; per pixel take the decayed memory where its heat is higher. No monitor, no parameter.",
        "Loop 12 (TEST):| night F1 0.337 $AR 0.680, blur 0.524 $AR 0.725; radar off flat; three front cameras off 0.653 $AR 0.600.",
        "Loop 17 (VAL, S):| recovers 60 % / 52 % / 13 % of the S loss at night / blur / three cameras off.",
        "Costs:| clean J 0.247 $AR 0.222 and clean route success 0.846 $AR 0.838.",
        "The builder's own time recursion adds nothing.|") 510 108 410 380 14 $NAVY $BLUE
    Add-Footer $s
    Add-Notes $s "E112 (TEST, 10 s failure phase), E195 / E196 (VAL series of loop 17). Calibration: route probability ECE 0.006 clean; route confidence AUROC 0.88-0.91 except for lost front cameras and extrinsics (0.69-0.71)."

    # ---------------------------------------------------------------- 17 consumer
    $s = New-Slide
    $null = Add-Title $s "A planner-side consumer reaches the modular level; the planner loss adds nothing"
    $null = Add-Table $s @(@("Consumer error at 20 m (TEST, m)", "exact map", "wrong map", "no map"), @("geometry only", "0.90", "0.90", "0.90"), @("+ route probability", "0.40", "0.98", "0.98"), @("+ route + ego motion (chain B)", "0.34", "0.76", "0.74"), @("modular chain A, same inputs", "0.36", "0.70", "0.73"), @("+ 128-d step token", "0.56", "0.80", "0.78")) 40 108 520 22 14 30 @(4)
    $null = Add-Bullets $s @(
        "Hand over:| edges + route probability + ego motion, spatial hold at the input, a consumer trained on failure frames (night 2.73 $AR 0.72 m with hold).",
        "Do not hand over| step tokens, and do not hard-cut the graph without a failure gate.",
        "Planner loss reaches the route head and the positions (loop 15) and adds no measurable gain; through the builder only the gradient reach was verified;| a route head trained only on it does not learn the attribute (AUROC 0.53).") 590 108 330 380 14 $NAVY $BLUE
    $null = Add-Callout $s "Loop 15: refining vertex positions end to end (MapTR-style) cuts vertex error 0.54 $AR 0.43 m and lifts F1@1 0.851 $AR 0.863. A seed check showed that the earlier route 'gain' was noise." 40 350 520 130 (C "E8EEFB") $NAVY 14
    Add-Footer $s
    Add-Notes $s "E117 - E121. 546 TEST frames with driven path >= 20 m, three seeds. Half of the vertex-error gain is sliding along the lane."

    # ---------------------------------------------------------------- 18 loop 16 design
    $s = New-Slide
    $null = Add-Title $s "Loop 16: eight differentiable builders, one champion by successive halving"
    $null = Add-Image $s "$FIG\p16_ladder.png" 40 100 520 380 "ladder"
    $null = Add-Bullets $s @(
        "Design:| 7 two-level factors, L8 array = 8 of 128 designs; rungs 8 $AR 4 $AR 2 $AR 1 with 1 / 2 / 4 / 8 training units; gates for gradient reach before any training.",
        "Confirmed:| teardrop pattern +0.053 (four flips), learned pair head -0.040.",
        "Not supported:| keep gate -0.015, stage B / C, three children with MCL, hybrid, more training, time recursion.",
        "Champion D7:| pieces and links, teardrop rays, survival read-out, junction head; 1.2 M parameters.") 570 108 350 270 14 $NAVY $BLUE
    Add-Footer $s
    Add-Notes $s "Successive halving with a control arm (E94 from the same stem, S 0.713/0.710/0.707/0.725 at 1/2/4/8 units). Noise band 0.010."

    # ---------------------------------------------------------------- 19 champion vs control
    $s = New-Slide
    $null = Add-Title $s "The champion is sparse and fast, not more accurate"
    $null = Add-Image $s "$DECK\c6_champion.png" 40 104 540 330 "champion vs control"
    $null = Add-Bullets $s @(
        "S +0.021 VAL / +0.026 TEST| rests on 38 % fewer vertices and less false length; S rewards sparsity up to 14.3 vertices.",
        "F1@1 -0.013, J -0.14, lanes found -0.10.| TEST (used once): S 0.728 vs 0.702, F1@1 0.839 vs 0.851.",
        "Plan criterion met in 1 of 3 points| (vertices); J >= 0.5 and lanes +0.03 not met.",
        "Forks are the weak spot:| 38 % of true forks are under 2 degrees (overlapping ground-truth lanes). A no-parameter relocation step lifted J 0.236 $AR 0.278.") 610 108 310 380 14 $NAVY $BLUE
    Add-Footer $s
    Add-Notes $s "VAL 873 frames. Champion f1 at tau 0.7. Replicate without time recursion S 0.741; replicate with recursion from step 0 failed (S 0.469)."

    # ---------------------------------------------------------------- 20 route gap + speed
    $s = New-Slide
    $null = Add-Title $s "The route gap was a representation error; the chain runs 20 times faster"
    $null = Add-Image $s "$DECK\c7_route_speed.png" 40 104 880 280 "route gap and speed"
    $null = Add-Bullets $s @(
        "Retracted explanation:| the TEST gap (0.72 vs 0.85) was blamed on lane recall; in fact the head was asked once per vertex of a long edge.",
        "Asking along the edges| (points <= 4 m): VAL 0.798 $AR 0.889; consumer error 0.31 $AR 0.26 m.",
        "Speed:| CUDA graphs + vectorised CPU assembly, identical graphs in 120 / 120 frames; foundation network not timed; assembly (13.5 ms) is now the largest stage.") 40 394 880 110 14 $NAVY $BLUE
    Add-Footer $s
    Add-Notes $s "Loop 16b, VAL only. No TEST number exists for the head along the edges and the fast configuration: it needs the user's decision because TEST is used once."

    # ---------------------------------------------------------------- 21 failure loop 17
    $s = New-Slide
    $null = Add-Title $s "Under failure the new builder loses as much as the old one; memory brings part of it back"
    $null = Add-Image $s "$DECK\c5_hold_gate.png" 40 104 540 330 "S under failure"
    $null = Add-Bullets $s @(
        "Protocol:| 22 VAL scenes, clean first half, perturbed second half (about 10 s), 291 frames per type.",
        "Hold +0.20 S at night; the learned gate (12.5 k parameters)| adds +0.021 / +0.015 / +0.001: small, none for lost cameras.",
        "Route success at night| 0.14 $AR 0.36 (hold) $AR 0.42 (gate), against 0.94 on the same frames unperturbed.",
        "Nothing else helped:| builder recursion, failure-trained dense head, fine-tuning on ~77 frames, joint fine-tune.") 610 108 310 380 14 $NAVY $BLUE
    Add-Footer $s
    Add-Notes $s "E193 - E201. Seed spread of route success 0.05-0.09, so gate vs hold on route success is a tie. Milder types (fog, radar off, extrinsics, one camera off) lose 0.04 - 0.13 S."

    # ---------------------------------------------------------------- 22 map
    $s = New-Slide
    $null = Add-Title $s "A good map helps; a poor map does not; a wrong map hurts"
    $null = Add-Image $s "$DECK\c8_map.png" 40 104 540 330 "S with map quality"
    $null = Add-Bullets $s @(
        "Map builder trained on perturbed rows,| gate behind it: S 0.676 / 0.677 / 0.681 with an exact map.",
        "1 m error keeps 36 - 50 % of the map benefit, 2 m none;| wrong map is -0.045 F1 worse than no map. The chain cannot abstain.",
        "Most gain is training on perturbed rows,| not the map (lanes found 0.09 $AR 0.21).",
        "'Exact map' is an upper bound:| the map alone scores S 0.693."
    ) 610 108 310 380 14 $NAVY $BLUE
    Add-Footer $s
    Add-Notes $s "E204-E207, E210. Route success of B + gate with exact map 0.82 / 0.88 / 0.81. Map-only baseline: S 0.693 exact, 0.524 at 1 m, 0.449 at 2 m."

    # ---------------------------------------------------------------- 23 consumer under failure
    $s = New-Slide
    $null = Add-Title $s "For the planner, training on failure frames matters more than the map"
    $null = Add-Image $s "$DECK\c9_consumer.png" 40 104 560 330 "consumer under failure"
    $null = Add-Bullets $s @(
        "Night, single frame 4.11 m $AR 1.78 m with the gate| (champion chain); unperturbed 0.21 m.",
        "A builder trained on perturbed rows| brings night to 0.42 - 0.46 m without any map (unperturbed 0.36 m).",
        "The exact map adds about 0.1 m;| its route is close to the driven path (upper bound).") 620 108 300 250 14 $NAVY $BLUE
    $null = Add-Callout $s "Unseen failure types (E209): no result. The run crashed and no checkpoint exists. Do not quote any number." 40 446 880 46 (C "FBECEB") $RED 15
    Add-Footer $s
    Add-Notes $s "E212: 82 frames per cell, three seeds, consumer trained on the loop-16b training data. E209: design written down, run crashed with a CUDA unknown error."

    # ---------------------------------------------------------------- 24 caveats + next
    $s = New-Slide
    $null = Add-Title $s "What not to trust, and what comes next"
    $null = Add-Box $s 40 104 430 380 "Do not repeat these" ("- Route-head seed spread 'up to 0.03': it is 0.05 - 0.09." + "`r" + "- 'TEST used once': true for loop 16 only; loops 10 - 15 and loop 5 thresholds used TEST." + "`r" + "- 'The SD map fills most of the gap'; 'the champion is more accurate'." + "`r" + "- Loop 8 first-version failure numbers (empty rows dropped)." + "`r" + "- 'Train tokens are out-of-fold' (loop 13); scene count 323 (it is 255)." + "`r" + "- Everything adverse is synthetic and the SD map is simulated from the ground truth.") (C "FBECEB") $RED 16
    $null = Add-Box $s 490 104 430 380 "Next" ("1. Finish loop 17: rerun E209 with checkpoints, one numbers review." + "`r" + "2. TEST once for the head along the edges, fast setting, gate and map builder (allowed once by the loop 18 prompt, not yet run)." + "`r" + "3. Map trust / abstain output; differentiable rough registration of the graph to the SD map." + "`r" + "4. Joint end-to-end fine-tuning with failure frames and a planner loss (more than one seed)." + "`r" + "5. Compiled assembly; recorded failures; a real navigation map; closed-loop test.") (C "E6F4EA") $GREEN 16
    Add-Footer $s
    Add-Notes $s "Full list of corrected claims: section 'Numbers and claims that were corrected' of both weekly reports."

    $pres.SaveAs($Out)
    Write-Output ("saved " + $Out + " slides " + $pres.Slides.Count)
    $pngdir = Join-Path $env:TEMP "weekly_deck_png"
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
