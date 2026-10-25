# Shared PowerPoint COM helpers (dot-source from a deck script; the script must create $app, $pres, set $TOTAL and $FOOT). ASCII only.
$ErrorActionPreference = "Stop"
$PROJ = [System.IO.Path]::GetFullPath("$PSScriptRoot\..\..\..")  # the problem folder, from the location of this script
$FIG = Join-Path $PROJ "reports\source\figures"
$DECK = Join-Path $PROJ "reports\source\weekly\deck"
$AR = [string][char]0x2192; $MI = [string][char]0x2212; $TI = [string][char]0x00D7; $PM = [string][char]0x00B1; $GE = [string][char]0x2265; $LE = [string][char]0x2264; $DOT = [string][char]0x00B7

function C([string]$hex) { $r = [Convert]::ToInt32($hex.Substring(0, 2), 16); $g = [Convert]::ToInt32($hex.Substring(2, 2), 16); $b = [Convert]::ToInt32($hex.Substring(4, 2), 16); return ($r + 256 * $g + 65536 * $b) }
$NAVY = C "14213D"; $BLUE = C "1D4EA3"; $ORANGE = C "C85F00"; $GREEN = C "1A7F3C"; $GREY = C "5B6570"; $PANEL = C "EEF1F6"; $WHITE = C "FFFFFF"; $AMBER = C "9A6A00"; $RED = C "B3261E"; $SOFT = C "D5DBE8"
$FONT = "Calibri"; $HFONT = "Cambria"

$script:n = 0

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
    $null = Add-Text $s $FOOT 40 510 600 16 10 $col
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

function Add-Table($s, $rows, [double]$x, [double]$y, [double]$tw, [double]$ignored, [double]$size, [double]$rowH = 0, [int[]]$hl = @(), [double[]]$wts = @()) {
    $nr = $rows.Count; $nc = $rows[0].Count
    $widths = @()
    if ($wts.Count -eq $nc) { $sum = 0; foreach ($q in $wts) { $sum += $q }; foreach ($q in $wts) { $widths += $tw * $q / $sum } }
    else { $first = [Math]::Round($tw * 0.46); $widths += $first; for ($k = 1; $k -lt $nc; $k++) { $widths += ($tw - $first) / ($nc - 1) } }
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


function Add-Line($s, [double]$x1, [double]$y1, [double]$x2, [double]$y2, [int]$color, [double]$weight = 2, [bool]$arrow = $true) {
    $c = $s.Shapes.AddConnector(1, $x1, $y1, $x2, $y2)
    $c.Line.ForeColor.RGB = $color
    $c.Line.Weight = $weight
    if ($arrow) { $c.Line.EndArrowheadStyle = 2 }
    return $c
}

function Add-Node($s, [double]$x, [double]$y, [double]$w, [double]$h, [string]$text, [int]$fill, [int]$fcolor, [double]$size = 13, [bool]$bold = $true, [int]$line = -1) {
    $sh = Add-Card $s $x $y $w $h $fill $line
    $tr = $sh.TextFrame.TextRange
    $tr.Text = $text
    $tr.Font.Name = $FONT; $tr.Font.Size = $size; $tr.Font.Bold = [int]$bold * -1; $tr.Font.Color.RGB = $fcolor
    $sh.TextFrame.WordWrap = -1
    $sh.TextFrame.MarginLeft = 6; $sh.TextFrame.MarginRight = 6; $sh.TextFrame.MarginTop = 3; $sh.TextFrame.MarginBottom = 3
    $sh.TextFrame.VerticalAnchor = 3
    return $sh
}

function Add-Pill($s, [double]$x, [double]$y, [double]$w, [string]$text, [int]$fill, [int]$fcolor, [double]$size = 12) {
    $sh = $s.Shapes.AddShape(5, $x, $y, $w, 22)
    $sh.Adjustments.Item(1) = 0.5
    $sh.Fill.Solid(); $sh.Fill.ForeColor.RGB = $fill; $sh.Line.Visible = 0; $sh.Shadow.Visible = 0
    $tr = $sh.TextFrame.TextRange
    $tr.Text = $text; $tr.Font.Name = $FONT; $tr.Font.Size = $size; $tr.Font.Bold = -1; $tr.Font.Color.RGB = $fcolor
    $sh.TextFrame.MarginLeft = 4; $sh.TextFrame.MarginRight = 4; $sh.TextFrame.MarginTop = 0; $sh.TextFrame.MarginBottom = 0
    return $sh
}
