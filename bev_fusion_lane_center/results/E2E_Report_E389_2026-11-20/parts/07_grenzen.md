# 7. Ehrliche Negative und Grenzen

**Was vor dem Lauf feststand und das Ergebnis begrenzt.**

1. **Die Karte ist simuliert.** Die „SD-Route“ ist aus der HD-GT abgeleitet (Fahrbahnmittel der Routenstücke, Spurwahl verloren) und mit Fehlern nach dem Protokoll versehen (Versatz 1 - 2 m, Gier, Ausdünnung, Spurzahl-Fehler). Eine echte Navigationskarte hat andere, korrelierte Fehler (Straßenversatz entlang ganzer Abschnitte, fehlende Straßen, veraltete Spurzahl). Das Ergebnis sagt, was ein Kopf unter *diesen* Fehlern lernt.
2. **Die Route ist die gefahrene Spur.** Das Tier-1-Label liest den zukünftigen Weg des Fahrzeugs (bis 40 m nach vorn entlang der Karte). Das ist als Label zulässig und nie eine Eingabe (Kapitel 3, Punkt 6); es ist aber **nicht die geplante Route** eines Navigationssystems. An Gabelungen, an denen das Fahrzeug von der geplanten Route abweicht, wäre das Label falsch.
3. **Das Label `on_road` ist eine deterministische Funktion der Position und Richtung des Tokens und der wahren Route.** Die Aufgabe ist Registrierung unter dem Kartenfehler. Ein Kopf, der sie löst, hat gelernt, eine verschobene Polylinie auf Token abzubilden. Das ist weniger als ein Verstehen der Straßenstruktur.
4. **Der Kopf ohne Karte ist nicht schwach.** Ein gerader Prior (|x| <= 3.5 m, Richtung nach vorn) erreicht schon einen großen Teil der Fahrbahn-F1 ({{E389_REF_STR_F1}}); der Spielraum zwischen ihm und einer Regel mit Karte ist der Raum, in dem die Karte etwas bewirken kann ({{E389_REF_GER_F1}} auf der fehlerhaften Route). Die Differenz m - z ist daher nicht „Kopf gegen nichts“.
5. **Die Kappung der Routen-Tabelle** (64 Stützstellen, 2 m Abstand) schneidet die Route bei vielen Frames ab; auf dem abgeschnittenen Teil kann kein Kopf die Fahrbahn finden. Das begrenzt die erreichbare F1 des Kopfes mit Karte unabhängig vom Training (Kapitel 4, letzter Absatz).
6. **Gelieferte Token kommen von B', das auf TRAIN besser ist als auf VAL.** Die Token von B' sind auf TRAIN sauberer als auf VAL (E388 / E381: die TRAIN-zu-VAL-Lücke der Token). z und m teilen diese Verschiebung, die absolute F1 ist davon betroffen, die Differenzen sind es, was die Regel liest.
7. **Zwei Seeds, ein Cache-Seed je Kopf, 3000 Schritte.** Das Rauschen eines Kopfes gegen einen anderen misst nur der Zwilling z2 auf einem Seed.
8. **Nicht beansprucht:** alles über TEST; die Stufe 2 des Champions (der Kopf liest die gecachten Token von B'); eine echte Navigationskarte; ein gemeinsames Training mit dem Champion-Kopf (eine eigene Karte); eine Spur-Zuordnung über den Eigenspur-Prior hinaus; ein Einfluss auf die Topologie des Graphen (der war schon in E374 ein TIE).

**Was das Ergebnis selbst an Negativem enthält.** {{E389_NEGATIVES}}
