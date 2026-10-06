# Zeitabstandsanalyse: gemeinsame Startpunkte und Bildvergleich

Neue Entwürfe speichern `selection_version: 2`, `block_seconds: 300` und `seed: 42`. Das gespeicherte Datensatzsampling gilt zuerst. Der mögliche Startbereich endet beim Zeitraumende minus dem größten Abstand. Jeder Zeitblock liefert höchstens einen zufälligen vorhandenen Startpunkt, der Partner für sämtliche Abstände besitzt. Partner werden im selben Zeitraum innerhalb ±0,5 Sekunden gesucht; nächster Treffer, bei Gleichstand der frühere. Damit stammen alle Kurvenpunkte aus denselben Situationen. Änderungen an Abständen können die Kandidaten und gezogenen Startpunkte verändern.

Die Zeitrechnung verwendet reale verstrichene Sekunden gemäß der bestehenden Europe/Berlin-Konvention. Blöcke beginnen am Zeitraumbeginn; innere Grenzen gehören zum nächsten Block, das Ende zum letzten. Restblöcke und ein einzelner gültiger Zeitpunkt werden berücksichtigt. Ohne gültige Auswahl in beiden Rollen kann kein Lauf starten. Alte Konfigurationen ohne Auswahlversion bleiben Version 1 und verwenden alle exakten Paare. Alte Ergebnisse und wartende Läufe werden nicht verändert.

## Bildvergleich

Bei fertigen Läufen kann zwischen Referenz und Vergleich umgeschaltet werden. Wähle ein bis drei berechnete Abstände und bis zu drei vorgeschlagene Startpunkte. Jede Zeile enthält das vorverarbeitete Startbild und die absoluten Differenzen zu diesem Bild. Die gemeinsame Differenzfarbskala entsteht aus den ungefilterten Karten. „Top 1 %“ filtert jede Karte anhand ihres Quantils; Gleichstände bleiben sichtbar, Nullwerte werden ausgeblendet. Der Anteil verändert weder Farbgrenzen noch die numerischen Analysewerte.

Die letzte erfolgreich erzeugte Matrix und ihre Einstellungen werden pro Rolle im Laufverzeichnis gespeichert. Vorschau und PNG-Download verwenden dieselbe Datei. Die PNG-Metadaten enthalten eingefrorene Datensatz-/Pipelinebeschreibungen, Zeitpunkte und Filter. Eine spätere neue Berechnung benötigt unveränderte Quelldateien; vorhandene PNGs bleiben ohne Quellenzugriff verfügbar. Löschen des Elternlaufs entfernt sämtliche Matrixartefakte mit.

## API

Unter `/api/temporal-difference/runs/{id}/matrix` liest GET mit `role` den gespeicherten Zustand. POST akzeptiert `role`, `deltas_seconds` (1–3), `start_times` (1–3 unterschiedliche gespeicherte Aufnahmezeitpunkte) und `top_percent` (null oder 0,01–100). GET `/start-points` erhält `role` und wiederholte `deltas`-Parameter und liefert verfügbare und vorgeschlagene Startpunkte. GET `/png` erhält `role`, den zurückgegebenen Artefaktnamen und optional `download=true`. Alle Zugriffe sind projektgebunden und auf fertige Läufe beschränkt. Es ist keine Datenbankmigration erforderlich.

## Prozentdarstellung und Zoom

Die sichtbaren Rollen heißen Normal und Anomalie; intern bleiben `reference` und
`comparison` erhalten. Kurve, Tabellen und Bildmatrix zeigen Änderungen als
`100 × absolute Änderung / 65535`, unabhängig vom Datentyp oder Preprocessing.
Es wird weder bildweise normalisiert noch bei 100 % abgeschnitten. Startbilder
bleiben in Graustufen. Median und IQR sowie der Top-Prozent-Filter bleiben
methodisch unverändert.

Die Datenbank und ursprünglichen CSV-Artefakte enthalten weiterhin Rohwerte.
`summary`, `pairs` und `csv/{kind}` akzeptieren `unit=raw|percent` (Default raw).
Die Oberfläche fordert percent an; CSV-Spalten tragen das Suffix `_percent`.
Matrix-Exporte ab Darstellungsversion 2 enthalten Einheit und Bezugswert in ihren
Metadaten. Alte PNGs bleiben unverändert verfügbar und werden als alte Einheiten
gekennzeichnet; eine Neuerstellung verwendet Prozentwerte.

Plot-Einstellungen mit `unit_version: 2` verwenden Prozentgrenzen. Beim Öffnen
älterer Einstellungen rechnet die Oberfläche Y-Grenzen einmal um und ersetzt den
alten Standardtitel; eigene Titel bleiben erhalten. „Zoom zurücksetzen“ stellt
manuelle Grenzen wieder her oder aktiviert für die jeweilige Achse Autorange.
