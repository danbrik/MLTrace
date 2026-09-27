# Referenzbild-Analyse

Der Reiter **Referenzbild-Analyse** im Arbeitsbereich Bilddaten vergleicht Bilder
mit einem pixelweise gemittelten Referenzbild. Es ist kein trainiertes Modell
und keine zusätzliche Methodenkonfiguration erforderlich.

## Ablauf

1. Einen Train/Test-Datensatz und eine Preprocessing-Pipeline auswählen. Die
   Pipeline muss gleich große, einkanalige Graustufenbilder liefern.
2. Den Referenzzeitraum einschließlich Beginn und Ende angeben. Entweder jedes
   n-te Bild auswählen oder eine feste Anzahl zufälliger Bilder mit Seed.
3. Den Anomaliezeitraum einschließlich Beginn und Ende und dessen Sampling
   festlegen. Hier gibt es kein Zufallssampling.
4. Shift, Clip-Grenzen und FPS wählen, **Auswahl prüfen** und **Berechnung starten**.
5. Den gespeicherten Lauf öffnen, MP4 abspielen oder herunterladen und anhand
   eines Aufnahmezeitpunkts ein Einzelbild anzeigen und als PNG herunterladen.

Gespeicherte Datensatzregeln einschließlich ihres Samplings gelten zuerst.
Danach werden Dateien dedupliziert, nach Aufnahmezeitpunkt und Dateipfad sortiert,
auf den jeweiligen Zeitraum eingeschränkt und erneut gesampelt. Rate 15 wählt
Bild 15, 30, 45 …; ein unvollständiger Restblock entfällt. Zufall wählt exakt die
angegebene Anzahl ohne Zurücklegen. Derselbe Seed und dieselbe Ausgangsauswahl
liefern dieselben Dateien. Eine zu große Zufallsanzahl wird nicht still gekürzt.
Referenz- und Anomaliezeitraum dürfen überlappen.

Standards: Sampling 1, Seed 42, Shift 10000, Clip-Minimum 0, Clip-Maximum 12000
und 10 FPS. Bildverarbeitung und FPS gehören zur gespeicherten Konfiguration; für Änderungen einen Lauf **Als
Vorlage übernehmen** und neu berechnen.

## Bedeutung der Darstellung

Die Rechnung verwendet die Pipeline-Ausgabe ohne zusätzliche Normalisierung pro
Bild. Das Referenzbild ist deren arithmetischer Mittelwert in Gleitkommapräzision.
Die Verarbeitung für neue Läufe ist:

```text
diff = Bild − Referenzbild
diff_shifted = diff + shift                 # Standard: 10000
diff_clipped = clip(diff_shifted, clip_min, clip_max)  # Standard: 0, 12000
output = uint16(round(diff_clipped))
```

Shift ist eine endliche Zahl; die Clip-Grenzen sind ganze Zahlen zwischen 0 und
65535, wobei Minimum kleiner als Maximum sein muss. Bruchteile werden nach dem
Clipping auf die nächste ganze Zahl gerundet (bei halben Werten zur geraden Zahl).
Die PNG-Werte werden nicht auf den gesamten 16-Bit-Bereich gestreckt. Ohne Änderung
beträgt der Ergebniswert standardmäßig 10000. Die Beschriftung oben rechts wird
mit derselben Bittiefe eingeblendet; Pixel außerhalb der Beschriftung behalten die
berechneten Werte.

Einzelbilder werden als echte einkanalige 16-Bit-PNGs heruntergeladen. Für Browser-
Vorschau und MP4 wird der Clip-Bereich gleichbleibend auf 0–255 abgebildet. Mit den
Standards erscheint 10000 als helles Grau, nicht als mittleres Grau.

Die bisherige Vorzeichen-Darstellung mit automatischer/manueller Kontrastskala
bleibt als ausdrücklich gekennzeichnete 8-Bit-Option verfügbar. Bestehende Läufe
behalten ihre Darstellung. „Als Vorlage übernehmen“ verwendet die neue
Shift-/Clipping-Verarbeitung. Die mittlere absolute Differenz wird weiterhin vor
Shift und Clipping berechnet; ein unverändertes Bild hat Abstand null.

Jeder Frame enthält oben rechts den Aufnahmezeitpunkt. Das Video verwendet eine
konstante Bildrate ohne Auffüllen zeitlicher Lücken. Die Zeitpunkt-Suche liefert
den exakten Treffer oder den ersten späteren Frame; hinter dem letzten Zeitpunkt
den letzten Frame. Bei einer Abweichung zeigt die Oberfläche beide Zeitpunkte.

PNG speichert die 16-Bit-Werte verlustfrei; Vorschau und MP4 verwenden die daraus
erzeugte 8-Bit-Abbildung. MP4 wird
mit H.264/YUV420p und Faststart für Browser kodiert. Ungerade Bildabmessungen werden
am unteren/rechten Rand um höchstens einen Pixel ergänzt. Das Referenzbild steht
zusätzlich als PNG-Vorschau bereit; sein Gleitkommaoriginal bleibt intern erhalten.

## Speicherung und API

Läufe werden als CPU-Jobs im bestehenden Scheduler ausgeführt. Die Auswahl und die
Pipeline werden beim Start eingefroren; veränderte oder fehlende Quelldateien
führen zu einem Fehler. Bilder werden sequenziell verarbeitet; temporäre
Differenzdateien erlauben eine gemeinsame Skala ohne die Bildserie im RAM zu halten.

Die Migration `0061_reference_image` ergänzt `reference_image_runs`. Sie wird wie
die bestehenden Projektmigrationen beim Backendstart angewendet. Artefakte liegen
projektgebunden unter `reference_image_runs/<id>/`. Im Data Manager lassen sich
Läufe und ihre Abhängigkeiten prüfen und abgeschlossene Läufe löschen. Aktive
Läufe müssen zuerst abgebrochen werden. Nur fertige Ergebnisse sind abrufbar.

API-Basis: `/api/reference-image-analysis` mit bestehendem Projektkontext:

- `POST /preview`: verfügbare/ausgewählte Bilder und Validierungsfehler.
- `POST /runs`, `GET /runs`, `GET /runs/{id}`: Konfiguration und Laufstatus.
- `POST /runs/{id}/abort`, `DELETE /runs/{id}`, `GET /runs/{id}/log`.
- `GET /runs/{id}/results`: Zusammenfassung und Frameindex mit Zeitstempeln/Abständen.
- `GET /runs/{id}/frame?timestamp=...`: exakter oder nächster späterer Frame.
- `GET /runs/{id}/artifacts/{name}`: `video.mp4`, `reference.png`,
  `frame_000000.png` (16 Bit im Shift-/Clipping-Modus) und
  `preview_frame_000000.png` (8-Bit-Vorschau) usw.; mit `download=true` als Download.

Neue Konfigurationsfelder: `processing_mode` (`shift_clip` als Standard, alternativ
`signed`), `shift`, `clip_min`, `clip_max`. Sie werden im vorhandenen Konfigurations-JSON
gespeichert; eine zusätzliche Datenbankmigration ist nicht erforderlich.

Medienlinks führen die Projekt-ID als `project_id` mit. Fehlgeschlagene und
abgebrochene Läufe veröffentlichen keine unvollständigen Artefakte.

## Gespeicherte Zeiträume

Referenz- und Anomaliezeitraum können benannte Vorlagen aus derselben
Projektbibliothek übernehmen, speichern und verwalten. Dabei werden nur Beginn
und Ende kopiert; jede Übernahme erfordert eine erneute Auswahlprüfung.
Die wiederverwendbare Schnittstelle ist in [Projektweite Zeitraumvorlagen](time_range_presets.md)
beschrieben.
