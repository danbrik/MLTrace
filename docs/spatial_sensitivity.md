# Spatial ROI Sensitivity — vier Änderungskarten

Methodikversion: `mad_median_q95_v2`.

## Stichproben und Referenz

Die vorhandenen Datensatzregeln, Strides, geschlossenen Ereignisintervalle und
halboffenen Normalfenster bleiben unverändert. Standardmäßig werden unabhängig
je 1.000 valide Originalbilder ohne Zurücklegen ausgewählt; kleinere Größen sind
weiterhin einstellbar, die Obergrenze ist 1.000. Seed: 42. Bei ausgeschöpftem
Fenster wird die kleinere gefundene Stichprobe verwendet. Ungültige Treffer
werden deterministisch ersetzt und protokolliert. Ohne valides Bild schlägt
das Fenster fehl.

Alle Berechnungen erfolgen auf vollständigen 1280×960-uint16-Graustufenbildern.
Die feste Polygon-ROI beeinflusst weder Auswahl noch Berechnung der Karten.
Warp bleibt ausschließlich Vorschau. Die rechnerischen Arrays werden als
float32 gespeichert, Summen und Aggregationen intern mit float64 berechnet.

Für Pixel p und Stichproben S_Ni, S_Ui:

    M_Ni(p)   = median_{t ∈ S_Ni} I_t(p)
    MAD_Ni(p) = median_{t ∈ S_Ni} |I_t(p) − M_Ni(p)|
    M_Ui(p)   = median_{t ∈ S_Ui} I_t(p)

## Ereigniskarten

    D_med_i(p) = |M_Ui(p) − M_Ni(p)|
    R_med_i(p) = D_med_i(p) / (MAD_Ni(p) + 1)
    D_q95_i(p) = Q_0.95,t∈S_Ui [|I_t(p) − M_Ni(p)|]
    R_q95_i(p) = Q_0.95,t∈S_Ui [|I_t(p) − M_Ni(p)| / (MAD_Ni(p) + 1)]

Es gibt keinen MAD-Skalierungsfaktor. Epsilon ist fest 1.0. Quantile verwenden
NumPys lineare Interpolation (`method="linear"`), auch bei kleinen Stichproben.
R_q95 wird aus den normalisierten Frame-Abweichungen berechnet, nicht aus einem
bereits gerundeten D_q95-Array. Die Event-Median- und Q95-Berechnungen erfolgen
räumlich blockweise auf der temporären Memmap.

## ROI-Kennzahlen und Aggregation

Für jede Karte F sind F_in und F_out die ungeclippten Mittelwerte über die
Innen- bzw. Außenpixel. Leere und vollflächige Masken sind ungültig.

    Q_D_med = D_med_in / max(D_med_out, 10⁻¹²)
    Q_R_med = R_med_in / max(R_med_out, 10⁻¹²)
    Q_D_q95 = D_q95_in / max(D_q95_out, 10⁻¹²)
    Q_R_q95 = R_q95_in / max(R_q95_out, 10⁻¹²)

Für positive Außenmittelwerte über der numerischen Untergrenze gilt exakt:
1 = gleiche mittlere Stärke, 3 = dreifache mittlere Stärke innen. Bei fehlendem
Außensignal ist der Quotient durch die Untergrenze begrenzt und nicht als
verlässlicher multiplikativer Effekt zu interpretieren; 0/0 wird zu 0.

Für D_med und D_q95 werden C_in, C_out (Pixelsummen) und
`P_in = 100 × C_in / (C_in + C_out)` berechnet. Ist die Gesamtsumme null,
wird P_in konventionsgemäß 0. Diese Anteile sind von der ROI-Fläche abhängig.

    D_agg_med(p) = mean_i D_med_i(p)
    R_agg_med(p) = mean_i R_med_i(p)
    D_agg_q95(p) = mean_i D_q95_i(p)
    R_agg_q95(p) = mean_i R_q95_i(p)

Ereignisse werden gleich gewichtet. Die Tabellen-Medianzeile ist für jede
numerische Spalte separat der Median der Ereigniswerte, nicht die Kennzahl der
aggregierten Karte. Kennzahlen der aggregierten Karten stehen separat im Ergebnis.

## Artefakte und Farbskalen

- `event_NNN_arrays.npz`: median_normal, mad_normal, median_event und alle vier Karten.
- `aggregate_arrays.npz`: D_agg_med, R_agg_med, D_agg_q95, R_agg_q95.
- Vier Einzelheatmaps je Ereignis, vier Ereignisraster, vier Aggregationsheatmaps,
  vier Inside-/Outside-Balkendiagramme und eine Übersichtsabbildung.
- PNG mit 300 dpi und PDF, unverändert vollständiges Bild mit ROI-Kontur.
- CSV, Konfigurationssnapshot, Manifest und ZIP bleiben erhalten.

Jede Metrik erhält eine eigene Skala: 0 bis zum exakten 0.995-Quantil aller
Ereigniswerte und Pixel. Einzelkarten, Raster, Übersicht und Aggregation teilen
dieselben metrikspezifischen Grenzen. Bei Quantil 0 wird für die Darstellung
1 als Obergrenze verwendet. Die Quantilbestimmung partitioniert eine temporäre
plattenbasierte Kopie; die Originalkarten werden nicht verändert. Clipping
betrifft ausschließlich die Darstellung.

## Historische Ergebnisse

Die Methodikversion ist Teil der Konfigurationssignatur. Gespeicherte alte
Konfigurationen werden beim Lesen für den Editor auf die neue Methodik,
Epsilon 1, Seed 42 und höchstens 1.000 Bilder pro Fenster angepasst. Historische
Snapshots und Artefakte bleiben unverändert; sie werden nicht in R-Karten
umbenannt oder als neue Ergebnisse wiederverwendet. Die Oberfläche bietet deren
Originalarchiv zum Download an. Bereits wartende Läufe mit alter Methodik müssen
aus ihrer Konfiguration neu gestartet werden. Neu gespeicherte Konfigurationen
finden exakt passende bereits berechnete Vier-Karten-Ergebnisse über die Signatur.
