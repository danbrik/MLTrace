# Visuelle Analyse von Autoencoder-Anomaly-Scores für Online-Anomaliedetektion

## Ziel

Bei einer Online-Anomaliedetektion darf zum Zeitpunkt `t` nur Information aus der Vergangenheit und Gegenwart verwendet werden:

\[
x_1, x_2, ..., x_t
\]

Es darf keine Information aus zukünftigen Punkten verwendet werden. Deshalb ist das Ziel der visuellen Analyse nicht, nachträglich zu erkennen, dass ein Ereignis groß war, sondern zu untersuchen, **ab welchem Zeitpunkt sich kausale Evidenz für eine echte Anomalie aufbaut**.

Kleine Hügel sollen daran erkennbar sein, dass sie:

- nur kurz ansteigen,
- wenig Evidenz akkumulieren,
- früh wieder abfallen,
- keine längere positive Steigung zeigen.

Große Anomalien sollen daran erkennbar sein, dass:

- der Score über mehrere Punkte hinweg steigt,
- kurzfristige Energie akkumuliert,
- CUSUM/Evidenz weiter anwächst,
- kein schneller Rückfall nach einem lokalen Maximum erfolgt.

---

## Notation

\[
x_t
\]

Anomaly Score bzw. MSE des Autoencoders zum Zeitpunkt `t`.

\[
x_{\text{smooth},t}
\]

Geglätteter Anomaly Score.

\[
W
\]

Kausales Zeitfenster, z. B. `1 min`, `3 min`, `5 min` oder `N` Samples.

\[
\epsilon
\]

Kleine Konstante zur numerischen Stabilisierung.

Alle Rolling-Größen sollen **kausal** berechnet werden, also nur mit Werten bis einschließlich Zeitpunkt `t`.

---

# 1. Originaler Anomaly Score

## Formel

\[
x_t
\]

## Bedeutung

Der rohe Rekonstruktionsfehler des Autoencoders. Dieser Plot ist die Referenz für alle weiteren abgeleiteten Merkmale.

## Wählbare Parameter

Keine direkten Parameter.

## Sinnvoll zu variieren

- Darstellungszeitraum
- lineare oder logarithmische y-Achse
- Markierung bekannter Ereignisse
- Markierung von Kandidatenbereichen

## Nutzen

Zeigt:

- Peaks
- Baseline
- Rauschen
- grobe Ereignisstruktur
- mögliche Drift im Score

---

# 2. Geglätteter Anomaly Score, z. B. EWMA

## Formel

\[
x_{\text{smooth},t}
=
\alpha x_t + (1 - \alpha)x_{\text{smooth},t-1}
\]

## Bedeutung

Kausale Glättung des MSE-Signals. Einzelne Ausreißer werden reduziert, während anhaltende Trends besser sichtbar bleiben.

## Wählbare Parameter

\[
\alpha
\]

## Typische Werte

\[
\alpha = 0.1
\]

\[
\alpha = 0.2
\]

\[
\alpha = 0.3
\]

## Interpretation

Kleines `alpha`:

- stärkere Glättung
- weniger Rauschen
- spätere Reaktion

Großes `alpha`:

- schnellere Reaktion
- stärker rauschanfällig

## Nutzen

Hilft, den Beginn eines echten Anstiegs robuster zu erkennen.

---

# 3. Erste Ableitung / Punkt-zu-Punkt-Steigung

## Formel auf Rohsignal

\[
d_t = x_t - x_{t-1}
\]

## Formel auf geglättetem Signal

\[
d_t = x_{\text{smooth},t} - x_{\text{smooth},t-1}
\]

## Bei unregelmäßigen Zeitabständen

\[
d_t =
\frac{x_{\text{smooth},t} - x_{\text{smooth},t-1}}
{\Delta t}
\]

mit:

\[
\Delta t = time_t - time_{t-1}
\]

## Bedeutung

Zeigt, ob der Anomaly Score aktuell steigt oder fällt.

## Wählbare Parameter

- Signalbasis: `raw score` oder `smoothed score`
- optional zusätzliche Glättung der Ableitung
- Zeitnormalisierung bei unregelmäßigen Zeitabständen

## Nutzen

Eine große Anomalie zeigt oft über längere Zeit positive Steigungen.  
Ein kleiner Hügel zeigt oft nur kurz positive Steigung und fällt dann wieder ab.

---

# 4. Geglättete erste Ableitung

## Formel

\[
d_{\text{smooth},t}
=
\beta d_t + (1 - \beta)d_{\text{smooth},t-1}
\]

## Bedeutung

Glättet die erste Ableitung, da numerische Ableitungen häufig stark rauschen.

## Wählbare Parameter

\[
\beta
\]

## Typische Werte

\[
\beta = 0.1
\]

\[
\beta = 0.2
\]

\[
\beta = 0.3
\]

## Nutzen

Besser interpretierbar als die rohe Punkt-zu-Punkt-Ableitung.

---

# 5. Zweite Ableitung / Beschleunigung

## Formel

\[
a_t = d_t - d_{t-1}
\]

oder auf geglätteter Ableitung:

\[
a_t = d_{\text{smooth},t} - d_{\text{smooth},t-1}
\]

## Bedeutung

Zeigt, ob die Steigung selbst zunimmt oder abnimmt.

## Wählbare Parameter

- Signalbasis: rohe oder geglättete Ableitung
- Glättungsparameter `alpha` und/oder `beta`
- optional Zeitnormalisierung

## Nutzen

Ein kleiner Hügel kann früh erkennen lassen, dass die Steigung nachlässt.  
Ein größerer Event zeigt oft eine länger anhaltende positive Steigung.

## Hinweis

Die zweite Ableitung ist sehr rauschanfällig und sollte nur ergänzend verwendet werden.

---

# 6. Rolling Slope über ein Zeitfenster

## Formel

\[
s_{W,t} = x_{\text{smooth},t} - x_{\text{smooth},t-W}
\]

## Zeitnormalisierte Formel

\[
s_{W,t}
=
\frac{x_{\text{smooth},t} - x_{\text{smooth},t-W}}
{W}
\]

## Bedeutung

Misst den Anstieg über ein kurzes vergangenes Fenster statt nur von Punkt zu Punkt.

## Wählbare Parameter

\[
W
\]

## Typische Werte

Zeitfenster:

- `1 min`
- `3 min`
- `5 min`

Sample-Fenster:

- `6 Samples`
- `12 Samples`
- `30 Samples`

## Nutzen

Robuster als die Punktableitung.  
Sehr relevant, um anhaltende Anstiege von kurzen Hügeln zu unterscheiden.

---

# 7. Rolling Median als lokale Baseline

## Formel

\[
m_t =
median(x_{\text{smooth},i})
\quad
\text{für}
\quad
i \in [t-W, t]
\]

## Bedeutung

Schätzt die lokale Baseline des Signals kausal.

## Wählbare Parameter

\[
W_{\text{baseline}}
\]

## Typische Werte

- `30 min`
- `60 min`
- `120 min`

## Nutzen

Hilft, langsame Baseline-Schwankungen zu entfernen.  
Wichtig, wenn der MSE über den Tag hinweg driftet.

---

# 8. Rolling MAD als robuste lokale Streuung

## Formel

\[
MAD_t =
median(|x_{\text{smooth},i} - m_t|)
\quad
\text{für}
\quad
i \in [t-W, t]
\]

## Robuste Skalierung

\[
\sigma_{\text{robust},t}
=
1.4826 \cdot MAD_t
\]

## Bedeutung

Robuste Schätzung der lokalen Signalstreuung.

## Wählbare Parameter

- `W_baseline`
- `epsilon`

## Typische Werte

- `W_baseline = 30 min` bis `120 min`
- `epsilon = 1e-12` oder abhängig von der Score-Skala

## Nutzen

Erlaubt eine robuste Normalisierung des Anomaly Scores.

---

# 9. Robuster normalisierter Score / z-Score

## Formel

\[
z_t =
\frac{x_{\text{smooth},t} - m_t}
{1.4826 \cdot MAD_t + \epsilon}
\]

## Bedeutung

Gibt an, wie stark der aktuelle Score relativ zur lokalen Baseline erhöht ist.

## Wählbare Parameter

- `W_baseline`
- `epsilon`
- Glättung `alpha` des Eingangssignals

## Typische Werte

- `W_baseline = 60 min`
- `alpha = 0.2`
- `epsilon = 1e-12`

## Nutzen

Macht verschiedene Zeitbereiche besser vergleichbar.  
Hilfreich für Schwellwerte, CUSUM und Rolling Area.

---

# 10. Positive Überschreitung oberhalb einer Baseline

## Formel auf z-Score

\[
u_t = \max(0, z_t - T_{\text{base}})
\]

## Formel auf geglättetem Score

\[
u_t = \max(0, x_{\text{smooth},t} - baseline_t)
\]

## Bedeutung

Berücksichtigt nur den Anteil des Signals, der oberhalb einer gewählten Baseline liegt.

## Wählbare Parameter

\[
T_{\text{base}}
\]

## Typische Werte bei z-Score

\[
T_{\text{base}} = 0
\]

\[
T_{\text{base}} = 1
\]

\[
T_{\text{base}} = 2
\]

\[
T_{\text{base}} = 3
\]

## Nutzen

Grundbaustein für Rolling Area und Evidenzakkumulation.

---

# 11. Rolling Area / lokale Energie

## Formel

\[
A_{W,t}
=
\sum_{i=t-W}^{t}
\max(0, z_i - T_{\text{base}})
\]

## Bedeutung

Misst, wie viel Anomalie-Evidenz sich im letzten Fenster angesammelt hat.

## Wählbare Parameter

- `W_area`
- `T_base`
- Signalbasis: `z_t` oder `x_smooth,t`

## Typische Werte

Fenster:

- `W_area = 1 min`
- `W_area = 3 min`
- `W_area = 5 min`

Threshold bei z-Score:

- `T_base = 1`
- `T_base = 2`
- `T_base = 3`

## Nutzen

Sehr wichtig zur Unterscheidung:

Kleiner Hügel:

- wenig akkumulierte Fläche

Große Anomalie:

- Fläche wächst kontinuierlich

---

# 12. Rolling Mean / Rolling Average des Scores

## Formel

\[
\mu_{W,t}
=
mean(x_{\text{smooth},i})
\quad
\text{für}
\quad
i \in [t-W, t]
\]

## Bedeutung

Lokaler Mittelwert des Scores im vergangenen Fenster.

## Wählbare Parameter

\[
W_{\text{mean}}
\]

## Typische Werte

- `1 min`
- `3 min`
- `5 min`

## Nutzen

Zeigt, ob das gesamte lokale Niveau steigt, nicht nur ein Einzelpunkt.

---

# 13. Rolling Maximum

## Formel

\[
max_{W,t}
=
max(x_{\text{smooth},i})
\quad
\text{für}
\quad
i \in [t-W, t]
\]

## Bedeutung

Höchster Score im letzten Fenster.

## Wählbare Parameter

\[
W_{\text{max}}
\]

## Typische Werte

- `3 min`
- `5 min`
- `10 min`

## Nutzen

Wichtig für die Drawdown-Berechnung.  
Zeigt, ob gerade neue lokale Maxima entstehen.

---

# 14. Drawdown vom lokalen Maximum

## Absolute Formel

\[
drawdown_{\text{abs},t}
=
max_{W,t} - x_{\text{smooth},t}
\]

## Relative Formel

\[
drawdown_{\text{rel},t}
=
\frac{max_{W,t} - x_{\text{smooth},t}}
{max_{W,t} + \epsilon}
\]

## Alternative als Peak Ratio

\[
peak\_ratio_t =
\frac{x_{\text{smooth},t}}
{max_{W,t} + \epsilon}
\]

## Bedeutung

Misst, wie stark das Signal vom letzten lokalen Maximum zurückgefallen ist.

## Wählbare Parameter

- `W_max`
- `epsilon`
- absolute oder relative Variante

## Typische Werte

- `W_max = 3 min`
- `W_max = 5 min`
- `W_max = 10 min`

## Nutzen

Sehr nützlich gegen kleine Hügel.

Kleiner Hügel:

- steigt kurz
- erreicht Maximum
- fällt schnell zurück
- Drawdown steigt früh

Große Anomalie:

- fällt während der Eskalation nicht stark zurück
- erzeugt neue lokale Maxima
- Drawdown bleibt zunächst kleiner

---

# 15. Anzahl positiver Steigungen im Fenster

## Formel

\[
N^+_{W,t}
=
\sum_{i=t-W}^{t}
\mathbf{1}(d_i > 0)
\]

## Erweiterte Formel mit Steigungsschwelle

\[
N^+_{W,t}
=
\sum_{i=t-W}^{t}
\mathbf{1}(d_i > T_{\text{slope}})
\]

## Bedeutung

Zählt, wie viele der letzten Punkte eine positive Steigung hatten.

## Wählbare Parameter

- `W_pos`
- `T_slope`

## Typische Werte

Fenster:

- `1 min`
- `3 min`
- `5 min`

Steigungsschwelle:

- `T_slope = 0`
- oder kleiner positiver Wert abhängig von der Score-Skala

## Nutzen

Ein echter Anstieg hat oft viele positive Steigungen hintereinander.  
Ein kleiner Hügel hat oft nur kurz positive Steigung.

---

# 16. Anteil positiver Steigungen im Fenster

## Formel

\[
p^+_{W,t}
=
\frac{N^+_{W,t}}{N_W}
\]

mit:

\[
N_W
\]

als Anzahl der Samples im Fenster `W`.

## Bedeutung

Normierte Version der positiven Steigungsanzahl.

## Wählbare Parameter

- `W_pos`
- `T_slope`

## Typische Werte

- `W_pos = 1 min` bis `5 min`
- `T_slope = 0` oder kleiner positiver Wert

## Nutzen

Besser vergleichbar bei unterschiedlich vielen Samples.  
Werte nahe `1` bedeuten, dass fast alle letzten Punkte gestiegen sind.

---

# 17. Länge der aktuellen Rising Streak

## Formel

Falls:

\[
d_t > T_{\text{slope}}
\]

dann:

\[
streak_t = streak_{t-1} + 1
\]

sonst:

\[
streak_t = 0
\]

## Bedeutung

Zählt, wie viele Punkte hintereinander das Signal gestiegen ist.

## Wählbare Parameter

- `T_slope`
- Signalbasis: `raw`, `smoothed` oder `z-score`

## Typische Werte

- `T_slope = 0`
- oder leicht positiver Wert, um Rauschen zu ignorieren

## Nutzen

Sehr intuitives Online-Merkmal.

Kleine Hügel:

- kurze Rising Streaks

Große Anomalien:

- häufig längere Rising Streaks

---

# 18. CUSUM / Evidenz-Akkumulator

## Formel

\[
g_t =
\max(0, g_{t-1} + z_t - k)
\]

## Schwelle

\[
g_t > h
\]

## Bedeutung

Akkumuliert positive Abweichungen vom Normalzustand.

## Wählbare Parameter

- `k`
- `h`
- Signalbasis: meistens `z_t`
- Reset-Verhalten

## Typische Werte

Drift-/Ignoranzschwelle:

- `k = 0.5`
- `k = 1.0`
- `k = 2.0`

Alarmschwelle:

- `h = 5`
- `h = 8`
- `h = 10`
- `h = 15`

## Interpretation

Kleines `k`:

- empfindlicher
- schnellere Reaktion
- mehr False Positives

Großes `k`:

- robuster
- langsamere Reaktion

Kleines `h`:

- früher Alarm
- mehr False Positives

Großes `h`:

- späterer Alarm
- weniger False Positives

## Nutzen

Sehr relevant für die Fragestellung.

Kleiner Hügel:

- CUSUM steigt kurz und fällt/resetet wieder

Große Anomalie:

- CUSUM steigt weiter an und überschreitet Schwellen

---

# 19. Page-Hinkley-artiger Akkumulator

## Einfache Formel

\[
PH_t =
\max(0, PH_{t-1} + z_t - \bar{z}_t - \delta)
\]

## Schwelle

\[
PH_t > \lambda
\]

## Bedeutung

Detektiert anhaltende Mittelwertverschiebungen.

## Wählbare Parameter

- `delta`
- `lambda`
- Fenster oder Update-Regel für \(\bar{z}_t\)

## Typische Werte

- `delta = 0.1` bis `1.0`
- `lambda = 5` bis `20`

## Nutzen

Gut für Online Change Detection.  
Kann robuster sein als einfache Schwellen auf Einzelpunkten.

---

# 20. Kurzfristige Evidenz mit positiver und negativer Evidenz

## Allgemeine Formel

\[
E_t =
\max(0, E_{t-1} + P_t - N_t)
\]

## Positive Evidenz

\[
P_t =
w_1 \max(0, z_t - T_z)
+
w_2 \max(0, d_t - T_d)
+
w_3 \mathbf{1}(d_t > T_d)
\]

## Negative Evidenz

\[
N_t =
v_1 \max(0, -d_t)
+
v_2 \mathbf{1}(z_t < T_z)
+
v_3 drawdown_{\text{rel},t}
\]

## Bedeutung

Ein allgemeiner Online-Evidenzscore, der steigt, wenn sich das Ereignis weiter aufbaut, und fällt, wenn der Verlauf zurückgeht.

## Wählbare Parameter

- `T_z`
- `T_d`
- `w1`, `w2`, `w3`
- `v1`, `v2`, `v3`
- minimaler/maximaler Score
- Warn- und Alarmschwellen

## Typische Zustände

\[
E_t < H_{\text{low}}
\]

Status: `normal`

\[
E_t \geq H_{\text{low}}
\]

Status: `early warning`

\[
E_t \geq H_{\text{high}}
\]

Status: `confirmed anomaly`

## Nutzen

Diese Größe ist besonders passend, wenn später eine State Machine aufgebaut werden soll.

---

# 21. Verhältnis aus aktueller Steigung und aktueller Höhe

## Formel

\[
r_{\text{slope},t}
=
\frac{d_t}
{z_t + \epsilon}
\]

oder:

\[
r_{\text{slope},t}
=
\frac{d_{\text{smooth},t}}
{z_t + \epsilon}
\]

## Bedeutung

Misst, ob das Signal relativ zu seiner aktuellen Höhe noch stark weiterwächst.

## Wählbare Parameter

- `epsilon`
- Signalbasis
- Glättung

## Nutzen

Kann helfen zu sehen, ob ein Hügel noch weiter eskaliert oder bereits ausläuft.

## Hinweis

Diese Größe ist empfindlich bei kleinen `z_t`-Werten.

---

# 22. Verhältnis kurzfristige zu langfristige Energie

## Formel

\[
R_{\text{energy},t}
=
\frac{A_{\text{short},t}}
{A_{\text{long},t} + \epsilon}
\]

mit:

\[
A_{\text{short},t}
\]

als Rolling Area über ein kurzes Fenster.

\[
A_{\text{long},t}
\]

als Rolling Area über ein längeres Fenster.

## Bedeutung

Vergleicht kurzfristige Aktivität mit längerfristiger Aktivität.

## Wählbare Parameter

- `W_short`
- `W_long`
- `T_base`
- `epsilon`

## Typische Werte

- `W_short = 1 min`
- `W_long = 5 min`
- `W_long = 10 min`

## Nutzen

Kann zeigen, ob gerade ein neuer schneller Anstieg beginnt.

---

# 23. Rolling Standard Deviation / lokale Volatilität

## Formel

\[
std_{W,t}
=
std(x_{\text{smooth},i})
\quad
\text{für}
\quad
i \in [t-W, t]
\]

## Bedeutung

Misst die lokale Schwankungsstärke.

## Wählbare Parameter

\[
W_{\text{std}}
\]

## Typische Werte

- `3 min`
- `5 min`
- `10 min`

## Nutzen

Kann zeigen, ob ein Bereich nur verrauscht ist oder strukturiert ansteigt.

## Hinweis

Für eine robuste Analyse ist MAD meist besser als Standardabweichung.

---

# 24. Rolling Coefficient of Variation

## Formel

\[
CV_{W,t}
=
\frac{std_{W,t}}
{mean_{W,t} + \epsilon}
\]

## Bedeutung

Normierte Schwankungsstärke.

## Wählbare Parameter

- `W_cv`
- `epsilon`

## Nutzen

Kann hilfreich sein, wenn sich der absolute Score-Level stark verändert.

## Hinweis

Bei sehr kleinen Mittelwerten vorsichtig interpretieren.

---

# 25. Time since Onset / Zeit seit erstem Kandidatenanstieg

## Onset-Bedingung

\[
z_t > T_{\text{onset}}
\]

und

\[
d_t > T_{\text{slope}}
\]

## Zeit seit Onset

\[
\tau_t = t - t_{\text{onset}}
\]

## Bedeutung

Misst, wie lange ein potenzielles Ereignis bereits läuft.

## Wählbare Parameter

- `T_onset`
- `T_slope`
- Reset-Bedingung
- maximale Kandidatendauer

## Nutzen

Wichtig für die Frage:

Nach wie vielen Sekunden oder Minuten nach Onset kann man kleine Hügel und große Anomalien unterscheiden?

---

# 26. Kandidatenstatus / State Machine als Visualisierung

## Mögliche Zustände

- `NORMAL`
- `EARLY_RISE`
- `LIKELY_ANOMALY`
- `CONFIRMED_ANOMALY`
- `RECOVERING`

## Beispielhafte Kriterien

### NORMAL

Kein auffälliger Score.

### EARLY_RISE

\[
z_t > T_{\text{low}}
\]

und

\[
d_t > T_{\text{slope}}
\]

### LIKELY_ANOMALY

\[
CUSUM_t > H_{\text{low}}
\]

oder

\[
A_{W,t} > A_{\text{low}}
\]

### CONFIRMED_ANOMALY

\[
CUSUM_t > H_{\text{high}}
\]

und Persistenzbedingung erfüllt.

### RECOVERING

Score fällt unter Ausschaltgrenze oder Drawdown steigt stark.

## Wählbare Parameter

- `T_low`
- `T_slope`
- `H_low`
- `H_high`
- `A_low`
- `T_off`
- Persistenzfenster
- Reset-Kriterien

## Nutzen

Hilft, die visuelle Analyse direkt in eine spätere Online-Detektionslogik zu überführen.

---

# Empfohlene erste Multi-Panel-Plots

Für die erste visuelle Analyse würde ich folgende Plots erzeugen:

1. Raw anomaly score und EWMA
2. Erste Ableitung / geglättete Ableitung
3. Rolling Slope, z. B. `1 min` und `3 min`
4. Robuster z-Score
5. Rolling Area, z. B. `1 min` und `3 min`
6. Drawdown vom Rolling Maximum
7. Positive Slope Fraction
8. CUSUM
9. Optional: Evidenzscore oder State Machine

---

# Wichtigste Empfehlung für die konkrete Fragestellung

Für die frühe Unterscheidung kleiner Hügel von großen Anomalien sollten zuerst diese Größen visuell untersucht werden:

1. EWMA Score
2. geglättete erste Ableitung
3. Rolling Slope über `1 min` bis `3 min`
4. Rolling Area über `1 min` bis `3 min`
5. Drawdown vom lokalen Maximum
6. CUSUM auf robust normalisiertem z-Score
7. Positive Slope Fraction

Diese Features sind kausal berechenbar und zeigen nicht die finale Größe des Ereignisses, sondern den bis zum aktuellen Zeitpunkt aufgebauten Verlauf. Genau das ist für eine Punkt-für-Punkt-Entscheidung entscheidend.