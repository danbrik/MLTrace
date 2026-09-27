# Projektweite Zeitraumvorlagen

„Gespeicherte Zeiträume“ ist eine gemeinsame Bibliothek pro Projekt. Die erste
Einbindung befindet sich in der Referenzbild-Analyse bei beiden Zeiträumen.
Auch der Mittelwert-/Varianzvergleich verwendet dieselbe Bibliothek für Normal-
und Anomaliephase.
Übernehmen kopiert ausschließlich Beginn und Ende. Sampling, Zufallsseed,
Datensatz und Preprocessing bleiben erhalten. Überlappende Zeiträume sowie
Beginn gleich Ende sind erlaubt.

„Zeitraum speichern“ öffnet einen Dialog mit den aktuellen Zeiten und einem
Namensfeld. „Zeiträume verwalten“ bietet Suche, Umbenennen, Ändern der Zeiten
und Löschen mit Bestätigung. Änderungen werden in allen eingebundenen
Auswahlfeldern desselben Projekts unmittelbar sichtbar. Im Data Manager steht
der Objekttyp „Gespeicherte Zeiträume“ zur Verfügung.

## Speicherung und API

Die Migration `0062_time_range_presets` folgt auf `0061_reference_image`.
`time_range_presets` liegt in der jeweiligen Projektdatenbank. Es gibt keine
Verknüpfung zu Datensätzen, Pipelines oder Analyse-Läufen. Der interne eindeutige
`name_key` ist der mit Python `casefold()` normalisierte Name; dadurch werden
auch Unicode-Groß-/Kleinschreibungsvarianten erkannt. Namen werden getrimmt,
sind 1–255 Zeichen lang und dürfen pro Projekt nur einmal vorkommen. Identische
Zeitwerte unter anderen Namen sind erlaubt.

Alle Anfragen verwenden `X-MLTrace-Project-ID` (die bestehende Middleware
unterstützt alternativ `project_id`).

| Methode | Pfad | Ergebnis |
| --- | --- | --- |
| GET | `/api/time-range-presets` | Liste, nach Namen sortiert |
| POST | `/api/time-range-presets` | Anlegen, 201 |
| PUT | `/api/time-range-presets/{id}` | Name und Zeiten vollständig ersetzen, 200 |
| DELETE | `/api/time-range-presets/{id}` | Löschen, 204 |

POST/PUT erwarten `{ "name": "Normalphase", "start": "2026-01-01T10:00:00", "end": "2026-01-01T11:00:00" }`.
Antwortobjekte enthalten `id`, `name`, `start`, `end`, `created_at`, `updated_at`.
Ungültige Eingaben liefern 422, Namenskonflikte 409 und fehlende IDs 404.

Beginn und Ende folgen der lokalen Bilddatensatz-Zeitkonvention. Es erfolgt
keine Zeitzonenumrechnung; ein mitgelieferter Offset wird wie bei den bestehenden
Datensatzformularen verworfen. Beispiel: `10:00:00+02:00` bleibt `10:00:00`.
Die Erstellungs-/Änderungsmetadaten verwenden die bestehende UTC-Konvention.

## Einbindung in weitere Reiter

Das eigenständige Modul `frontend/src/timeRangePresets` enthält Typen,
Validierung, einen gemeinsamen Store, den Hook `useTimeRangePresets` und die
Komponente `TimeRangePresetPicker`. HTTP-Funktionen stehen in `src/api.ts`.
Jeder HTTP-Aufruf erhält eine explizite Projekt-ID, damit ein noch laufender
Aufruf beim Projektwechsel immer an sein ursprüngliches Projekt gebunden bleibt.

```tsx
import { TimeRangePresetPicker } from '../timeRangePresets/TimeRangePresetPicker';
import { copyTimeRange } from '../timeRangePresets/helpers';

<TimeRangePresetPicker
  projectId={projectId}
  active={active}
  value={form.period}
  min={dataset?.start_timestamp ?? undefined}
  max={dataset?.end_timestamp ?? undefined}
  disabled={busy}
  applyDisabledReason={!dataset ? 'Bitte zuerst einen Datensatz auswählen.' : undefined}
  onApply={range => {
    setPreview(null);
    setForm(current => ({
      ...current,
      period: copyTimeRange(current.period, range),
    }));
  }}
/>
```

`value` und `onApply` verwenden `{ start: string; end: string }`. `min` und `max`
sind optionale einschließlich gültige Datensatzgrenzen. Unpassende Vorlagen
werden mit Erklärung deaktiviert und nie beschnitten. `applyDisabledReason`
sperrt nur das Übernehmen, etwa bis ein Datensatz gewählt wurde. `disabled`
sperrt alle Aktionen während einer Berechnung. `active` verhindert unnötiges
Laden bei ausgeblendeten Reitern und lädt beim erneuten Öffnen die aktuelle
Liste, auch nach Änderungen im Data Manager.

Der aufrufende Reiter invalidiert seine Auswahlvorschau sowohl bei `onApply`
(auch bei denselben Zeitwerten) als auch bei manuellen Änderungen. Ob im
Zeitraum tatsächlich Bilder vorhanden sind, prüft weiterhin die bestehende
Analyse-Auswahlprüfung. Die Komponente enthält keine Referenz-/Anomalielogik.

Alle Auswahlfelder verwenden denselben nach Projekt-ID getrennten Store.
Parallele Leseanfragen werden zusammengefasst. Eine veraltete Ladeantwort darf
keine zwischenzeitliche Änderung oder Löschung überschreiben. Ein Projektwechsel
setzt Dialoge zurück und zeigt ausschließlich Einträge des neuen Projekts.
Bearbeiten/Löschen ruft `onApply` niemals auf: Übernommene Formularwerte und
bereits gespeicherte Analyse-Konfigurationen bleiben unverändert.

## Prüfung

Backend: `backend/tests/test_time_range_presets.py` prüft CRUD, Migration und
Neuverbindung, Namensvalidierung inklusive Unicode, lokale Zeiten,
Projekttrennung, Data Manager und unveränderte Analyse-Läufe.
Frontend: `src/timeRangePresets/*.test.ts` prüft API-Routing, Zeitgrenzen,
Kopieren ohne andere Einstellungen, synchrone Listen, Ladefehler und
Projektwechsel einschließlich verspäteter Antworten.
