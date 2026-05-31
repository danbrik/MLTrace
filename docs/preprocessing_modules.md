# Preprocessing-Module erstellen und anbinden

Stand: 2026-05-31

Diese Anleitung beschreibt, wie ein neues Preprocessing-Modul ("Step") in MLTrace korrekt
erstellt und angebunden wird. Sie richtet sich an Entwickler und an KI-Modelle, die den Code
erweitern. Prosa ist deutsch, alle Code-Bezeichner und UI-Strings bleiben englisch (siehe
[project_doc.md](project_doc.md): "UI-Sprache: Englisch").

## Architektur in Kürze

- **Basisklasse:** [`BasePreprocessingStep`](../backend/app/preprocessing/base.py) definiert den
  Standard-Vertrag: Klassen-Attribute (`type`, `label`, `category`, `input_kind`, `output_kind`,
  `default_config`, `config_schema`) und die Methode `apply(image, config, context) -> np.ndarray`.
- **Auto-Discovery:** [`steps/__init__.py`](../backend/app/preprocessing/steps/__init__.py) scannt
  beim Import alle Module im Paket `steps/`, findet jede nicht-abstrakte `BasePreprocessingStep`-
  Subklasse und registriert sie automatisch. **Es gibt keine zentrale Liste, die gepflegt werden
  muss** – die Datei genügt.
- **Registry:** [`registry.py`](../backend/app/preprocessing/registry.py) hält alle Steps und
  liefert sie über `/api/preprocessing/steps` ans Frontend (inkl. `config_schema`).
- **Pipeline:** Eine Pipeline ist ein linearer Graph (`load_image` zuerst, dann eine Kette).
  [`execute_preview`](../backend/app/preprocessing/pipeline.py) führt die Steps der Reihe nach aus
  und reicht das Ausgabebild jedes Steps als Eingabe des nächsten weiter (Chaining). Pro Node
  entsteht ein Vorschaubild.
- **Frontend:** [`PreprocessingPipelinesPage.tsx`](../frontend/src/pages/PreprocessingPipelinesPage.tsx)
  baut die Konfig-Oberfläche **generisch** aus dem `config_schema` und zeigt pro Step automatisch
  ein Input→Output-Vorschaufenster.

## Der `apply`-Vertrag

```python
def apply(self, image: np.ndarray | None, config: dict, context: dict) -> np.ndarray:
    ...
```

- `image`: das Eingabebild (Ausgabe des Vorgängerschritts). Nur bei `load_image` ist es `None`;
  alle anderen Steps sollen bei `None` eine `ValueError` werfen.
- `config`: die Parameter dieses Steps. Über `self.merged_config(config)` werden fehlende Werte
  mit `default_config` aufgefüllt.
- `context`: enthält u.a. `source_image_path` (vom `load_image`-Step genutzt) und `source_shape`
  (Form des ursprünglich geladenen Bildes, gesetzt nach dem ersten Step).
- Rückgabe: ein `np.ndarray` (das nächste Bild in der Pipeline).

## Typ-Kette (`output_spec`)

Vor dem Ausführen prüft `validate_linear_graph` die Pipeline **symbolisch**: ein `ImageSpec`
(`channels`, `width`, `height`, `dtype`; Größen dürfen `None` sein, bis ein echtes Bild geladen ist)
wird von `load_image` durch alle Steps gefädelt. Jeder Step deklariert das über
`output_spec(spec_in, config) -> ImageSpec` ([base.py](../backend/app/preprocessing/base.py)):

```python
def output_spec(self, spec_in: ImageSpec | None, config: dict) -> ImageSpec:
    if spec_in is None:
        raise ValueError(f"{self.type} requires an input image.")
    return spec_in   # Default: Bild erforderlich, Kanäle/Größe unverändert
```

So macht man Anforderungen prüfbar: ein Step, der Farbe braucht, wirft bei `spec_in.channels != 3`;
ein größenändernder Step (resize/crop/warp) setzt `width`/`height` im zurückgegebenen Spec.
Inkompatible Ketten werden **hart blockiert** (beim Speichern und in der Vorschau). Den Default
muss man nur überschreiben, wenn der Step Kanäle/Größe ändert oder Eingaben einschränkt.

Zusätzlich prüft `validate_step_config` jeden Wert gegen das `config_schema`
(`type`, `minimum`/`maximum`, `enum`) und wirft bei Verstößen einen klaren Fehler.

## Rezept A – Einfacher Step (nur Parameter)

Für Steps ohne interaktives Bild-Werkzeug (z.B. Filter, Farbumwandlungen) ist **nur eine
Backend-Datei** nötig. Vorschaufenster und Eingabefelder entstehen automatisch.

### Schritt 1: Datei anlegen

`backend/app/preprocessing/steps/gaussian_blur.py`:

```python
from __future__ import annotations

import cv2
import numpy as np

from app.preprocessing.base import BasePreprocessingStep


class GaussianBlurStep(BasePreprocessingStep):
    type = "gaussian_blur"            # eindeutiger Schlüssel
    label = "Gaussian blur"           # Anzeigename in der UI
    category = "Filters"              # Gruppierung in der Palette
    input_kind = "image ndarray"
    output_kind = "blurred image ndarray"
    default_config = {"kernel_size": 5, "sigma": 0.0}
    config_schema = {
        "type": "object",
        "properties": {
            "kernel_size": {"type": "integer", "label": "Kernel size (odd)", "minimum": 1, "default": 5},
            "sigma": {"type": "number", "label": "Sigma (0 = auto)", "minimum": 0, "default": 0.0},
        },
    }

    def apply(self, image: np.ndarray | None, config: dict, context: dict) -> np.ndarray:
        if image is None:
            raise ValueError("gaussian_blur requires an input image.")
        cfg = self.merged_config(config)
        kernel = max(1, int(cfg["kernel_size"]))
        if kernel % 2 == 0:               # OpenCV verlangt ungerade Kernelgröße
            kernel += 1
        sigma = float(cfg["sigma"])       # 0 => OpenCV leitet Sigma aus der Kernelgröße ab
        return cv2.GaussianBlur(image, (kernel, kernel), sigmaX=sigma)
```

Das ist alles. Kein Eintrag in `__init__.py`, keine Änderung an `schemas.py`/`main.py`/`api.ts`,
**keine** DB-Migration (Pipelines werden als JSON-Graph gespeichert; ein neuer `type` funktioniert
sofort).

### Schritt 2: `config_schema` → Frontend-Eingaben

Das Frontend baut die Eingabefelder aus `config_schema.properties`:

| Schema-Property | gerendertes Control |
| --- | --- |
| `"enum": [...]` | `Select` (Dropdown) |
| `"type": "integer"` / `"number"` (`minimum`/`maximum` optional) | `NumberInput` (mit Grenzen) |
| sonst | `TextInput` |

Optionale Property-Hints:
- `"maximum": N` — Obergrenze (zusätzlich zu `"minimum"`), server- und clientseitig geprüft.
- `"default_from": "input_width" | "input_height"` — beim Hinzufügen des Steps wird der Wert aus
  der tatsächlichen Pixelgröße des Vorgängerschritts vorbelegt (statt aus `default`). So „übernimmt"
  ein neuer crop/resize/warp die aktuelle Bildgröße. Beispiel:
  `"width": {"type": "integer", "minimum": 1, "default": 128, "default_from": "input_width"}`.

Beispiel für wählbare Stärke-Presets statt Zahlen:

```python
default_config = {"strength": "medium"}
config_schema = {
    "type": "object",
    "properties": {
        "strength": {"type": "string", "label": "Strength", "enum": ["light", "medium", "strong"], "default": "medium"},
    },
}
# in apply(): kernel = {"light": 3, "medium": 7, "strong": 15}[cfg["strength"]]
```

### Schritt 3: Gratis dazu

- Der Step erscheint automatisch in der **Step-Palette** (gruppiert nach `category`).
- Im Konfig-Block gibt es automatisch ein **Vorschaufenster**: links das Eingabebild (Ausgabe des
  Vorgängerschritts), rechts das Ergebnis dieses Steps. Die **Auto-Vorschau** aktualisiert das
  Ergebnis (debounced), sobald Parameter geändert werden.

## Rezept B – Step mit interaktivem Picker

Manche Steps brauchen ein interaktives Werkzeug auf dem Bild (Punkte ziehen, Rechteck aufziehen).
Solche Controls liegen im Frontend unter
[`frontend/src/preprocessing/controls/`](../frontend/src/preprocessing/controls/) und werden über
das Feld `config_schema.ui_control` (auf **Wurzelebene** des Schemas) ausgewählt.

Aktuell vorhandene Controls:

| `ui_control` | Control | verwaltete Config-Keys (`ownedKeys`) |
| --- | --- | --- |
| `point_picker` | 4-Punkt-Picker (Perspektive) | `source_points` |
| `crop_box` | verschieb-/skalierbares Rechteck | `x`, `y`, `width`, `height` |

### B1 – Vorhandenes Control wiederverwenden (nur Backend)

Setze im Step `config_schema["ui_control"]` auf einen vorhandenen Wert und liefere die passenden
Config-Keys. Beispiel [crop.py](../backend/app/preprocessing/steps/crop.py):

```python
config_schema = {
    "type": "object",
    "ui_control": "crop_box",      # <- aktiviert den Crop-Picker im Frontend
    "properties": {
        "x": {"type": "integer", "label": "X", "minimum": 0, "default": 0},
        "y": {"type": "integer", "label": "Y", "minimum": 0, "default": 0},
        "width": {"type": "integer", "label": "Width", "minimum": 1, "default": 128, "default_from": "input_width"},
        "height": {"type": "integer", "label": "Height", "minimum": 1, "default": 128, "default_from": "input_height"},
        # Crop kann zusätzlich die Ausgabegröße wählen (s. crop.py):
        "output_size": {"type": "string", "enum": ["cropped", "input", "source"], "default": "cropped"},
        "interpolation": {"type": "string", "enum": ["nearest", "linear", "area", "cubic"], "default": "area"},
    },
}
```

Die vom Control verwalteten Keys (`ownedKeys`, hier `x/y/width/height`) blendet das Frontend als
rohe Eingabefelder automatisch aus – sie werden interaktiv über das Bild gesetzt; restliche Felder
(`output_size`, `interpolation`) erscheinen als normale Controls. **Kein Frontend-Edit nötig.**
Crop-Modi: `cropped` (nur croppen), `input` (zurück auf Eingangsgröße interpolieren), `source`
(zurück auf die Ursprungsgröße via `context["source_shape"]`).

### B2 – Neues Control-Typ hinzufügen (eine Komponente + ein Registry-Eintrag)

1. Neue Komponente unter `controls/`, die dem Vertrag aus
   [`controls/types.ts`](../frontend/src/preprocessing/controls/types.ts) folgt:

   ```ts
   type StepControlProps = {
     inputImage: PreprocessingPreviewImage;             // Bild vom Vorgängerschritt
     config: Record<string, unknown>;                   // aktuelle Step-Config
     onChange: (partial: Record<string, unknown>) => void; // schreibt Config-Keys zurück
   };
   ```

   Die Komponente zeichnet ihr Werkzeug auf `inputImage` und meldet Änderungen über `onChange`
   (ein Partial-Update – beliebig viele Keys gleichzeitig). Pointer-/Geometrie-Helfer liegen in
   [`controls/geometry.ts`](../frontend/src/preprocessing/controls/geometry.ts).

2. Control exportieren und in der Registry eintragen
   ([`controls/index.ts`](../frontend/src/preprocessing/controls/index.ts)):

   ```ts
   export const CONTROL_REGISTRY: Record<string, StepControl> = {
     point_picker: pointPickerControl,
     crop_box: cropBoxControl,
     my_new_control: myNewControl,   // { component, ownedKeys: [...] }
   };
   ```

3. Im Backend-Step `config_schema["ui_control"] = "my_new_control"` setzen.

`PreprocessingPipelinesPage.tsx` muss dafür **nicht** angefasst werden – die Seite wählt das
Control rein über `config_schema.ui_control` + `CONTROL_REGISTRY`.

## Tests

Tests liegen in [`backend/tests/test_preprocessing.py`](../backend/tests/test_preprocessing.py).
Für einen neuen Step bietet sich an:

- Registry-Discovery prüfen (der neue `type` ist in `registry.list_definitions()`).
- `apply()` direkt aufrufen und Shape/Verhalten prüfen (siehe `test_gaussian_blur_*`).

Ausführen:

```bash
PYTHONPATH=backend pytest backend/tests/test_preprocessing.py -q
```

## Checkliste "kein Frontend-Edit nötig?"

- Nur Zahlen-/Enum-/Text-Parameter → **ja**, reines Backend (Rezept A).
- Interaktiver Picker, aber `point_picker`/`crop_box` reicht → **ja**, nur `ui_control` setzen (B1).
- Komplett neues Bild-Werkzeug → **nein**: eine Control-Komponente + ein Registry-Eintrag (B2),
  aber weiterhin kein Eingriff in die Pipeline-Seite selbst.
