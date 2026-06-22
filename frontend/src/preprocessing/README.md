# Preprocessing Frontend Extension Guide

The preprocessing UI is schema-driven. Most backend steps appear in the UI
without frontend code: the page reads `GET /api/preprocessing/steps`, renders
fields from each step's `config_schema`, and stores the resulting config in the
pipeline graph.

## Backend-Only Steps

If a step only needs normal scalar fields, add it in the backend and define a
`config_schema` with supported field types:

- `integer` / `number`: rendered as number inputs.
- `string` with `enum`: rendered as a select.
- `string` without `enum`: rendered as a text input.
- `boolean`: rendered as a switch.

Useful schema keys:

- `label`: displayed field name.
- `default`: value used when the user has not configured the field.
- `minimum` / `maximum`: numeric bounds.
- `description`: hover/help copy when rendered by shared schema forms.

No frontend file needs to change for this kind of step.

## Interactive Controls

Some steps need image-aware editing, for example dragging warp points. These use
`config_schema.ui_control`.

Existing controls live under `frontend/src/preprocessing/controls/` and are
registered in `controls/index.ts`:

- `point_picker`: edits `source_points`.
- `crop_box`: edits `x`, `y`, `width`, `height`.

To add a new control:

1. Create a React component that implements `StepControlProps`.
2. Export it as a `StepControl` with an `ownedKeys` list.
3. Register it in `CONTROL_REGISTRY`.
4. Set the same `ui_control` string in the backend step's `config_schema`.

`ownedKeys` tells the generic form which config fields are managed by the
interactive control, so those raw fields are hidden from the normal field grid.

## Control Contract

Controls receive:

- `inputImage`: preview image before the current step runs.
- `config`: current node config.
- `disabled`: read-only state.
- `onChange(partial)`: merge patch for the node config.

Controls should only write their own `ownedKeys`. They should not call backend
APIs directly. The page automatically reruns preview after config changes.

## Preview, Save, Training

Preview executes the full saved graph through the backend and displays every
intermediate output. Saving stores only graph JSON and design sizes, not image
artifacts. Training/testing later use the same saved graph, but the backend
compiles it before repeated execution so the UI format remains flexible while
the hot path stays efficient.

Preview PNGs use an absolute dtype scale. A `uint16` value therefore has the
same displayed brightness in every image; the browser preview no longer expands
each image's own minimum and maximum to black and white. Raw min/max metadata is
still shown alongside the preview.

Schema properties may declare `visible_when`, for example:

```python
"output_width": {
    "type": "integer",
    "visible_when": {"output_shape_mode": "manual"},
}
```

The generic field renderer only shows the property while all declared config
dependencies match. This keeps mode-specific fields out of page-level code.

## When Frontend Code Is Needed

Backend-only is enough for normal numeric/text/select/switch config fields.
Frontend code is only needed when the user must interact directly with an image
or another custom visual editor.
