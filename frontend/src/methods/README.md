# Methods UI Extension Guide

This folder contains the extensible frontend layer for MLTrace method definitions. The backend remains the source of truth for available methods, schemas, layers, validation, and diagrams. The frontend chooses a builder UI from `builder_kind` and renders backend-provided schemas and graph data.

## Mental Model

- A saved Method is the generic UI object for neural models, statistical baselines, and future approaches such as PatchCore or Optical Flow.
- Backend method definitions expose `type`, `builder_kind`, `method_schema`, defaults, and capabilities.
- Frontend builders render editing controls for one `builder_kind`.
- Schema-only methods use the existing `form` builder and need no frontend code.
- Complex graph-like methods add a new builder component and one registry entry.

## Key Files

- `registry.ts` maps backend `builder_kind` strings to frontend builders.
- `types.ts` defines the builder contract.
- `utils.ts` contains shared formatting, payload creation, and default graph helpers.
- `schema/SchemaForm.tsx` renders backend config schemas with Mantine controls and info tooltips.
- `builders/` contains method-specific editing UIs.
- `panels/` contains reusable side panels for validation, diagrams, torch checks, and saved methods.
- `../pages/ModelsPage.tsx` orchestrates data loading, save/load/delete, and builder selection.

## Adding a Schema-Only Method

Use this when the method is fully configured through scalar fields.

1. Add the backend method definition with `builder_kind = "form"`.
2. Define `method_schema` and `default_method_config`.
3. Ensure backend validation accepts the method payload.
4. No frontend code is required unless a custom field control is needed.

The `FormMethodBuilder` will render all schema properties automatically.

## Adding a New Complex Builder

Use this when a method needs specialized graph editing, sequence controls, ROI controls, or custom previews.

1. Add a backend method definition with a stable `builder_kind`, for example `patchcore_memory_bank`.
2. Create `builders/PatchCoreBuilder.tsx`.
3. Implement the `MethodBuilderProps` contract:
   - read `method`, `modelConfig`, `modelGraph`, `layers`, and `validation`
   - update scalar config through `onConfigChange(key, value)`
   - update graph data through `onGraphChange(nextGraph)` or `onGraphChange(current => next)`
4. Add a `MethodBuilderDefinition` to `registry.ts`.
5. If the method needs an initial graph, provide `createDefaultGraph(...)`.
6. Keep save payload shape compatible with `buildMethodPayload(...)` or update that utility if the graph contract intentionally changes.

Unknown `builder_kind` values are shown as an explicit UI error, so missing registry entries are visible during development.

## Payload and Validation Flow

`ModelsPage.tsx` builds two payload variants through `buildMethodPayload(...)`:

- Save payload: includes method graph, method config, training config, and inference config.
- Diagram payload: includes only graph and method config, because static shape validation should not rerun due to training or inference changes.

The diagram payload is serialized into a stable signature and debounced. This prevents validation requests from running on every React render.

Static validation is automatic and save-blocking. Torch dummy forward is manual through `TorchCheckPanel`; it must not run from diagram validation or save.

## Sequential CNN Builders

`SequentialAutoencoderBuilder` and `SequentialVaeBuilder` share `SequentialMethodBuilder`.

- CNN-AE is strict: the final encoder output feature count must match `latent_dim`.
- CNN-VAE may use the implicit `mu/logvar -> z -> decoder seed` bridge.
- Layer definitions come from the backend layer catalog.
- Layer cards are collapsed by default to keep dense architectures readable.

## fastAnoGAN Builder

`fast_anogan` uses `FastAnoganBuilder` instead of the sequential layer builder.
It edits block sections rather than arbitrary Torch layers:

- Generator blocks are residual upsampling blocks.
- Critic blocks are residual downsampling blocks and must use `layer_norm` or `none`.
- Encoder blocks are residual downsampling blocks from image to latent vector.

Do not model the encoder as an upsampling generator path. In fastAnoGAN the
encoder direction is always image -> z. The frontend shows block direction as a
disabled field and the backend validates it again before saving.

## Code Style Rules

- Keep `ModelsPage.tsx` as orchestration only. Do not add method-specific JSX there.
- Put reusable UI in `panels/`, schema rendering in `schema/`, and builder-specific controls in `builders/`.
- Keep comments short and structural. Explain extension contracts and non-obvious data flow; avoid narrating simple JSX.
- Prefer backend-provided metadata over hardcoded frontend lists unless the UI behavior is genuinely builder-specific.
- New builders should fail visibly when required backend metadata is missing instead of silently inventing defaults.
