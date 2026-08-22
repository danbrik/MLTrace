import { Alert, Button, Group, Select, Stack, Text } from '@mantine/core';
import { ArrowLeft, FilePlus2, FolderOpen } from 'lucide-react';
import { useCallback, useState } from 'react';

import { ConfirmDialog } from './ConfirmDialog';

export type LayoutStage = 'choose' | 'pick' | 'edit';

export type LayoutOption = { id: number; name: string; version: number; calculation_count: number };

/**
 * Guards a destructive navigation (creating a new layout, switching to another one)
 * behind an explicit choice whenever the visible layout has unsaved edits.
 * `save` reports success so a rejected save does not discard the edits.
 */
export function useUnsavedGuard(dirty: boolean, save: () => Promise<boolean>, note?: string) {
  const [pending, setPending] = useState<(() => void) | null>(null);
  const [saving, setSaving] = useState(false);

  const guard = useCallback((action: () => void) => {
    if (!dirty) { action(); return; }
    setPending(() => action);
  }, [dirty]);

  const dialog = (
    <ConfirmDialog
      opened={pending !== null}
      title="Unsaved layout"
      message={`The visible layout has changes that are not saved yet. Save them before continuing, or discard them.${note ? ` ${note}` : ''}`}
      confirmLoading={saving}
      onCancel={() => setPending(null)}
      actions={[
        { label: 'Discard and continue', color: 'red', variant: 'subtle', onClick: () => { const action = pending; setPending(null); action?.(); } },
        {
          label: 'Save and continue',
          onClick: () => {
            const action = pending;
            setSaving(true);
            void save().then((ok) => { setSaving(false); if (ok) { setPending(null); action?.(); } });
          },
        },
      ]}
    />
  );

  return { guard, dialog };
}

/**
 * Entry point for layout work, shared by Event Separation and Score Stability:
 * pick an existing layout or start a new one, and switch between them later.
 */
export function LayoutChooser({ stage, layouts, layoutId, kind, disabled, onStage, onPick, onCreate }: {
  stage: LayoutStage;
  layouts: LayoutOption[];
  layoutId: number | null;
  kind: string;
  disabled?: boolean;
  onStage: (stage: LayoutStage) => void;
  onPick: (id: number) => void;
  onCreate: () => void;
}) {
  const options = layouts.map((item) => ({ value: String(item.id), label: `${item.name} · v${item.version}` }));

  if (stage === 'choose') {
    return (
      <Stack gap="sm">
        <Text size="sm" c="dimmed">Start from a saved {kind} layout or define a new one.</Text>
        <Group>
          <Button
            variant="light"
            leftSection={<FolderOpen size={16} />}
            disabled={disabled || layouts.length === 0}
            onClick={() => onStage('pick')}
          >
            Load layout
          </Button>
          <Button leftSection={<FilePlus2 size={16} />} disabled={disabled} onClick={onCreate}>
            Create layout
          </Button>
        </Group>
        {!disabled && layouts.length === 0 && (
          <Alert color="gray">No saved {kind} layout exists for this inference dataset yet.</Alert>
        )}
      </Stack>
    );
  }

  if (stage === 'pick') {
    return (
      <Stack gap="sm">
        <Select
          label="Saved layout"
          placeholder={`Choose a saved ${kind} layout`}
          data={options}
          value={layoutId ? String(layoutId) : null}
          searchable
          onChange={(value) => value && onPick(Number(value))}
        />
        <Group>
          <Button variant="subtle" color="gray" leftSection={<ArrowLeft size={16} />} onClick={() => onStage('choose')}>
            Back
          </Button>
        </Group>
      </Stack>
    );
  }

  return (
    <Group align="end" wrap="wrap" gap="sm">
      <Select
        label="Saved layout"
        placeholder="Unsaved layout"
        data={options}
        value={layoutId ? String(layoutId) : null}
        searchable
        style={{ flex: 1, minWidth: 240 }}
        onChange={(value) => value && onPick(Number(value))}
      />
      <Button variant="light" leftSection={<FilePlus2 size={16} />} onClick={onCreate}>
        Create layout
      </Button>
    </Group>
  );
}
