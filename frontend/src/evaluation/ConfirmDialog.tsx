import { Button, Group, Modal, Stack, Text } from '@mantine/core';
import type { ReactNode } from 'react';

export type ConfirmAction = {
  label: string;
  color?: string;
  variant?: string;
  onClick: () => void;
};

/** Small confirmation modal for the evaluation workspace: one or more explicit
 *  choices plus Cancel, so an unsaved layout is never dropped silently. */
export function ConfirmDialog({ opened, title, message, actions, confirmLoading, onCancel }: {
  opened: boolean;
  title: string;
  message: ReactNode;
  actions: ConfirmAction[];
  confirmLoading?: boolean;
  onCancel: () => void;
}) {
  return (
    <Modal opened={opened} onClose={onCancel} title={title} centered>
      <Stack gap="md">
        {typeof message === 'string' ? <Text size="sm">{message}</Text> : message}
        <Group justify="flex-end" wrap="wrap">
          <Button variant="subtle" color="gray" onClick={onCancel}>Cancel</Button>
          {actions.map((action) => (
            <Button
              key={action.label}
              color={action.color}
              variant={action.variant}
              loading={confirmLoading}
              onClick={action.onClick}
            >
              {action.label}
            </Button>
          ))}
        </Group>
      </Stack>
    </Modal>
  );
}
