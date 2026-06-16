import { Badge, Group, Paper, Stack, Text, Title } from '@mantine/core';
import { Check } from 'lucide-react';
import type { ReactNode } from 'react';

/** Consistent accent colors for numbered step sequences across the app. */
export const STEP_COLORS = ['blue', 'violet', 'teal', 'grape', 'orange', 'cyan'] as const;

/**
 * A titled section card with a colored left accent and an optional numbered
 * badge — the shared building block for the guided, step-by-step page layouts.
 *
 * - `index` shows a numbered badge (omit for an un-numbered accent section).
 * - `complete` swaps the number for a check to signal a satisfied step.
 * - `action` renders right-aligned in the header (e.g. a "Change" button).
 */
export function StepCard({
  index,
  title,
  subtitle,
  color = 'blue',
  complete = false,
  action,
  children,
}: {
  index?: number;
  title: string;
  subtitle?: ReactNode;
  color?: string;
  complete?: boolean;
  action?: ReactNode;
  children: ReactNode;
}) {
  return (
    <Paper withBorder p="md" radius="sm" style={{ borderLeft: `4px solid var(--mantine-color-${color}-5)` }}>
      <Stack gap="md">
        <Group justify="space-between" align="center" wrap="nowrap">
          <Group gap="xs" wrap="nowrap">
            {index !== undefined && (
              <Badge color={complete ? 'green' : color} variant="filled" radius="sm" size="lg">
                {complete ? <Check size={14} /> : index}
              </Badge>
            )}
            <div>
              <Title order={4}>{title}</Title>
              {subtitle && (
                <Text size="xs" c="dimmed">
                  {subtitle}
                </Text>
              )}
            </div>
          </Group>
          {action}
        </Group>
        {children}
      </Stack>
    </Paper>
  );
}
