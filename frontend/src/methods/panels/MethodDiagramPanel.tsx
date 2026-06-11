import { Alert, Badge, Paper, ScrollArea, Stack, Text } from '@mantine/core';
import { ArrowRight } from 'lucide-react';

import type { ModelDiagram } from '../../types';

export function MethodDiagramPanel({ diagram, error }: { diagram: ModelDiagram | null; error: string | null }) {
  if (error) {
    return (
      <Alert color="red" title="Method validation failed">
        {error}
      </Alert>
    );
  }
  if (!diagram) {
    return <Alert color="blue">Diagram preview will appear after the method definition validates.</Alert>;
  }
  if (!Array.isArray(diagram.nodes) || !Array.isArray(diagram.edges)) {
    return <Alert color="yellow">Diagram preview is unavailable until the method definition validates.</Alert>;
  }
  return (
    <ScrollArea>
      <div className="model-diagram-row">
        {diagram.nodes.map((node, index) => (
          <div key={node.id} className="model-diagram-item">
            <Paper withBorder p="sm" radius="sm" className={`model-diagram-node model-section-${node.section}`}>
              <Stack gap={4}>
                <Badge size="xs" variant="light">
                  {node.section}
                </Badge>
                <Text fw={700} size="sm">
                  {node.label}
                </Text>
                <Text size="xs" c="dimmed">
                  {node.detail}
                </Text>
              </Stack>
            </Paper>
            {index < diagram.nodes.length - 1 && <ArrowRight size={18} className="model-diagram-arrow" />}
          </div>
        ))}
      </div>
    </ScrollArea>
  );
}
