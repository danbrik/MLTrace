import { Alert, Button, Stack, Text } from '@mantine/core';
import { RotateCcw } from 'lucide-react';
import { Component, type ReactNode } from 'react';

type Props = {
  children: ReactNode;
  label: string;
};

type State = {
  error: Error | null;
};

export class PageErrorBoundary extends Component<Props, State> {
  state: State = { error: null };

  static getDerivedStateFromError(error: Error): State {
    return { error };
  }

  render() {
    if (!this.state.error) {
      return this.props.children;
    }

    return (
      <Alert color="red" title={`${this.props.label} crashed`}>
        <Stack gap="sm">
          <Text size="sm">{this.state.error.message}</Text>
          <Button
            variant="light"
            leftSection={<RotateCcw size={16} />}
            onClick={() => this.setState({ error: null })}
          >
            Reset page
          </Button>
        </Stack>
      </Alert>
    );
  }
}
