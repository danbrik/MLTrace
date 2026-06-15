import { AppShell, Box, Button, Group, Stack, Text, Title } from '@mantine/core';
import { BrainCircuit, Database, FlaskConical, ListChecks, Rocket, Route, Workflow } from 'lucide-react';
import { useState } from 'react';

import { DatasetsPage } from './pages/DatasetsPage';
import { MethodsPage } from './pages/ModelsPage';
import { PreprocessingPipelinesPage } from './pages/PreprocessingPipelinesPage';
import { TrainingDatasetsPage } from './pages/TrainingDatasetsPage';
import { TrainingPipelinesPage } from './pages/TrainingPipelinesPage';
import { TrainingRunsPage } from './pages/TrainingRunsPage';
import { TestingRunsPage } from './pages/TestingRunsPage';

type Page =
  | 'datasets'
  | 'training-datasets'
  | 'preprocessing'
  | 'methods'
  | 'training-pipelines'
  | 'training-runs'
  | 'testing-runs';

export function App() {
  const [page, setPage] = useState<Page>('datasets');

  return (
    <AppShell
      header={{ height: 64 }}
      navbar={{ width: 260, breakpoint: 0 }}
      padding="md"
    >
      <AppShell.Header>
        <Group h="100%" px="md" justify="space-between">
          <Box>
            <Title order={2}>MLTrace</Title>
            <Text size="xs" c="dimmed">
              Dataset catalog, preprocessing, and method registry
            </Text>
          </Box>
        </Group>
      </AppShell.Header>

      <AppShell.Navbar p="md">
        <Stack gap="xs">
          <Button
            leftSection={<Database size={18} />}
            variant={page === 'datasets' ? 'filled' : 'subtle'}
            justify="flex-start"
            onClick={() => setPage('datasets')}
          >
            Datasets
          </Button>
          <Button
            leftSection={<ListChecks size={18} />}
            variant={page === 'training-datasets' ? 'filled' : 'subtle'}
            justify="flex-start"
            onClick={() => setPage('training-datasets')}
          >
            Train/Test Datasets
          </Button>
          <Button
            leftSection={<Workflow size={18} />}
            variant={page === 'preprocessing' ? 'filled' : 'subtle'}
            justify="flex-start"
            onClick={() => setPage('preprocessing')}
          >
            Preprocessing
          </Button>
          <Button
            leftSection={<BrainCircuit size={18} />}
            variant={page === 'methods' ? 'filled' : 'subtle'}
            justify="flex-start"
            onClick={() => setPage('methods')}
          >
            Methods
          </Button>
          <Button
            leftSection={<Route size={18} />}
            variant={page === 'training-pipelines' ? 'filled' : 'subtle'}
            justify="flex-start"
            onClick={() => setPage('training-pipelines')}
          >
            Training Pipelines
          </Button>
          <Button
            leftSection={<Rocket size={18} />}
            variant={page === 'training-runs' ? 'filled' : 'subtle'}
            justify="flex-start"
            onClick={() => setPage('training-runs')}
          >
            Training Runs
          </Button>
          <Button
            leftSection={<FlaskConical size={18} />}
            variant={page === 'testing-runs' ? 'filled' : 'subtle'}
            justify="flex-start"
            onClick={() => setPage('testing-runs')}
          >
            Testing Runs
          </Button>
        </Stack>
      </AppShell.Navbar>

      <AppShell.Main>
        <Box display={page === 'datasets' ? 'block' : 'none'}>
          <DatasetsPage />
        </Box>
        <Box display={page === 'training-datasets' ? 'block' : 'none'}>
          <TrainingDatasetsPage />
        </Box>
        <Box display={page === 'preprocessing' ? 'block' : 'none'}>
          <PreprocessingPipelinesPage />
        </Box>
        <Box display={page === 'methods' ? 'block' : 'none'}>
          <MethodsPage />
        </Box>
        <Box display={page === 'training-pipelines' ? 'block' : 'none'}>
          <TrainingPipelinesPage active={page === 'training-pipelines'} />
        </Box>
        <Box display={page === 'training-runs' ? 'block' : 'none'}>
          <TrainingRunsPage active={page === 'training-runs'} />
        </Box>
        <Box display={page === 'testing-runs' ? 'block' : 'none'}>
          <TestingRunsPage active={page === 'testing-runs'} />
        </Box>
      </AppShell.Main>
    </AppShell>
  );
}
