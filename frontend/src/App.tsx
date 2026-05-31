import { AppShell, Box, Button, Group, Stack, Text, Title } from '@mantine/core';
import { Database, ListChecks } from 'lucide-react';
import { useState } from 'react';

import { DatasetsPage } from './pages/DatasetsPage';
import { TrainingDatasetsPage } from './pages/TrainingDatasetsPage';

type Page = 'datasets' | 'training-datasets';

export function App() {
  const [page, setPage] = useState<Page>('datasets');

  return (
    <AppShell
      header={{ height: 64 }}
      navbar={{ width: 260, breakpoint: 'sm' }}
      padding="md"
    >
      <AppShell.Header>
        <Group h="100%" px="md" justify="space-between">
          <Box>
            <Title order={2}>MLTrace</Title>
            <Text size="xs" c="dimmed">
              Dataset catalog and training dataset builder
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
            Training Datasets
          </Button>
        </Stack>
      </AppShell.Navbar>

      <AppShell.Main>
        {page === 'datasets' ? <DatasetsPage /> : <TrainingDatasetsPage />}
      </AppShell.Main>
    </AppShell>
  );
}

