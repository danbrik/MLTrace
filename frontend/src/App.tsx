import { AppShell, Box, Button, Group, Stack, Text, Title, Tooltip, ActionIcon } from '@mantine/core';
import {
  BarChart3,
  BrainCircuit,
  CalendarClock,
  Database,
  FlaskConical,
  ListChecks,
  PanelLeftClose,
  PanelLeftOpen,
  Route,
  Workflow,
} from 'lucide-react';
import { useState } from 'react';
import type React from 'react';

import { PageErrorBoundary } from './components/PageErrorBoundary';
import { AnalysisPage } from './pages/AnalysisPage';
import { DatasetsPage } from './pages/DatasetsPage';
import { MethodsPage } from './pages/ModelsPage';
import { PreprocessingPipelinesPage } from './pages/PreprocessingPipelinesPage';
import { SchedulerPage } from './pages/SchedulerPage';
import { TrainingDatasetsPage } from './pages/TrainingDatasetsPage';
import { TrainingPipelinesPage } from './pages/TrainingPipelinesPage';
import { TestingRunsPage } from './pages/TestingRunsPage';

type Page =
  | 'datasets'
  | 'training-datasets'
  | 'preprocessing'
  | 'methods'
  | 'training-pipelines'
  | 'testing'
  | 'analysis'
  | 'scheduler';

export function App() {
  const [page, setPage] = useState<Page>('datasets');
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);

  const navItems: Array<{ id: Page; label: string; icon: React.ReactNode }> = [
    { id: 'datasets', label: 'Datasets', icon: <Database size={18} /> },
    { id: 'training-datasets', label: 'Train/Test Datasets', icon: <ListChecks size={18} /> },
    { id: 'preprocessing', label: 'Preprocessing', icon: <Workflow size={18} /> },
    { id: 'methods', label: 'Methods', icon: <BrainCircuit size={18} /> },
    { id: 'training-pipelines', label: 'Training Pipelines', icon: <Route size={18} /> },
    { id: 'testing', label: 'Inference', icon: <FlaskConical size={18} /> },
    { id: 'analysis', label: 'Analysis', icon: <BarChart3 size={18} /> },
    { id: 'scheduler', label: 'Scheduler', icon: <CalendarClock size={18} /> },
  ];

  return (
    <AppShell
      header={{ height: 64 }}
      navbar={{ width: sidebarCollapsed ? 78 : 260, breakpoint: 0 }}
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
          <Group justify={sidebarCollapsed ? 'center' : 'space-between'} mb="xs">
            {!sidebarCollapsed && (
              <Text size="xs" fw={700} c="dimmed" tt="uppercase">
                Navigation
              </Text>
            )}
            <Tooltip label={sidebarCollapsed ? 'Expand sidebar' : 'Collapse sidebar'} position="right">
              <ActionIcon
                variant="subtle"
                aria-label={sidebarCollapsed ? 'Expand sidebar' : 'Collapse sidebar'}
                onClick={() => setSidebarCollapsed((current) => !current)}
              >
                {sidebarCollapsed ? <PanelLeftOpen size={18} /> : <PanelLeftClose size={18} />}
              </ActionIcon>
            </Tooltip>
          </Group>
          {navItems.map((item) =>
            sidebarCollapsed ? (
              <Tooltip key={item.id} label={item.label} position="right">
                <ActionIcon
                  size="lg"
                  radius="sm"
                  variant={page === item.id ? 'filled' : 'subtle'}
                  aria-label={item.label}
                  onClick={() => setPage(item.id)}
                >
                  {item.icon}
                </ActionIcon>
              </Tooltip>
            ) : (
              <Button
                key={item.id}
                leftSection={item.icon}
                variant={page === item.id ? 'filled' : 'subtle'}
                justify="flex-start"
                onClick={() => setPage(item.id)}
              >
                {item.label}
              </Button>
            ),
          )}
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
          <TrainingPipelinesPage active={page === 'training-pipelines'} onRunQueued={() => setPage('scheduler')} />
        </Box>
        <Box display={page === 'testing' ? 'block' : 'none'}>
          <TestingRunsPage active={page === 'testing'} onRunQueued={() => setPage('scheduler')} />
        </Box>
        <Box display={page === 'analysis' ? 'block' : 'none'}>
          <PageErrorBoundary label="Analysis">
            <AnalysisPage active={page === 'analysis'} />
          </PageErrorBoundary>
        </Box>
        <Box display={page === 'scheduler' ? 'block' : 'none'}>
          <PageErrorBoundary label="Scheduler">
            <SchedulerPage active={page === 'scheduler'} />
          </PageErrorBoundary>
        </Box>
      </AppShell.Main>
    </AppShell>
  );
}
