import { AppShell, Box, Button, Group, Stack, Text, Title, Tooltip, ActionIcon } from '@mantine/core';
import {
  Archive,
  BarChart3,
  BrainCircuit,
  CalendarClock,
  Database,
  Eye,
  FlaskConical,
  ListChecks,
  SlidersHorizontal,
  PanelLeftClose,
  PanelLeftOpen,
  Route,
  Workflow,
} from 'lucide-react';
import { useEffect, useState } from 'react';
import type React from 'react';

import { PageErrorBoundary } from './components/PageErrorBoundary';
import { AnalysisPage } from './pages/AnalysisPage';
import { DataManagerPage } from './pages/DataManagerPage';
import { DatasetsPage } from './pages/DatasetsPage';
import { InspectPage } from './pages/InspectPage';
import { MethodsPage } from './pages/ModelsPage';
import { OptimizationPage } from './pages/OptimizationPage';
import { PreprocessingPipelinesPage } from './pages/PreprocessingPipelinesPage';
import { SchedulerPage } from './pages/SchedulerPage';
import { ProjectsPage } from './pages/ProjectsPage';
import { TrainingDatasetsPage } from './pages/TrainingDatasetsPage';
import { TrainingPipelinesPage } from './pages/TrainingPipelinesPage';
import { TestingRunsPage } from './pages/TestingRunsPage';
import { getProject, setActiveProject } from './api';
import type { Project } from './types';

type Page =
  | 'datasets'
  | 'training-datasets'
  | 'preprocessing'
  | 'methods'
  | 'training-pipelines'
  | 'testing'
  | 'inspect'
  | 'optimization'
  | 'analysis'
  | 'scheduler'
  | 'data-manager';

export function App() {
  const [location, setLocation] = useState(window.location.pathname);
  const match = location.match(/^\/projects\/([^/]+)(?:\/([^/]+))?\/?$/);
  const projectId = match?.[1] ?? null;
  // Set the request context during render so child page effects cannot issue a
  // project-scoped request before the parent effect runs on a direct URL load.
  setActiveProject(projectId);
  const requestedPage = match?.[2] as Page | undefined;
  const page: Page = requestedPage && [
    'datasets', 'training-datasets', 'preprocessing', 'methods', 'training-pipelines', 'testing',
    'inspect', 'optimization', 'analysis', 'scheduler', 'data-manager',
  ].includes(requestedPage) ? requestedPage : 'datasets';
  const [project, setProject] = useState<Project | null>(null);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);

  useEffect(() => {
    const update = () => setLocation(window.location.pathname);
    window.addEventListener('popstate', update);
    return () => window.removeEventListener('popstate', update);
  }, []);

  useEffect(() => {
    let cancelled = false;
    setActiveProject(projectId);
    if (!projectId) {
      setProject(null);
      return undefined;
    }
    getProject(projectId)
      .then((value) => { if (!cancelled) setProject(value); })
      .catch(() => {
        if (!cancelled) navigate('/');
      });
    return () => { cancelled = true; };
  }, [projectId]);

  function navigate(path: string) {
    window.history.pushState({}, '', path);
    setLocation(path);
  }

  function setPage(next: Page) {
    if (projectId) navigate(`/projects/${projectId}/${next}`);
  }

  if (!projectId) {
    return <ProjectsPage onOpen={(selected) => navigate(`/projects/${selected.id}/datasets`)} />;
  }

  const navItems: Array<{ id: Page; label: string; icon: React.ReactNode }> = [
    { id: 'datasets', label: 'Datasets', icon: <Database size={18} /> },
    { id: 'training-datasets', label: 'Train/Test Datasets', icon: <ListChecks size={18} /> },
    { id: 'preprocessing', label: 'Preprocessing', icon: <Workflow size={18} /> },
    { id: 'methods', label: 'Methods', icon: <BrainCircuit size={18} /> },
    { id: 'training-pipelines', label: 'Training Pipelines', icon: <Route size={18} /> },
    { id: 'testing', label: 'Inference', icon: <FlaskConical size={18} /> },
    { id: 'inspect', label: 'Inspect', icon: <Eye size={18} /> },
    { id: 'optimization', label: 'Optimization', icon: <SlidersHorizontal size={18} /> },
    { id: 'analysis', label: 'Analysis', icon: <BarChart3 size={18} /> },
    { id: 'scheduler', label: 'Scheduler', icon: <CalendarClock size={18} /> },
    { id: 'data-manager', label: 'Data Manager', icon: <Archive size={18} /> },
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
              {project?.name ?? 'Loading project…'}
            </Text>
          </Box>
          <Button variant="subtle" onClick={() => { setActiveProject(null); navigate('/'); }}>
            Leave project
          </Button>
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
          <DatasetsPage active={page === 'datasets'} />
        </Box>
        <Box display={page === 'training-datasets' ? 'block' : 'none'}>
          <TrainingDatasetsPage active={page === 'training-datasets'} />
        </Box>
        <Box display={page === 'preprocessing' ? 'block' : 'none'}>
          <PreprocessingPipelinesPage active={page === 'preprocessing'} />
        </Box>
        <Box display={page === 'methods' ? 'block' : 'none'}>
          <MethodsPage active={page === 'methods'} />
        </Box>
        <Box display={page === 'training-pipelines' ? 'block' : 'none'}>
          <TrainingPipelinesPage active={page === 'training-pipelines'} onRunQueued={() => setPage('scheduler')} />
        </Box>
        <Box display={page === 'testing' ? 'block' : 'none'}>
          <TestingRunsPage active={page === 'testing'} onRunQueued={() => setPage('scheduler')} />
        </Box>
        <Box display={page === 'inspect' ? 'block' : 'none'}>
          <PageErrorBoundary label="Inspect">
            <InspectPage active={page === 'inspect'} />
          </PageErrorBoundary>
        </Box>
        <Box display={page === 'optimization' ? 'block' : 'none'}>
          <PageErrorBoundary label="Optimization">
            <OptimizationPage active={page === 'optimization'} />
          </PageErrorBoundary>
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
        <Box display={page === 'data-manager' ? 'block' : 'none'}>
          <PageErrorBoundary label="Data Manager">
            <DataManagerPage active={page === 'data-manager'} />
          </PageErrorBoundary>
        </Box>
      </AppShell.Main>
    </AppShell>
  );
}
