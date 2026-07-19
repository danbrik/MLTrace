import {
  Alert,
  Button,
  Card,
  Container,
  Group,
  Modal,
  SimpleGrid,
  Stack,
  Text,
  Textarea,
  TextInput,
  Title,
} from '@mantine/core';
import { notifications } from '@mantine/notifications';
import { FolderOpen, Plus } from 'lucide-react';
import { useEffect, useState } from 'react';

import { createProject, listProjects, markProjectOpened } from '../api';
import type { Project } from '../types';

export function ProjectsPage({ onOpen }: { onOpen: (project: Project) => void }) {
  const [projects, setProjects] = useState<Project[]>([]);
  const [loading, setLoading] = useState(true);
  const [opened, setOpened] = useState(false);
  const [name, setName] = useState('');
  const [description, setDescription] = useState('');
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    listProjects()
      .then(setProjects)
      .catch((error) => notifications.show({ color: 'red', title: 'Could not load projects', message: error.message }))
      .finally(() => setLoading(false));
  }, []);

  async function openProject(project: Project) {
    const updated = await markProjectOpened(project.id).catch(() => project);
    onOpen(updated);
  }

  async function submit() {
    if (!name.trim() || !description.trim() || saving) return;
    setSaving(true);
    try {
      const project = await createProject({ name: name.trim(), description: description.trim() });
      setProjects((current) => [project, ...current]);
      setOpened(false);
      setName('');
      setDescription('');
      await openProject(project);
    } catch (error) {
      notifications.show({
        color: 'red',
        title: 'Could not create project',
        message: error instanceof Error ? error.message : 'Unknown error',
      });
    } finally {
      setSaving(false);
    }
  }

  return (
    <Container size="lg" py={64}>
      <Stack gap="xl">
        <Group justify="space-between" align="end">
          <div>
            <Title order={1}>MLTrace</Title>
            <Text c="dimmed" mt={4}>Choose an isolated ML project or create a new workspace.</Text>
          </div>
          <Button leftSection={<Plus size={18} />} onClick={() => setOpened(true)}>New project</Button>
        </Group>

        {!loading && projects.length === 0 && (
          <Alert color="blue" title="No projects yet">Create your first project to start working with MLTrace.</Alert>
        )}
        <SimpleGrid cols={{ base: 1, sm: 2, lg: 3 }}>
          {projects.map((project) => (
            <Card key={project.id} withBorder padding="lg" radius="md">
              <Stack h="100%" gap="sm">
                <Title order={3}>{project.name}</Title>
                <Text size="sm" c="dimmed" style={{ flex: 1 }}>{project.description}</Text>
                <Text size="xs" c="dimmed">
                  {project.last_opened_at
                    ? `Last opened ${new Date(project.last_opened_at).toLocaleString()}`
                    : `Created ${new Date(project.created_at).toLocaleString()}`}
                </Text>
                <Button variant="light" leftSection={<FolderOpen size={17} />} onClick={() => openProject(project)}>
                  Open project
                </Button>
              </Stack>
            </Card>
          ))}
        </SimpleGrid>
      </Stack>

      <Modal opened={opened} onClose={() => setOpened(false)} title="Create project">
        <Stack>
          <TextInput label="Name" required maxLength={100} value={name} onChange={(event) => setName(event.currentTarget.value)} />
          <Textarea
            label="Short description"
            required
            minRows={3}
            maxLength={500}
            value={description}
            onChange={(event) => setDescription(event.currentTarget.value)}
          />
          <Group justify="flex-end">
            <Button variant="default" onClick={() => setOpened(false)}>Cancel</Button>
            <Button loading={saving} disabled={!name.trim() || !description.trim()} onClick={submit}>Create</Button>
          </Group>
        </Stack>
      </Modal>
    </Container>
  );
}
