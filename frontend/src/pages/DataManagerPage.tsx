import {
  ActionIcon,
  Alert,
  Badge,
  Button,
  Checkbox,
  Code,
  Collapse,
  Drawer,
  Group,
  Loader,
  Modal,
  Paper,
  ScrollArea,
  Select,
  Stack,
  Table,
  Text,
  TextInput,
  Title,
  Tooltip,
} from '@mantine/core';
import { notifications } from '@mantine/notifications';
import { ChevronDown, ChevronRight, Info, RefreshCw, Search, Trash2 } from 'lucide-react';
import { useCallback, useEffect, useMemo, useState } from 'react';

import {
  getRegistryDetail,
  getRegistrySummary,
  listRegistry,
  registryDelete,
  registryDeletePreview,
} from '../api';
import type {
  RegistryDeletePreview,
  RegistryDetail,
  RegistryFilterDef,
  RegistryItemRef,
  RegistryRow,
  RegistrySummary,
} from '../types';

const PAGE_SIZE = 50;

function formatBytes(bytes: number | null | undefined): string {
  if (bytes == null || !Number.isFinite(bytes) || bytes <= 0) return '—';
  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  let value = bytes;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit += 1;
  }
  return `${value >= 10 || unit === 0 ? Math.round(value) : value.toFixed(1)} ${units[unit]}`;
}

function formatDate(value: unknown): string {
  if (typeof value !== 'string') return '—';
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? '—' : date.toLocaleString();
}

function notifyError(title: string, error: unknown) {
  notifications.show({ color: 'red', title, message: error instanceof Error ? error.message : 'Unknown error' });
}

function JsonField({ label, value }: { label: string; value: unknown }) {
  const [open, setOpen] = useState(false);
  return (
    <Paper withBorder p="xs" radius="sm">
      <Group gap="xs" style={{ cursor: 'pointer' }} onClick={() => setOpen((current) => !current)}>
        {open ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
        <Text size="sm" fw={500}>
          {label}
        </Text>
      </Group>
      <Collapse in={open}>
        <ScrollArea.Autosize mah={320} mt="xs">
          <Code block>{JSON.stringify(value, null, 2)}</Code>
        </ScrollArea.Autosize>
      </Collapse>
    </Paper>
  );
}

function DetailDrawer({
  detail,
  labelOf,
  onClose,
  onOpenDependent,
  onDelete,
}: {
  detail: RegistryDetail | null;
  labelOf: (entityType: string) => string;
  onClose: () => void;
  onOpenDependent: (dep: RegistryItemRef) => void;
  onDelete: (item: RegistryItemRef) => void;
}) {
  if (!detail) return null;
  const primitives = Object.entries(detail.fields).filter(
    ([, value]) => value === null || ['string', 'number', 'boolean'].includes(typeof value),
  );
  const jsonFields = Object.entries(detail.fields).filter(
    ([, value]) => value !== null && typeof value === 'object',
  );
  return (
    <Drawer opened onClose={onClose} title={detail.name} position="right" size="xl">
      <Stack gap="md">
        <Group gap="xs">
          <Badge variant="light">{labelOf(detail.entity_type)}</Badge>
          <Badge variant="light" color="gray">#{detail.id}</Badge>
          <Button
            size="compact-sm"
            color="red"
            variant="light"
            leftSection={<Trash2 size={14} />}
            onClick={() => onDelete({ entity_type: detail.entity_type, id: detail.id })}
          >
            Delete…
          </Button>
        </Group>

        {detail.blockers.length > 0 && (
          <Alert color="orange">{detail.blockers.join(' ')}</Alert>
        )}

        <div>
          <Text size="sm" fw={500} c="dimmed" mb={4}>
            Used by ({detail.dependents.length})
          </Text>
          {detail.dependents.length === 0 ? (
            <Badge variant="light" color="green">
              Not used by any other object
            </Badge>
          ) : (
            <Group gap={6}>
              {detail.dependents.map((dep) => (
                <Badge
                  key={`${dep.entity_type}-${dep.id}`}
                  variant="light"
                  style={{ cursor: 'pointer', textTransform: 'none' }}
                  onClick={() => onOpenDependent({ entity_type: dep.entity_type, id: dep.id })}
                >
                  {labelOf(dep.entity_type)}: {dep.name}
                </Badge>
              ))}
            </Group>
          )}
        </div>

        {detail.artifacts.length > 0 && (
          <div>
            <Text size="sm" fw={500} c="dimmed" mb={4}>
              Files on disk
            </Text>
            <Stack gap={4}>
              {detail.artifacts.map((artifact) => (
                <Group key={artifact.path} gap="xs" wrap="nowrap">
                  <Badge variant="light" color={artifact.exists ? 'blue' : 'gray'} w={80}>
                    {artifact.exists ? formatBytes(artifact.size_bytes) : 'missing'}
                  </Badge>
                  <Text size="xs" ff="monospace" style={{ wordBreak: 'break-all' }}>
                    {artifact.path}
                  </Text>
                </Group>
              ))}
            </Stack>
          </div>
        )}

        <div>
          <Text size="sm" fw={500} c="dimmed" mb={4}>
            Properties
          </Text>
          <Table striped verticalSpacing={4}>
            <Table.Tbody>
              {primitives.map(([key, value]) => (
                <Table.Tr key={key}>
                  <Table.Td w={220}>
                    <Text size="xs" c="dimmed">
                      {key}
                    </Text>
                  </Table.Td>
                  <Table.Td>
                    <Text size="sm" style={{ wordBreak: 'break-all' }}>
                      {value === null ? '—' : String(value)}
                    </Text>
                  </Table.Td>
                </Table.Tr>
              ))}
            </Table.Tbody>
          </Table>
        </div>

        {jsonFields.length > 0 && (
          <div>
            <Text size="sm" fw={500} c="dimmed" mb={4}>
              Configuration (JSON)
            </Text>
            <Stack gap="xs">
              {jsonFields.map(([key, value]) => (
                <JsonField key={key} label={key} value={value} />
              ))}
            </Stack>
          </div>
        )}
      </Stack>
    </Drawer>
  );
}

function DeleteModal({
  items,
  labelOf,
  onClose,
  onDeleted,
}: {
  items: RegistryItemRef[];
  labelOf: (entityType: string) => string;
  onClose: () => void;
  onDeleted: () => void;
}) {
  const [preview, setPreview] = useState<RegistryDeletePreview | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [cascade, setCascade] = useState(false);
  const [deleting, setDeleting] = useState(false);

  useEffect(() => {
    setPreview(null);
    setError(null);
    setCascade(false);
    registryDeletePreview(items)
      .then(setPreview)
      .catch((err) => setError(err instanceof Error ? err.message : 'Unknown error'));
  }, [items]);

  const blocked = (preview?.blockers.length ?? 0) > 0;
  const needsCascade = (preview?.dependent_objects ?? 0) > 0;
  const canDelete = preview != null && !blocked && (!needsCascade || cascade) && !deleting;

  async function handleDelete() {
    if (!preview) return;
    setDeleting(true);
    try {
      const result = await registryDelete(items, cascade);
      const total = Object.values(result.deleted).reduce((sum, count) => sum + count, 0);
      notifications.show({
        color: 'green',
        title: `${total} object(s) deleted`,
        message: result.freed_bytes > 0 ? `Freed ${formatBytes(result.freed_bytes)} on disk.` : 'Done.',
      });
      onDeleted();
    } catch (err) {
      notifyError('Delete failed', err);
    } finally {
      setDeleting(false);
    }
  }

  return (
    <Modal opened onClose={onClose} title="Delete objects" size="lg">
      {error && <Alert color="red">{error}</Alert>}
      {!preview && !error && (
        <Group gap="xs">
          <Loader size="sm" />
          <Text size="sm">Collecting affected objects and files…</Text>
        </Group>
      )}
      {preview && (
        <Stack gap="md">
          {blocked && (
            <Alert color="red" title="Blocked">
              {preview.blockers.map((blocker) => (
                <Text key={blocker} size="sm">
                  {blocker}
                </Text>
              ))}
            </Alert>
          )}
          <div>
            <Text size="sm" fw={500} mb={4}>
              This will delete {preview.total_objects} object(s)
              {preview.dependent_objects > 0 ? ` (${preview.dependent_objects} dependent)` : ''}:
            </Text>
            <Stack gap="xs">
              {preview.groups.map((group) => (
                <Paper key={group.entity_type} withBorder p="xs" radius="sm">
                  <Text size="sm" fw={500}>
                    {group.label} ({group.items.length})
                  </Text>
                  <ScrollArea.Autosize mah={120}>
                    <Stack gap={2} mt={4}>
                      {group.items.map((item) => (
                        <Group key={item.id} gap={6}>
                          {item.selected ? (
                            <Badge size="xs" variant="filled">
                              selected
                            </Badge>
                          ) : (
                            <Badge size="xs" variant="light" color="orange">
                              dependent
                            </Badge>
                          )}
                          <Text size="xs">{item.name}</Text>
                        </Group>
                      ))}
                    </Stack>
                  </ScrollArea.Autosize>
                </Paper>
              ))}
            </Stack>
          </div>
          {preview.files.length > 0 && (
            <Alert color="yellow" title={`Deletes ${preview.files.length} file location(s) · frees ${formatBytes(preview.total_bytes)}`}>
              <ScrollArea.Autosize mah={100}>
                <Stack gap={2}>
                  {preview.files.map((file) => (
                    <Text key={file.path} size="xs" ff="monospace" style={{ wordBreak: 'break-all' }}>
                      {formatBytes(file.size_bytes)} · {file.path}
                    </Text>
                  ))}
                </Stack>
              </ScrollArea.Autosize>
            </Alert>
          )}
          {preview.notes.map((note) => (
            <Alert key={note} color="blue">
              {note}
            </Alert>
          ))}
          {needsCascade && (
            <Checkbox
              checked={cascade}
              onChange={(event) => setCascade(event.currentTarget.checked)}
              label="Also delete dependent objects and their files (CSVs, plots/frames, model artifacts)"
            />
          )}
          <Group justify="flex-end">
            <Button variant="default" onClick={onClose}>
              Cancel
            </Button>
            <Button color="red" disabled={!canDelete} loading={deleting} onClick={handleDelete}>
              Delete {preview.total_objects} object(s)
            </Button>
          </Group>
        </Stack>
      )}
    </Modal>
  );
}

export function DataManagerPage({ active = true }: { active?: boolean }) {
  const [summary, setSummary] = useState<RegistrySummary | null>(null);
  const [selectedType, setSelectedType] = useState<string>('preprocessing_pipeline');
  const [search, setSearch] = useState('');
  const [filters, setFilters] = useState<Record<string, string>>({});
  const [offset, setOffset] = useState(0);
  const [rows, setRows] = useState<RegistryRow[]>([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(false);
  const [selectedIds, setSelectedIds] = useState<Set<number>>(new Set());
  const [detail, setDetail] = useState<RegistryDetail | null>(null);
  const [deleteItems, setDeleteItems] = useState<RegistryItemRef[] | null>(null);

  const typeSummary = summary?.types.find((t) => t.key === selectedType) ?? null;
  const labelOf = useCallback(
    (entityType: string) => summary?.types.find((t) => t.key === entityType)?.label ?? entityType,
    [summary],
  );

  const refreshSummary = useCallback(() => {
    getRegistrySummary()
      .then(setSummary)
      .catch((error) => notifyError('Could not load data manager summary', error));
  }, []);

  const refreshRows = useCallback(() => {
    setLoading(true);
    listRegistry(selectedType, { search: search || undefined, filters, limit: PAGE_SIZE, offset })
      .then((result) => {
        setRows(result.rows);
        setTotal(result.total);
      })
      .catch((error) => notifyError('Could not load objects', error))
      .finally(() => setLoading(false));
  }, [selectedType, search, filters, offset]);

  useEffect(() => {
    if (!active) return;
    refreshSummary();
  }, [active, refreshSummary]);

  useEffect(() => {
    if (!active) return;
    refreshRows();
  }, [active, refreshRows]);

  function switchType(type: string) {
    setSelectedType(type);
    setFilters({});
    setSearch('');
    setOffset(0);
    setSelectedIds(new Set());
  }

  async function openDetail(item: RegistryItemRef) {
    try {
      const loaded = await getRegistryDetail(item.entity_type, item.id);
      setDetail(loaded);
    } catch (error) {
      notifyError('Could not load details', error);
    }
  }

  function afterDelete() {
    setDeleteItems(null);
    setDetail(null);
    setSelectedIds(new Set());
    refreshSummary();
    refreshRows();
  }

  const hasStatus = rows.some((row) => typeof row.status === 'string');
  const hasSize = rows.some((row) => row.disk_size_bytes != null);
  const allChecked = rows.length > 0 && rows.every((row) => selectedIds.has(row.id));

  const filterControls = useMemo(() => {
    if (!typeSummary) return null;
    const renderFilter = (filter: RegistryFilterDef) => {
      if (filter.kind === 'daterange') {
        const value = filters[filter.key] ?? '..';
        const [start, end] = value.split('..', 2);
        const update = (nextStart: string, nextEnd: string) => {
          const next = `${nextStart}..${nextEnd}`;
          setOffset(0);
          setFilters((current) => {
            const copy = { ...current };
            if (next === '..') delete copy[filter.key];
            else copy[filter.key] = next;
            return copy;
          });
        };
        return (
          <Group key={filter.key} gap={4}>
            <input
              type="datetime-local"
              value={start ?? ''}
              onChange={(event) => update(event.currentTarget.value, end ?? '')}
              title={`${filter.label} from`}
            />
            <Text size="xs" c="dimmed">
              –
            </Text>
            <input
              type="datetime-local"
              value={end ?? ''}
              onChange={(event) => update(start ?? '', event.currentTarget.value)}
              title={`${filter.label} to`}
            />
          </Group>
        );
      }
      const options = filter.kind === 'usage' ? ['used', 'unused'] : filter.options ?? [];
      return (
        <Select
          key={filter.key}
          placeholder={filter.label}
          size="xs"
          w={170}
          data={options}
          value={filters[filter.key] ?? null}
          clearable
          onChange={(value) => {
            setOffset(0);
            setFilters((current) => {
              const copy = { ...current };
              if (value) copy[filter.key] = value;
              else delete copy[filter.key];
              return copy;
            });
          }}
        />
      );
    };
    return typeSummary.filters.map(renderFilter);
  }, [typeSummary, filters]);

  return (
    <Stack gap="md">
      <div>
        <Title order={2}>Data Manager</Title>
        <Text c="dimmed" size="sm">
          Browse, search, and clean up every stored object — with full detail, usage tracking, and cascade delete.
        </Text>
      </div>

      <Group align="flex-start" wrap="nowrap" gap="md">
        <Paper withBorder p="sm" radius="sm" w={250} style={{ flexShrink: 0 }}>
          <Stack gap={4}>
            {(summary?.types ?? []).map((type) => (
              <Button
                key={type.key}
                variant={type.key === selectedType ? 'filled' : 'subtle'}
                justify="space-between"
                size="compact-sm"
                rightSection={
                  <Badge size="sm" variant={type.key === selectedType ? 'white' : 'light'} color="gray">
                    {type.count}
                  </Badge>
                }
                onClick={() => switchType(type.key)}
              >
                {type.label}
              </Button>
            ))}
            {!summary && <Loader size="sm" mx="auto" />}
          </Stack>
        </Paper>

        <Stack gap="sm" style={{ flex: 1, minWidth: 0 }}>
          <Group gap="xs">
            <TextInput
              placeholder="Search name, description…"
              leftSection={<Search size={14} />}
              size="xs"
              w={260}
              value={search}
              onChange={(event) => {
                setOffset(0);
                setSearch(event.currentTarget.value);
              }}
            />
            {filterControls}
            <Tooltip label="Refresh">
              <ActionIcon variant="subtle" onClick={() => { refreshSummary(); refreshRows(); }}>
                <RefreshCw size={16} />
              </ActionIcon>
            </Tooltip>
            {selectedIds.size > 0 && (
              <Button
                size="compact-sm"
                color="red"
                variant="light"
                leftSection={<Trash2 size={14} />}
                onClick={() =>
                  setDeleteItems([...selectedIds].map((id) => ({ entity_type: selectedType, id })))
                }
              >
                Delete {selectedIds.size} selected…
              </Button>
            )}
          </Group>

          <Paper withBorder radius="sm">
            <ScrollArea>
              <Table striped highlightOnHover verticalSpacing="xs" miw={760}>
                <Table.Thead>
                  <Table.Tr>
                    <Table.Th w={36}>
                      <Checkbox
                        size="xs"
                        checked={allChecked}
                        indeterminate={selectedIds.size > 0 && !allChecked}
                        onChange={() =>
                          setSelectedIds(allChecked ? new Set() : new Set(rows.map((row) => row.id)))
                        }
                      />
                    </Table.Th>
                    <Table.Th w={70}>ID</Table.Th>
                    <Table.Th>Name</Table.Th>
                    {hasStatus && <Table.Th>Status</Table.Th>}
                    <Table.Th>Created</Table.Th>
                    <Table.Th>Used by</Table.Th>
                    {hasSize && <Table.Th>Size</Table.Th>}
                    <Table.Th w={90} />
                  </Table.Tr>
                </Table.Thead>
                <Table.Tbody>
                  {rows.map((row) => (
                    <Table.Tr key={row.id}>
                      <Table.Td>
                        <Checkbox
                          size="xs"
                          checked={selectedIds.has(row.id)}
                          onChange={(event) => {
                            const next = new Set(selectedIds);
                            if (event.currentTarget.checked) next.add(row.id);
                            else next.delete(row.id);
                            setSelectedIds(next);
                          }}
                        />
                      </Table.Td>
                      <Table.Td>
                        <Text size="xs" c="dimmed">
                          {row.id}
                        </Text>
                      </Table.Td>
                      <Table.Td
                        style={{ cursor: 'pointer' }}
                        onClick={() => openDetail({ entity_type: selectedType, id: row.id })}
                      >
                        <Text size="sm">{row.name}</Text>
                      </Table.Td>
                      {hasStatus && (
                        <Table.Td>
                          {typeof row.status === 'string' ? (
                            <Badge size="sm" variant="light">
                              {row.status}
                            </Badge>
                          ) : (
                            '—'
                          )}
                        </Table.Td>
                      )}
                      <Table.Td>
                        <Text size="xs">{formatDate(row.created_at)}</Text>
                      </Table.Td>
                      <Table.Td>
                        <Badge size="sm" variant="light" color={row.usage_count > 0 ? 'blue' : 'green'}>
                          {row.usage_count > 0 ? `${row.usage_count} object(s)` : 'unused'}
                        </Badge>
                      </Table.Td>
                      {hasSize && (
                        <Table.Td>
                          <Text size="xs">{formatBytes(row.disk_size_bytes)}</Text>
                        </Table.Td>
                      )}
                      <Table.Td>
                        <Group gap={4} justify="flex-end" wrap="nowrap">
                          <Tooltip label="Details">
                            <ActionIcon
                              variant="subtle"
                              onClick={() => openDetail({ entity_type: selectedType, id: row.id })}
                            >
                              <Info size={16} />
                            </ActionIcon>
                          </Tooltip>
                          <Tooltip label="Delete…">
                            <ActionIcon
                              color="red"
                              variant="subtle"
                              onClick={() => setDeleteItems([{ entity_type: selectedType, id: row.id }])}
                            >
                              <Trash2 size={16} />
                            </ActionIcon>
                          </Tooltip>
                        </Group>
                      </Table.Td>
                    </Table.Tr>
                  ))}
                </Table.Tbody>
              </Table>
            </ScrollArea>
            {loading && (
              <Group justify="center" p="sm">
                <Loader size="sm" />
              </Group>
            )}
            {!loading && rows.length === 0 && (
              <Text size="sm" c="dimmed" ta="center" p="md">
                No objects match the current search/filters.
              </Text>
            )}
          </Paper>

          <Group justify="space-between">
            <Text size="xs" c="dimmed">
              {total} object(s) total
            </Text>
            <Group gap="xs">
              <Button size="compact-xs" variant="default" disabled={offset === 0} onClick={() => setOffset(Math.max(0, offset - PAGE_SIZE))}>
                Previous
              </Button>
              <Text size="xs" c="dimmed">
                {Math.floor(offset / PAGE_SIZE) + 1} / {Math.max(1, Math.ceil(total / PAGE_SIZE))}
              </Text>
              <Button
                size="compact-xs"
                variant="default"
                disabled={offset + PAGE_SIZE >= total}
                onClick={() => setOffset(offset + PAGE_SIZE)}
              >
                Next
              </Button>
            </Group>
          </Group>
        </Stack>
      </Group>

      <DetailDrawer
        detail={detail}
        labelOf={labelOf}
        onClose={() => setDetail(null)}
        onOpenDependent={(dep) => openDetail(dep)}
        onDelete={(item) => setDeleteItems([item])}
      />

      {deleteItems && (
        <DeleteModal
          items={deleteItems}
          labelOf={labelOf}
          onClose={() => setDeleteItems(null)}
          onDeleted={afterDelete}
        />
      )}
    </Stack>
  );
}
