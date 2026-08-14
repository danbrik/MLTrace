import { Group, Pagination, Text } from '@mantine/core';

export const DEFAULT_TABLE_PAGE_SIZE = 10;

type TablePaginationProps = {
  totalItems: number;
  page: number;
  onChange: (page: number) => void;
  pageSize?: number;
};

export function TablePagination({
  totalItems,
  page,
  onChange,
  pageSize = DEFAULT_TABLE_PAGE_SIZE,
}: TablePaginationProps) {
  const totalPages = Math.max(1, Math.ceil(totalItems / pageSize));
  if (totalPages <= 1) return null;

  const firstItem = (page - 1) * pageSize + 1;
  const lastItem = Math.min(page * pageSize, totalItems);

  return (
    <Group justify="space-between" gap="sm" wrap="wrap">
      <Text size="xs" c="dimmed">
        Showing {firstItem}–{lastItem} of {totalItems}
      </Text>
      <Pagination total={totalPages} value={page} onChange={onChange} withEdges />
    </Group>
  );
}

