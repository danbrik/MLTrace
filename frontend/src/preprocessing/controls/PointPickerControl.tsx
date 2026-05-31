import { Badge, Group, Stack, Text } from '@mantine/core';
import { useEffect, useState } from 'react';

import { defaultPoints, pointFromEvent, type Point } from './geometry';
import type { StepControl, StepControlProps } from './types';

function PointPickerControl({ inputImage, config, onChange }: StepControlProps) {
  const [draggingPoint, setDraggingPoint] = useState<number | null>(null);
  const points = (config.source_points as Point[] | undefined) ?? [];

  // Initialise the four source points in the input image's coordinate space once available.
  useEffect(() => {
    if (points.length !== 4) {
      onChange({ source_points: defaultPoints(inputImage.width, inputImage.height) });
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [inputImage.node_id]);

  function updatePoint(index: number, point: Point) {
    const next = points.length === 4 ? [...points] : defaultPoints(inputImage.width, inputImage.height);
    next[index] = point;
    onChange({ source_points: next });
  }

  return (
    <Stack gap="xs">
      <Text size="xs" c="dimmed">
        Drag the four points on the input image. The orange area is the region that gets transformed.
      </Text>
      <div
        className="warp-picker"
        onPointerMove={(event) => {
          if (draggingPoint !== null) updatePoint(draggingPoint, pointFromEvent(event, inputImage));
        }}
        onPointerUp={() => setDraggingPoint(null)}
        onPointerLeave={() => setDraggingPoint(null)}
      >
        <img src={inputImage.image_data_url} alt="Step input" className="warp-picker-image" />
        {points.length === 4 && (
          <svg
            className="warp-overlay"
            viewBox={`0 0 ${inputImage.width} ${inputImage.height}`}
            preserveAspectRatio="none"
          >
            <polygon
              points={points.map((point) => `${point.x},${point.y}`).join(' ')}
              fill="rgba(255, 140, 0, 0.32)"
              stroke="#fd7e14"
              strokeWidth={Math.max(1, inputImage.width / 260)}
              strokeLinejoin="round"
            />
          </svg>
        )}
        {points.map((point, index) => (
          <button
            key={index}
            type="button"
            className="warp-point"
            style={{
              left: `${(point.x / inputImage.width) * 100}%`,
              top: `${(point.y / inputImage.height) * 100}%`,
            }}
            onPointerDown={(event) => {
              event.preventDefault();
              event.currentTarget.setPointerCapture(event.pointerId);
              setDraggingPoint(index);
            }}
            onPointerUp={(event) => {
              event.currentTarget.releasePointerCapture(event.pointerId);
              setDraggingPoint(null);
            }}
          >
            {index + 1}
          </button>
        ))}
      </div>
      <Group gap="xs">
        {points.map((point, index) => (
          <Badge key={`${point.x}-${point.y}-${index}`} variant="light">
            {index + 1}: {point.x}, {point.y}
          </Badge>
        ))}
      </Group>
    </Stack>
  );
}

export const pointPickerControl: StepControl = {
  component: PointPickerControl,
  ownedKeys: ['source_points'],
};
