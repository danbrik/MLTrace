import { Badge, Group, Stack, Text } from '@mantine/core';
import { useRef } from 'react';
import type { PointerEvent } from 'react';

import { clamp, cropRectFromConfig, pointFromEvent, type CropDrag, type CropMode } from './geometry';
import type { StepControl, StepControlProps } from './types';

function CropBoxControl({ inputImage, config, disabled, onChange }: StepControlProps) {
  const cropDragRef = useRef<CropDrag | null>(null);
  const rect = cropRectFromConfig(config, inputImage);
  const toPct = {
    left: `${(rect.x / inputImage.width) * 100}%`,
    top: `${(rect.y / inputImage.height) * 100}%`,
    width: `${(rect.width / inputImage.width) * 100}%`,
    height: `${(rect.height / inputImage.height) * 100}%`,
  };

  function onPointerMove(event: PointerEvent<HTMLDivElement>) {
    if (disabled) return;
    const drag = cropDragRef.current;
    if (!drag) return;
    const point = pointFromEvent(event, inputImage);
    if (drag.mode === 'move') {
      const dx = point.x - drag.startX;
      const dy = point.y - drag.startY;
      const x = clamp(drag.x + dx, 0, inputImage.width - drag.width);
      const y = clamp(drag.y + dy, 0, inputImage.height - drag.height);
      onChange({ x, y });
    } else if (drag.mode === 'tl') {
      const right = drag.x + drag.width;
      const bottom = drag.y + drag.height;
      const x = clamp(point.x, 0, right - 1);
      const y = clamp(point.y, 0, bottom - 1);
      onChange({ x, y, width: right - x, height: bottom - y });
    } else {
      const width = clamp(point.x - drag.x, 1, inputImage.width - drag.x);
      const height = clamp(point.y - drag.y, 1, inputImage.height - drag.y);
      onChange({ width, height });
    }
  }

  function startDrag(event: PointerEvent<HTMLElement>, mode: CropMode) {
    if (disabled) return;
    event.preventDefault();
    event.stopPropagation();
    const point = pointFromEvent(event as unknown as PointerEvent<HTMLDivElement>, inputImage);
    cropDragRef.current = { mode, startX: point.x, startY: point.y, ...rect };
    (event.currentTarget as HTMLElement).setPointerCapture?.(event.pointerId);
  }

  const endDrag = () => {
    cropDragRef.current = null;
  };

  return (
    <Stack gap="xs">
      <Text size="xs" c="dimmed">
        Drag the orange box to move it, or drag a corner handle to resize the cropped region.
      </Text>
      <div className="warp-picker" onPointerMove={onPointerMove} onPointerUp={endDrag} onPointerLeave={endDrag}>
        <img src={inputImage.image_data_url} alt="Step input" className="warp-picker-image" />
        <div
          className="crop-rect"
          style={toPct}
          onPointerDown={(event) => startDrag(event, 'move')}
          onPointerUp={endDrag}
        >
          <span className="crop-handle crop-handle-tl" onPointerDown={(event) => startDrag(event, 'tl')} onPointerUp={endDrag} />
          <span className="crop-handle crop-handle-br" onPointerDown={(event) => startDrag(event, 'br')} onPointerUp={endDrag} />
        </div>
      </div>
      <Group gap="xs">
        <Badge variant="light">x: {rect.x}</Badge>
        <Badge variant="light">y: {rect.y}</Badge>
        <Badge variant="light">w: {rect.width}</Badge>
        <Badge variant="light">h: {rect.height}</Badge>
      </Group>
    </Stack>
  );
}

export const cropBoxControl: StepControl = {
  component: CropBoxControl,
  ownedKeys: ['x', 'y', 'width', 'height'],
};
