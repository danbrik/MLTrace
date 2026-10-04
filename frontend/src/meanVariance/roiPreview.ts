// Cancellation also guards responses from fetch implementations that finish after abort.
export function requestRoiPreview(url: string, ready: (image: Blob) => void, failed: (message: string) => void) {
  const controller = new AbortController();
  let current = true;
  const timer = setTimeout(async () => {
    try {
      const response = await fetch(url, {signal: controller.signal});
      if (!response.ok) throw new Error('Die Heatmap-Vorschau konnte nicht geladen werden. Bitte die ROI-Auswertung erneut öffnen.');
      const blob = await response.blob();
      if (current) ready(blob);
    } catch (error) {
      if (current) failed(error instanceof Error ? error.message : String(error));
    }
  }, 150);
  return () => {current = false; clearTimeout(timer); controller.abort();};
}
