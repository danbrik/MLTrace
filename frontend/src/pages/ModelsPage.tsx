import { Alert, Badge, Button, Group, Paper, Select, Stack, Text, Textarea, TextInput, Title } from '@mantine/core';
import { useDebouncedValue } from '@mantine/hooks';
import { notifications } from '@mantine/notifications';
import { BrainCircuit, Pencil, RotateCcw, Save } from 'lucide-react';
import { useCallback, useEffect, useMemo, useState } from 'react';

import {
  buildMethodDiagram,
  createMethodConfiguration,
  deleteMethodConfiguration,
  getMethodConfiguration,
  listMethodConfigurations,
  listMethodDefinitions,
  listModelLayers,
  runMethodTorchCheck,
  updateMethodConfiguration,
} from '../api';
import { StepCard } from '../components/StepCard';
import { usePendingIds } from '../hooks/usePendingIds';
import { ArchitectureCheckPanel } from '../methods/panels/ArchitectureCheckPanel';
import { MethodDiagramPanel } from '../methods/panels/MethodDiagramPanel';
import { SavedMethodsTable } from '../methods/panels/SavedMethodsTable';
import { TorchCheckPanel } from '../methods/panels/TorchCheckPanel';
import { getMethodBuilder } from '../methods/registry';
import type { NumericDraftState } from '../methods/types';
import {
  buildMethodPayload,
  nextAvailableMethodName,
  schemaDefaults,
} from '../methods/utils';
import type {
  MethodConfiguration,
  MethodDefinition,
  MethodTorchCheckResponse,
  MethodValidationResponse,
  ModelDiagram,
  ModelGraph,
  ModelLayerDefinition,
} from '../types';

export function MethodsPage() {
  const [methodDefinitions, setMethodDefinitions] = useState<MethodDefinition[]>([]);
  const [layers, setLayers] = useState<ModelLayerDefinition[]>([]);
  const [methods, setMethods] = useState<MethodConfiguration[]>([]);
  const [methodType, setMethodType] = useState<string | null>(null);
  const [name, setName] = useState('');
  const [nameTouched, setNameTouched] = useState(false);
  const [description, setDescription] = useState('');
  const [modelConfig, setModelConfig] = useState<Record<string, unknown>>({});
  const [trainingConfig, setTrainingConfig] = useState<Record<string, unknown>>({});
  const [inferenceConfig, setInferenceConfig] = useState<Record<string, unknown>>({});
  const [modelGraph, setModelGraph] = useState<ModelGraph>({});
  const [diagram, setDiagram] = useState<ModelDiagram | null>(null);
  const [diagramError, setDiagramError] = useState<string | null>(null);
  const [architectureCheck, setArchitectureCheck] = useState<MethodValidationResponse | null>(null);
  const [torchCheck, setTorchCheck] = useState<MethodTorchCheckResponse | null>(null);
  const [torchCheckLoading, setTorchCheckLoading] = useState(false);
  const [torchCheckLogs, setTorchCheckLogs] = useState<string[]>([]);
  const [loadedMethodId, setLoadedMethodId] = useState<number | null>(null);
  const [isEditingLoadedMethod, setIsEditingLoadedMethod] = useState(true);
  const [loading, setLoading] = useState(false);
  const [numericDrafts, setNumericDrafts] = useState<Record<string, NumericDraftState>>({});
  const rowActions = usePendingIds();

  async function refresh() {
    const [nextMethodDefinitions, nextLayers, nextMethods] = await Promise.all([
      listMethodDefinitions(),
      listModelLayers(),
      listMethodConfigurations(),
    ]);
    setMethodDefinitions(nextMethodDefinitions);
    setLayers(nextLayers);
    setMethods(nextMethods);
  }

  useEffect(() => {
    refresh().catch((error) => {
      notifications.show({
        color: 'red',
        title: 'Could not load method data',
        message: error instanceof Error ? error.message : 'Unknown error',
      });
    });
  }, []);

  useEffect(() => {
    if (loadedMethodId == null && !nameTouched) {
      setName(nextAvailableMethodName(methods));
    }
  }, [methods, loadedMethodId, nameTouched]);

  const methodByType = useMemo(
    () => new Map(methodDefinitions.map((method) => [method.type, method])),
    [methodDefinitions],
  );
  const layerByType = useMemo(() => new Map(layers.map((layer) => [layer.type, layer])), [layers]);
  const selectedMethod = methodType ? methodByType.get(methodType) : undefined;
  const builderDefinition = getMethodBuilder(selectedMethod?.builder_kind);
  const BuilderComponent = builderDefinition?.component;
  const isTrainableArchitecture = selectedMethod?.training_mode === 'gradient';
  const builderTitle = isTrainableArchitecture ? 'Build Architecture' : 'Build Method';
  const saveLabel = isTrainableArchitecture ? 'Save Architecture' : 'Save Method';
  const updateLabel = isTrainableArchitecture ? 'Update Architecture' : 'Update Method';
  const saveAsNewLabel = isTrainableArchitecture ? 'Save as New Architecture' : 'Save as New Method';
  const resetLabel = isTrainableArchitecture ? 'Reset Architecture' : 'Reset Method';
  const loadedReadOnly = loadedMethodId != null && !isEditingLoadedMethod;
  const loadedMethod = loadedMethodId != null ? methods.find((method) => method.id === loadedMethodId) ?? null : null;
  const invalidNumericDrafts = useMemo(
    () => Object.entries(numericDrafts).filter(([, draft]) => draft.dirty && !draft.valid),
    [numericDrafts],
  );

  const nameClash = useMemo(() => {
    const trimmed = name.trim().toLowerCase();
    if (!trimmed) return false;
    return methods.some((method) => method.name.trim().toLowerCase() === trimmed && method.id !== loadedMethodId);
  }, [name, methods, loadedMethodId]);

  function resetForMethod(method: MethodDefinition) {
    const nextModelConfig = { ...schemaDefaults(method.method_schema), ...method.default_method_config };
    const nextBuilder = getMethodBuilder(method.builder_kind);
    setMethodType(method.type);
    setModelConfig(nextModelConfig);
    setTrainingConfig({ ...schemaDefaults(method.training_schema), ...method.default_training_config });
    setInferenceConfig({ ...schemaDefaults(method.inference_schema), ...method.default_inference_config });
    setModelGraph(nextBuilder?.createDefaultGraph?.(method.builder_kind, layerByType, nextModelConfig) ?? {});
    setDiagram(null);
    setDiagramError(null);
    setArchitectureCheck(null);
    setTorchCheck(null);
    setTorchCheckLogs([]);
    setLoadedMethodId(null);
    setIsEditingLoadedMethod(true);
    setNumericDrafts({});
  }

  function buildPayload(options: { diagramOnly?: boolean } = {}) {
    return buildMethodPayload(selectedMethod, modelGraph, modelConfig, trainingConfig, inferenceConfig, options);
  }

  const diagramPayload = useMemo(() => buildPayload({ diagramOnly: true }), [
    selectedMethod?.type,
    selectedMethod?.builder_kind,
    modelGraph,
    modelConfig,
  ]);
  const diagramPayloadSignature = useMemo(
    () => (diagramPayload ? JSON.stringify(diagramPayload) : ''),
    [diagramPayload],
  );
  const [debouncedDiagramPayloadSignature] = useDebouncedValue(diagramPayloadSignature, 300);

  useEffect(() => {
    setTorchCheck(null);
    setTorchCheckLogs([]);
  }, [diagramPayloadSignature]);

  useEffect(() => {
    if (!debouncedDiagramPayloadSignature) return;
    const payload = JSON.parse(debouncedDiagramPayloadSignature);
    let cancelled = false;
    buildMethodDiagram(payload)
      .then((result) => {
        if (cancelled) return;
        setDiagram(result.diagram);
        setArchitectureCheck(result);
        setDiagramError(null);
      })
      .catch((error) => {
        if (cancelled) return;
        setDiagram(null);
        setArchitectureCheck(null);
        setDiagramError(error instanceof Error ? error.message : 'Unknown error');
      });
    return () => {
      cancelled = true;
    };
  }, [debouncedDiagramPayloadSignature]);

  function loadMethodIntoBuilder(method: MethodConfiguration) {
    setLoadedMethodId(method.id);
    setName(method.name);
    setNameTouched(true);
    setDescription(method.description ?? '');
    setMethodType(method.method_type);
    setModelConfig(method.method_config ?? method.model_config);
    setTrainingConfig(method.training_config);
    setInferenceConfig(method.inference_config);
    setModelGraph(method.method_graph ?? method.model_graph ?? {});
    setDiagram(method.diagram);
    setArchitectureCheck(method.validation);
    setTorchCheck(null);
    setTorchCheckLogs([]);
    setDiagramError(null);
    setNumericDrafts({});
    setIsEditingLoadedMethod(false);
  }

  async function handleLoadMethod(methodId: number) {
    await rowActions.runPending(`load:${methodId}`, async () => {
      loadMethodIntoBuilder(await getMethodConfiguration(methodId));
    }).catch((error) => {
      notifications.show({
        color: 'red',
        title: 'Could not load method',
        message: error instanceof Error ? error.message : 'Unknown error',
      });
    });
  }

  async function handleSave(asNew: boolean) {
    const payload = buildPayload();
    if (!payload) return;
    if (invalidNumericDrafts.length > 0) {
      notifications.show({
        color: 'red',
        title: 'Fix numeric inputs before saving',
        message: invalidNumericDrafts[0][1].message ?? 'One numeric input has an invalid draft value.',
      });
      return;
    }
    const subject = selectedMethod?.training_mode === 'gradient' ? 'Architecture' : 'Method';
    setLoading(true);
    try {
      const savePayload = { ...payload, name: name.trim(), description };
      const saved =
        loadedMethodId != null && !asNew
          ? await updateMethodConfiguration(loadedMethodId, savePayload)
          : await createMethodConfiguration(savePayload);
      await refresh();
      loadMethodIntoBuilder(saved);
      notifications.show({ color: 'green', title: asNew ? `${subject} saved` : `${subject} updated`, message: saved.name });
    } catch (error) {
      notifications.show({
        color: 'red',
        title: 'Save failed',
        message: error instanceof Error ? error.message : 'Unknown error',
      });
    } finally {
      setLoading(false);
    }
  }

  const handleNumberDraftChange = useCallback((fieldId: string, state: NumericDraftState | null) => {
    setNumericDrafts((current) => {
      const next = { ...current };
      if (!state || !state.dirty) {
        delete next[fieldId];
      } else {
        next[fieldId] = state;
      }
      return next;
    });
  }, []);

  async function handleRunTorchCheck() {
    const payload = buildPayload();
    if (!payload) return;
    setTorchCheckLoading(true);
    setTorchCheck(null);
    setTorchCheckLogs(['Building encoder', 'Running dummy input', 'Checking decoder output']);
    try {
      const result = await runMethodTorchCheck(payload);
      setTorchCheck(result);
      setTorchCheckLogs(result.logs);
      notifications.show({
        color: result.valid ? 'green' : result.status === 'missing' ? 'yellow' : 'red',
        title: result.valid ? 'Torch check passed' : 'Torch check failed',
        message: result.torch_check?.message ?? result.errors[0] ?? result.warnings[0] ?? result.status,
      });
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Unknown error';
      setTorchCheck({
        valid: false,
        status: 'failed',
        errors: [message],
        warnings: [],
        logs: ['Building encoder', 'Failed'],
        torch_check: { status: 'failed', message },
      });
      setTorchCheckLogs(['Building encoder', `Failed: ${message}`]);
      notifications.show({ color: 'red', title: 'Torch check failed', message });
    } finally {
      setTorchCheckLoading(false);
    }
  }

  function handleResetCurrentMethod() {
    if (!selectedMethod) return;
    setDescription('');
    setNameTouched(false);
    resetForMethod(selectedMethod);
  }

  async function handleDelete(method: MethodConfiguration) {
    if (!window.confirm(`Delete method "${method.name}"?`)) return;
    await rowActions.runPending(`delete:${method.id}`, async () => {
      await deleteMethodConfiguration(method.id);
      await refresh();
      if (loadedMethodId === method.id) {
        setLoadedMethodId(null);
        setIsEditingLoadedMethod(true);
        setNameTouched(false);
      }
      notifications.show({ color: 'green', title: 'Method deleted', message: method.name });
    }).catch((error) => {
      notifications.show({
        color: 'red',
        title: 'Delete failed',
        message: error instanceof Error ? error.message : 'Unknown error',
      });
    });
  }

  const saveDisabled = !name.trim() || nameClash || invalidNumericDrafts.length > 0 || architectureCheck?.valid !== true;

  function renderSaveActions(showReset = false) {
    if (loadedReadOnly) {
      return (
        <Group justify="flex-end">
          {showReset && selectedMethod && (
            <Button variant="default" leftSection={<RotateCcw size={18} />} onClick={handleResetCurrentMethod}>
              {resetLabel}
            </Button>
          )}
          <Button leftSection={<Pencil size={18} />} onClick={() => setIsEditingLoadedMethod(true)} disabled={loadedMethod?.is_update_locked}>
            Edit {isTrainableArchitecture ? 'Architecture' : 'Method'}
          </Button>
        </Group>
      );
    }

    return (
      <Group justify="flex-end">
        {showReset && selectedMethod && (
          <Button variant="default" leftSection={<RotateCcw size={18} />} onClick={handleResetCurrentMethod}>
            {resetLabel}
          </Button>
        )}
        {loadedMethodId != null && (
          <Button variant="light" leftSection={<Save size={18} />} loading={loading} disabled={saveDisabled} onClick={() => handleSave(true)}>
            {saveAsNewLabel}
          </Button>
        )}
        <Button leftSection={<Save size={18} />} loading={loading} disabled={saveDisabled} onClick={() => handleSave(false)}>
          {loadedMethodId != null ? updateLabel : saveLabel}
        </Button>
      </Group>
    );
  }

  return (
    <Stack gap="lg">
      <div>
        <Title order={2}>Methods</Title>
        <Text c="dimmed" size="sm">
          Build named reusable method configurations for later training, fit, and inference workflows.
        </Text>
      </div>

      <StepCard title="Method details" color="blue">
        <TextInput
          label={isTrainableArchitecture ? 'Architecture name' : 'Method name'}
          value={name}
          disabled={loadedReadOnly}
          onChange={(event) => {
            setNameTouched(true);
            setName(event.currentTarget.value);
          }}
          error={nameClash ? 'A method with this name already exists.' : undefined}
        />
        <Textarea
          label="Description"
          value={description}
          disabled={loadedReadOnly}
          onChange={(event) => setDescription(event.currentTarget.value)}
        />
        {loadedReadOnly && (
          <Alert color={loadedMethod?.is_update_locked ? 'yellow' : 'blue'} title="Loaded read-only">
            <Stack gap={4}>
              <Text size="sm">Click Edit before changing this saved {isTrainableArchitecture ? 'architecture' : 'method'}.</Text>
              {loadedMethod?.update_lock_reasons.map((reason) => (
                <Text key={reason} size="sm">
                  {reason}
                </Text>
              ))}
            </Stack>
          </Alert>
        )}
        {nameClash && (
          <Alert color="red" title="Name already exists">
            Choose a unique name before saving.
          </Alert>
        )}
        {invalidNumericDrafts.length > 0 && (
          <Alert color="red" title="Invalid numeric input">
            {invalidNumericDrafts[0][1].message ?? 'Commit or discard the current numeric draft before saving.'}
          </Alert>
        )}
        <Group justify="space-between" align="center">
          {selectedMethod ? (
            <Badge color={architectureCheck?.valid ? 'green' : 'red'} variant="light" size="lg">
              {architectureCheck?.valid ? 'Ready to save' : 'Blocked'}
            </Badge>
          ) : (
            <Text size="sm" c="dimmed">
              Select a method type before saving.
            </Text>
          )}
          {renderSaveActions(true)}
        </Group>
      </StepCard>

      <StepCard index={1} title="Method type" color="violet">
        <Select
          label="Method type"
          placeholder="Select a method type"
          data={methodDefinitions.map((method) => ({ value: method.type, label: method.label }))}
          value={methodType}
          disabled={loadedReadOnly}
          onChange={(value) => {
            const method = value ? methodByType.get(value) : undefined;
            if (method) resetForMethod(method);
          }}
        />

        {selectedMethod ? (
          <Paper withBorder p="sm" radius="sm">
            <Group gap="xs">
              <BrainCircuit size={18} />
              <Text fw={700}>{selectedMethod.label}</Text>
            </Group>
            <Text size="sm" c="dimmed" mt={4}>
              {selectedMethod.description}
            </Text>
          </Paper>
        ) : (
          <Alert color="blue">Select the method type before configuring architecture parameters.</Alert>
        )}
      </StepCard>

      <StepCard index={2} title={builderTitle} color="teal">
          {!selectedMethod && <Alert color="blue">Select a method type to configure its builder.</Alert>}
          {selectedMethod && !BuilderComponent && (
            <Alert color="red" title="Unsupported method builder">
              No frontend builder is registered for builder_kind "{selectedMethod.builder_kind}".
            </Alert>
          )}
          {selectedMethod && BuilderComponent && (
            <BuilderComponent
              method={selectedMethod}
              modelConfig={modelConfig}
              modelGraph={modelGraph}
              layers={layers}
              validation={architectureCheck}
              disabled={loadedReadOnly}
              onConfigChange={(key, value) => setModelConfig((current) => ({ ...current, [key]: value }))}
              onGraphChange={setModelGraph}
              onNumberDraftChange={handleNumberDraftChange}
            />
          )}
      </StepCard>

      <StepCard index={3} title="Validate" color="grape">
          <ArchitectureCheckPanel validation={architectureCheck} modelConfig={modelConfig} />
          {selectedMethod?.builder_kind !== 'form' && (
            <TorchCheckPanel
              validation={architectureCheck}
              torchCheck={torchCheck}
              loading={torchCheckLoading}
              logs={torchCheckLogs}
              onRun={handleRunTorchCheck}
            />
          )}
          <Title order={3}>Diagram</Title>
          <MethodDiagramPanel diagram={diagram} error={diagramError} />
      </StepCard>

      <SavedMethodsTable
        methods={methods}
        methodByType={methodByType}
        onLoad={handleLoadMethod}
        onDelete={handleDelete}
        isLoading={(methodId) => rowActions.isPending(`load:${methodId}`)}
        isDeleting={(methodId) => rowActions.isPending(`delete:${methodId}`)}
      />
    </Stack>
  );
}

export const ModelsPage = MethodsPage;
