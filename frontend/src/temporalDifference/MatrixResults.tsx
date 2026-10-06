import { Alert, Button, Group, Loader, MultiSelect, NumberInput, Paper, SegmentedControl, Select, Stack, Switch, Text, Title } from '@mantine/core';
import { useEffect, useRef, useState } from 'react';
import { createTemporalMatrix, getTemporalMatrix, getTemporalMatrixCandidates, temporalMatrixUrl } from '../api';
import type { MatrixConfig, MatrixState, Role, Run } from './types';
import { displayTime, labels } from './helpers';

export function MatrixResults({run, projectId}: {run: Run; projectId: string}) {
  const [role, setRole] = useState<Role>('reference');
  return <Paper withBorder p="lg"><Stack>
    <Title order={3}>Bildvergleich</Title>
    <SegmentedControl value={role} onChange={value => setRole(value as Role)} data={Object.entries(labels).map(([value,label]) => ({value,label}))} />
    <MatrixEditor key={`${projectId}-${run.id}-${role}`} run={run} projectId={projectId} role={role} />
  </Stack></Paper>;
}
function MatrixEditor({run, projectId, role}: {run: Run; projectId: string; role: Role}) {
  const [config, setConfig] = useState<MatrixConfig>({role, deltas_seconds:run.config.deltas_seconds.slice(0,3), start_times:[], top_percent:null});
  const [availableDeltas, setAvailableDeltas] = useState(run.config.deltas_seconds);
  const [saved, setSaved] = useState<MatrixState | null>(null);
  const [ready, setReady] = useState(false), [busy,setBusy] = useState(false), [picking,setPicking] = useState(false);
  const [choices,setChoices] = useState<string[]>([]), [error,setError] = useState<string|null>(null);
  const [percent,setPercent] = useState<number|string>(1);
  const alive = useRef(true);
  useEffect(() => { alive.current=true; let current=true;
    getTemporalMatrix(run.id,role,projectId).then(value => {if(current){setSaved(value);setAvailableDeltas(value.available_deltas ?? run.config.deltas_seconds);if(!value.config)setConfig(c=>({...c,deltas_seconds:(value.available_deltas ?? run.config.deltas_seconds).slice(0,3)}));if(value.config){setConfig(value.config);setPercent(value.config.top_percent ?? 1);}setReady(true);}})
      .catch(reason=>{if(current)setError(String(reason));});
    return ()=>{current=false;alive.current=false;};
  },[run.id,role,projectId]);
  const deltas = JSON.stringify(config.deltas_seconds);
  useEffect(()=>{
    if(!ready) return;
    let current=true;setPicking(true);setChoices([]);
    if(!config.deltas_seconds.length){setPicking(false);return;}
    getTemporalMatrixCandidates(run.id,role,config.deltas_seconds,projectId).then(value=>{
      if(!current)return;
      setChoices(value.start_times);
      setConfig(c=>({...c,start_times:c.start_times.length === Math.min(3,value.start_times.length) && c.start_times.every(t=>value.start_times.includes(t)) ? c.start_times : value.suggested}));
    }).catch(reason=>{if(current)setError(String(reason));}).finally(()=>{if(current)setPicking(false);});
    return ()=>{current=false;};
  },[ready,deltas,run.id,role,projectId]);
  async function generate(){
    setBusy(true);setError(null);
    try{const next=await createTemporalMatrix(run.id,config,projectId);if(alive.current)setSaved(next);}
    catch(reason){if(alive.current)setError(String(reason));}
    finally{if(alive.current)setBusy(false);}
  }
  const valid = ready && !picking && config.deltas_seconds.length>0 && config.start_times.length>0 && config.start_times.every(t=>choices.includes(t))
    && (config.top_percent===null || (Number.isFinite(config.top_percent) && config.top_percent>=.01 && config.top_percent<=100));
  return <Stack>
    {error && <Alert color="red">{error}<Button variant="subtle" onClick={()=>{setError(null);setReady(false);getTemporalMatrix(run.id,role,projectId).then(value=>{if(alive.current){setSaved(value);setReady(true);}}).catch(reason=>{if(alive.current)setError(String(reason));});}}>Erneut laden</Button></Alert>}
    {!ready && !error && <Loader size="sm" />}
    <MultiSelect label="Zeitabstände für Bildvergleich (maximal 3)" data={availableDeltas.map(d=>({value:String(d),label:`${d} s`}))} value={config.deltas_seconds.map(String)} maxValues={3} disabled={!ready||busy} onChange={values=>setConfig(c=>({...c,deltas_seconds:values.map(Number).sort((a,b)=>a-b)}))} />
    {picking ? <Loader size="sm" /> : config.start_times.map((time,index)=><Select key={index} label={`Startpunkt ${index+1}`} searchable allowDeselect={false} value={time} disabled={busy||!config.deltas_seconds.length}
      data={choices.map(t=>({value:t,label:displayTime(t),disabled:t!==time && config.start_times.includes(t)}))}
      onChange={value=>value && setConfig(c=>({...c,start_times:c.start_times.map((t,i)=>i===index?value:t)}))} />)}
    {ready && !picking && choices.length<3 && <Text size="sm">{choices.length} geeignete Startpunkte verfügbar. Es werden höchstens drei Zeilen angezeigt.</Text>}
    <Switch label="Nur stärkste Änderungen anzeigen" disabled={busy} checked={config.top_percent!==null} onChange={event=>{const checked=event.currentTarget.checked;setConfig(c=>({...c,top_percent:checked ? Number(percent) : null}));}} />
    {config.top_percent!==null && <NumberInput label="Stärkste Änderungen (%)" min={.01} max={100} decimalScale={2} value={percent} disabled={busy} onChange={value=>{setPercent(value);setConfig(c=>({...c,top_percent:value===''?NaN:Number(value)}));}} />}
    <Text size="sm" c="dimmed">Jede Differenz bezieht sich auf das Bild bei t. Der Filter gilt je Karte; gleiche Werte an der Schwelle bleiben gemeinsam sichtbar. Daher können mehr Pixel als der gewählte Anteil sichtbar sein. Die Farbskala und die Analysewerte bleiben unverändert. Tatsächliche Partnerzeiten können bis zu 0,5 s vom Soll abweichen.</Text>
    <Text size="sm" c="dimmed">Pixeländerungen in Prozent des festen 16-Bit-Wertebereichs: 100 × |Differenz| / 65535.</Text>
    <Button disabled={!valid||busy} loading={busy} onClick={()=>void generate()}>Plot erstellen</Button>
    {saved?.artifact && <>
      {saved.unit !== 'percent' && <Alert color="orange">Dieser gespeicherte Plot verwendet noch Pipeline-Einheiten und die bisherigen Bezeichnungen. „Plot erstellen“ erzeugt die Prozentdarstellung mit Normal/Anomalie.</Alert>}
      <Text fw={600}>Zuletzt erstellter Plot · {labels[role]} · {saved.config?.top_percent == null ? 'Alle Änderungen' : `Top ${saved.config.top_percent} %`}</Text>
      <img alt={`Bildvergleich ${labels[role]}`} src={temporalMatrixUrl(run.id,role,saved.artifact,projectId)} style={{width:'100%',height:'auto'}} />
      {saved.warnings.map(w=><Text key={w} size="sm" c="dimmed">{w}</Text>)}
      <Group><Button component="a" href={temporalMatrixUrl(run.id,role,saved.artifact,projectId,true)}>Bildvergleich als PNG herunterladen</Button></Group>
    </>}
  </Stack>;
}
