"""Shared window training/evaluation. Test data is only visited after checkpoint selection."""
import hashlib
import math
import random
import time

import numpy as np
import torch

from app.time_series.networks import build_model, errors, reconstruction_loss, usad_losses, representation_metadata
from app.time_series.training_service import write_json, write_npz


class Cancelled(Exception):
    pass


def execute(directory, snapshot, progress=lambda **kwargs: None):
    torch.set_num_threads(2)
    cfg, kind = snapshot['training'], snapshot['model']['kind']
    seed = cfg['seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with np.load(directory / 'input.npz') as data:
        scaled, groups, endpoints = data['scaled'], data['groups'], data['endpoints']
    length, channels = snapshot['window_length'], scaled.shape[1]
    model = build_model(kind, length, channels, snapshot['model']['config']).to(device)
    train = endpoints[groups[endpoints] == 0]
    validation = endpoints[groups[endpoints] == 1]
    batch_size = cfg['batch_size']
    offsets = np.arange(length - 1, -1, -1)
    def batches(indices):
        for start in range(0, len(indices), batch_size):
            progress()
            ends = indices[start:start + batch_size]
            yield ends, torch.from_numpy(scaled[ends[:, None] - offsets]).to(device)
    if kind == 'usad':
        optimizers = [torch.optim.Adam(list(model.encoder.parameters()) + list(decoder.parameters()), lr=cfg['lr'])
                      for decoder in (model.decoder1, model.decoder2)]
    else:
        optimizers = [torch.optim.Adam(model.parameters(), lr=cfg['lr'], amsgrad=cfg.get('amsgrad', False))]
    best, selected_epoch, stale = math.inf, 0, 0
    selected_metric = None
    progress(current_step='training')
    for epoch in range(1, cfg['epochs'] + 1):
        model.train()
        order = np.random.permutation(train) if cfg['shuffle'] else train
        total = 0.0
        for _, x in batches(order):
            if kind == 'usad':
                losses = []
                for index, optimizer in enumerate(optimizers):
                    model.zero_grad(set_to_none=True)
                    loss = usad_losses(x, model(x), epoch)[index]
                    if not torch.isfinite(loss):
                        raise ValueError('Nichtendlicher Trainingsloss.')
                    loss.backward()
                    optimizer.step()
                    losses.append(loss.item())
                batch_loss = sum(losses) / 2
            else:
                optimizers[0].zero_grad(set_to_none=True)
                inputs = x + torch.randn_like(x) * cfg.get('noise_std', 0) if cfg.get('noise_std', 0) else x
                loss = reconstruction_loss(kind, x, model(inputs, sample=kind == 'lstm_vae'), cfg)
                if not torch.isfinite(loss):
                    raise ValueError('Nichtendlicher Trainingsloss.')
                loss.backward()
                optimizers[0].step()
                batch_loss = loss.item()
            total += batch_loss * len(x)
        model.eval()
        val_loss = None
        if len(validation):
            val_total = 0.0
            with torch.no_grad():
                for _, x in batches(validation):
                    val_total += reconstruction_loss(kind, x, model(x), cfg).item() * len(x)
            val_loss = val_total / len(validation)
            if not math.isfinite(val_loss):
                raise ValueError('Nichtendliche Validation-Metrik.')
        improved = val_loss is None or val_loss < best
        if improved:
            selected_epoch, selected_metric, stale = epoch, val_loss, 0
            best = val_loss if val_loss is not None else math.inf
            temporary = directory / 'weights.pt.part'
            torch.save({key: value.detach().cpu() for key, value in model.state_dict().items()}, temporary)
            temporary.replace(directory / 'weights.pt')
        else:
            stale += 1
        progress(epoch=epoch, train_loss=total / len(train), val_loss=val_loss,
                 selected_epoch=selected_epoch, selected_metric=selected_metric, current_step='training')
        print(f'Epoch {epoch}: train={total / len(train):.7g} validation={val_loss}', flush=True)
        if len(validation) and cfg['early_stopping'] and stale >= cfg['patience']:
            break
    model.load_state_dict(torch.load(directory / 'weights.pt', map_location=device, weights_only=True))
    model.eval()
    checkpoint = dict(snapshot['checkpoint_metric'], epoch=selected_epoch, metric_value=selected_metric,
                      sha256=hashlib.sha256((directory / 'weights.pt').read_bytes()).hexdigest())
    manifest = dict(version=1, runtime=dict(torch_version=str(torch.__version__), numpy_version=np.__version__, device=str(device), seed=seed), architecture_version=snapshot['model']['version'], checkpoint=checkpoint,
                    representation=representation_metadata(kind, length, snapshot['model']['config']),
                    sensor_order=snapshot['preview']['columns'], input_fingerprint=snapshot['preview']['input_fingerprint'],
                    window_length=length, score_space='train_minmax', reconstruction_readout='last_window_position', chunks=[])
    processed = 0
    with torch.no_grad():
        for chunk_index, start in enumerate(range(0, len(endpoints), 1024)):
            arrays = {}
            for ends, x in batches(endpoints[start:start + 1024]):
                output = model(x)
                error = errors(kind, x, output, cfg)
                values = dict(end_indices=ends, window_score=error.mean((1, 2)).cpu().numpy(),
                              endpoint_score=error[:, -1].mean(1).cpu().numpy(),
                              reconstruction=output['reconstruction'][:, -1].cpu().numpy(), latent=output['latent'].cpu().numpy())
                for name in ('cascade', 'second', 'variance'):
                    if name in output:
                        values[name] = output[name][:, -1].cpu().numpy()
                for key, value in values.items():
                    if not np.isfinite(value).all():
                        raise ValueError(f'Nichtendliche Ergebnisse: {key}.')
                    arrays.setdefault(key, []).append(value)
            filename = f'results-{chunk_index:05d}.npz'
            write_npz(directory / filename, **{key: np.concatenate(value) for key, value in arrays.items()})
            count = sum(len(v) for v in arrays['end_indices'])
            manifest['chunks'].append(dict(file=filename, count=count))
            processed += count
            progress(current_step='evaluation', processed_windows=processed)
    write_json(directory / 'manifest.json', manifest)
    return manifest
