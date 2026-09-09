"""CPU-only dynamic queue and canonical, worker-independent result accounting."""
from collections import defaultdict
from contextlib import closing
import json
from pathlib import Path
import sqlite3
import time

PRIORITY = ('Camera Viewpoints', 'Robot Initial States', 'Sensor Noise', 'Objects Layout')
OTHER = ('Background Textures', 'Language Instructions', 'Light Conditions')
SUITES = ('libero_10', 'libero_goal', 'libero_object', 'libero_spatial')
DIFFICULTY_ORDER = (5, 4, 3, 2, 1, None)


def collect(folder, manifest):
    """Old fixed-suite and new worker journals; count each episode once, validate identities."""
    records, headers, contracts = {}, {}, {}
    expected = {(s, r['task_id']): r for s, rows in manifest['final'].items() for r in rows}
    for path in sorted((folder / 'journals').glob('*.jsonl')):
        lines = path.read_text().splitlines()
        header = None
        for i, line in enumerate(lines):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                if i == len(lines) - 1:
                    continue  # A live writer may be appending its final line.
                raise
            if row.get('record_type') == 'run':
                header = row
                config = dict(row['run_config'])
                config.pop('run_id', None)
                suite = config['task_suite_name']
                if config['benchmark_mode'] != 'plus' or config['seed'] != manifest['final_episode_seed']:
                    raise ValueError(f'Unexpected evaluation contract: {path}')
                if config.get('explicit_task_ids') != [r['task_id'] for r in manifest['final'][suite]]:
                    raise ValueError(f'Incomplete full-suite contract: {path}')
                if suite in contracts and contracts[suite] != config:
                    raise ValueError(f'Worker evaluation contract mismatch: {path}')
                contracts[suite], headers[suite] = config, header
            elif row.get('record_type') == 'episode':
                if header is None or row['run_fingerprint'] != header['run_fingerprint']:
                    raise ValueError(f'Episode/header mismatch: {path}')
                key = (row['task_suite_name'], row['task_id'], row['episode_idx'])
                target = expected[key[:2]]
                if key[2] != 0 or row['seed'] != manifest['final_episode_seed']:
                    raise ValueError(f'Episode index/seed mismatch: {key}')
                for field, name in [('task_name', 'name'), ('category', 'category'), ('difficulty_level', 'difficulty_level')]:
                    if row[field] != target[name]:
                        raise ValueError(f'Episode metadata mismatch: {key} / {field}')
                previous = records.get(key)
                if previous:
                    for field in ('episode_seed', 'max_steps', 'task_description'):
                        if previous[field] != row[field]:
                            raise ValueError(f'Repeated episode identity mismatch: {key}')
                    if previous['status'] != 'error' and row['status'] != 'error' and previous['status'] != row['status']:
                        raise ValueError(f'Conflicting completed outcomes: {key}')
                    if previous['status'] != 'error':
                        continue
                records[key] = row
    return records, headers


def totals(records):
    groups = defaultdict(lambda: dict(count=0, success=0, failure=0, error=0, steps_sum=0))
    for row in records.values():
        g = groups[f"{row['task_suite_name']}/{row['category']}"]
        if row['status'] not in ('success', 'failure', 'error'):
            raise ValueError('Unexpected outcome')
        g['count'] += 1
        g[row['status']] += 1
        g['steps_sum'] += row['num_steps']
    for g in groups.values():
        g['success_rate'] = g['success'] / g['count']
        g['mean_steps'] = g['steps_sum'] / g['count']
    return dict(groups)


def make_batches(manifest, records, size=32):
    """Hardest first; within each level retain the existing category/suite ordering."""
    rows = [r for values in manifest['final'].values() for r in values]
    if any(r.get('difficulty_level') not in DIFFICULTY_ORDER for r in rows):
        raise ValueError('Unexpected difficulty level')
    result = []
    for level in DIFFICULTY_ORDER:
        subset = dict(final={s: [r for r in manifest['final'].get(s, [])
                                 if r.get('difficulty_level') == level] for s in SUITES})
        result.extend(_category_batches(subset, records, size))
    return result


def _category_batches(manifest, records, size=32):
    if size < 1:
        raise ValueError('Batch size must be positive')
    done = {key[:2] for key, row in records.items() if row['status'] in ('success', 'failure')}
    buckets = {}
    for suite in SUITES:
        for category in PRIORITY + OTHER:
            ids = [r['task_id'] for r in manifest['final'][suite]
                   if r['category'] == category and (suite, r['task_id']) not in done]
            buckets[suite, category] = [ids[i:i+size] for i in range(0, len(ids), size)]
    result = []
    for categories in (PRIORITY, OTHER):
        rounds = max((len(buckets[s, c]) for s in SUITES for c in categories), default=0)
        for index in range(rounds):
            for category in categories:
                for suite in SUITES:
                    batches = buckets[suite, category]
                    if index < len(batches):
                        result.append((suite, category, batches[index]))
    return result


def reprioritize_pending(path, manifest, category_first=None):
    """Atomically regroup only unclaimed work; running/done rows and journals stay untouched.

    Callers must back up the live SQLite database before invoking this operation.
    Existing workers already claim ORDER BY id and need no restart.
    """
    if category_first is not None and category_first not in PRIORITY + OTHER:
        raise ValueError('Unknown priority category')
    expected = {(s, r['task_id']): r for s, rows in manifest['final'].items() for r in rows}
    with closing(sqlite3.connect(path, timeout=60)) as db, db:
        db.execute('BEGIN IMMEDIATE')
        pending = db.execute("SELECT id,suite,category,tasks FROM batches WHERE state='pending' ORDER BY id").fetchall()
        keys = []
        for _, suite, category, ids in pending:
            for task_id in json.loads(ids):
                key = (suite, task_id)
                if expected[key]['category'] != category:
                    raise ValueError('Pending task category mismatch')
                keys.append(key)
        if len(keys) != len(set(keys)):
            raise ValueError('Duplicate pending tasks')
        selected = set(keys)
        for suite, ids in db.execute("SELECT suite,tasks FROM batches WHERE state!='pending'"):
            if any((suite, task_id) in selected for task_id in json.loads(ids)):
                raise ValueError('Pending/running or completed task overlap')
        subset = dict(final={s: [r for r in manifest['final'].get(s, [])
                                 if (s, r['task_id']) in selected] for s in SUITES})
        batches = make_batches(subset, {})
        if category_first is not None:
            # Short priority batches distribute the final category tasks across free workers.
            first = [(s, c, ids[i:i+4]) for s,c,ids in batches if c == category_first
                     for i in range(0, len(ids), 4)]
            batches = first + [b for b in batches if b[1] != category_first]
        reordered = [(s, i) for s, _, ids in batches for i in ids]
        if len(reordered) != len(keys) or set(reordered) != selected:
            raise ValueError('Reordering changed task coverage')
        start = db.execute('SELECT COALESCE(MAX(id),-1)+1 FROM batches').fetchone()[0]
        db.execute("DELETE FROM batches WHERE state='pending'")
        db.executemany('INSERT INTO batches VALUES (?, ?, ?, ?, ?, NULL, NULL)',
                       [(start+i, s, c, json.dumps(ids), 'pending') for i, (s,c,ids) in enumerate(batches)])
        return dict(pending_tasks=len(keys), old_batches=len(pending), new_batches=len(batches),
                    category_first=category_first,
                    category_first_tasks=sum(expected[k]['category'] == category_first for k in keys),
                    difficulty_order=list(DIFFICULTY_ORDER), first_batch_id=start,
                    pending_by_level={str(level): sum(expected[k].get('difficulty_level') == level
                                                     for k in keys) for level in DIFFICULTY_ORDER})


def create_queue(path, batches):
    db = sqlite3.connect(path)
    db.execute('CREATE TABLE batches (id INTEGER PRIMARY KEY, suite TEXT, category TEXT, tasks TEXT, state TEXT, gpu INTEGER, slot INTEGER)')
    db.executemany('INSERT INTO batches VALUES (?, ?, ?, ?, ?, NULL, NULL)',
                   [(i, s, c, json.dumps(ids), 'pending') for i, (s, c, ids) in enumerate(batches)])
    db.commit()
    db.close()


def claim(path, gpu, slot=0):
    with closing(sqlite3.connect(path, timeout=60)) as db, db:
        db.execute('BEGIN IMMEDIATE')
        row = db.execute("SELECT id,suite,category,tasks FROM batches WHERE state='pending' ORDER BY id LIMIT 1").fetchone()
        if row:
            db.execute("UPDATE batches SET state='running',gpu=?,slot=? WHERE id=?", (gpu, slot, row[0]))
    return (row[0], row[1], row[2], json.loads(row[3])) if row else None


def finish(path, batch_id):
    with closing(sqlite3.connect(path, timeout=60)) as db, db:
        db.execute("UPDATE batches SET state='done' WHERE id=?", (batch_id,))


def queue_status(path):
    with closing(sqlite3.connect(path, timeout=60)) as db, db:
        return [dict(gpu=gpu, slot=slot, batch_id=i, suite=s, category=c, tasks=len(json.loads(ids)))
                for i, s, c, ids, gpu, slot in db.execute("SELECT id,suite,category,tasks,gpu,slot FROM batches WHERE state='running'")]


def effective_limits(path, now=None):
    """Hot-adjust renderer concurrency; an expired canary drains at batch boundaries."""
    config = json.loads(Path(path).read_text())
    limits = list(config['slots_per_gpu'])
    if len(limits) != 4 or any(type(n) is not int or not 1 <= n <= 16 for n in limits):
        raise ValueError('slots_per_gpu must contain four integers in [1, 16]')
    trial = config.get('trial')
    if trial and (time.time() if now is None else now) < trial['until_unix']:
        gpu, slots = trial['gpu'], trial['slots']
        if type(gpu) is not int or gpu not in range(4) or type(slots) is not int or not 1 <= slots <= 16:
            raise ValueError('Invalid concurrency trial')
        limits[gpu] = slots
    return limits


def has_pending(path):
    with closing(sqlite3.connect(path, timeout=60)) as db:
        return db.execute("SELECT 1 FROM batches WHERE state='pending' LIMIT 1").fetchone() is not None


def export_canonical(folder, records, headers):
    output = folder / 'canonical'
    output.mkdir(exist_ok=True)
    for suite, header in headers.items():
        rows = [header]
        for key, record in sorted(records.items()):
            if key[0] == suite:
                rows.append(dict(record, run_fingerprint=header['run_fingerprint']))
        path = output / f'plus_{suite}.jsonl'
        temp = path.with_suffix('.tmp')
        temp.write_text(''.join(json.dumps(r) + '\n' for r in rows))
        temp.replace(path)
