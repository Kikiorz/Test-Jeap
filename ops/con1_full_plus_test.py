import json
from pathlib import Path
import tempfile
import unittest
from contextlib import closing

from con1_eval_dashboard import snapshot
from paper_con1_protocol import CATEGORIES
from run_con1_full_plus import EXPECTED, build_manifest
from con1_dynamic_queue import PRIORITY, make_batches, create_queue, claim, finish, queue_status, collect, export_canonical, effective_limits, reprioritize_pending, DIFFICULTY_ORDER


class FullPlusTest(unittest.TestCase):
    def fixture(self):
        return {s: [dict(id=i+1, name=f'{s}_{i}', category=CATEGORIES[i % 7], difficulty_level=None)
                    for i in range(n)] for s, n in EXPECTED.items()}

    def test_complete_inventory(self):
        m = build_manifest(self.fixture())
        self.assertEqual(sum(map(len, m['final'].values())), 10030)
        self.assertNotIn('monitor', m)
        self.assertEqual(m['final_episode_seed'], 743)
        for suite, rows in m['final'].items():
            self.assertEqual([r['task_id'] for r in rows], list(range(EXPECTED[suite])))

    def test_reject_missing_and_duplicate(self):
        d = self.fixture()
        d['libero_10'].pop()
        with self.assertRaises(ValueError):
            build_manifest(d)
        d = self.fixture()
        d['libero_10'][1]['id'] = 1
        with self.assertRaises(ValueError):
            build_manifest(d)

    def test_dashboard_waiting_and_aggregate(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.assertIsNone(snapshot(root)['models']['15k']['success_rate'])
            (root/'10k').mkdir()
            group = dict(count=3, success=1, failure=1, error=1)
            (root/'10k/progress.json').write_text(json.dumps(dict(completed=3, successes=1,
                groups={'libero_goal/Sensor Noise': group})))
            state = snapshot(root)
            self.assertEqual(state['models']['10k']['errors'], 1)
            self.assertEqual(state['models']['10k']['categories']['Sensor Noise'], group)
            self.assertAlmostEqual(state['models']['10k']['success_rate'], 1/3)
            self.assertFalse(state['models']['10k']['complete'])

    def test_dynamic_priority_exact_cover_and_resume(self):
        manifest = build_manifest(self.fixture())
        done = {('libero_10', 0, 0): {'status': 'failure'}}
        batches = make_batches(manifest, done)
        keys = [(s, i) for s, c, ids in batches for i in ids]
        self.assertEqual(len(keys), 10029)
        self.assertEqual(len(set(keys)), 10029)
        self.assertNotIn(('libero_10', 0), keys)
        flags = [c not in PRIORITY for s, c, ids in batches]
        self.assertEqual(flags, sorted(flags))
        self.assertTrue(all(0 < len(ids) <= 32 for s,c,ids in batches))

    def test_concurrent_claims_disjoint(self):
        from concurrent.futures import ThreadPoolExecutor
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder)/'queue.sqlite'
            batches = [('suite', 'Camera Viewpoints', [i]) for i in range(100)]
            create_queue(p, batches)
            def consume(gpu):
                ids = []
                while batch := claim(p, gpu):
                    ids.append(batch[0])
                    finish(p, batch[0])
                return ids
            with ThreadPoolExecutor(max_workers=4) as pool:
                result = list(pool.map(consume, range(4)))
            self.assertEqual(sorted(i for worker in result for i in worker), list(range(100)))
            self.assertEqual(queue_status(p), [])

    def test_camera_first_only_changes_pending(self):
        import sqlite3
        m = build_manifest(self.fixture())
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'queue.sqlite'
            create_queue(p,make_batches(m,{}))
            running=claim(p,0)
            before=queue_status(p)
            report=reprioritize_pending(p,m,category_first='Camera Viewpoints')
            self.assertEqual(before,queue_status(p))
            with closing(sqlite3.connect(p)) as db:
                rows=db.execute("SELECT suite,category,tasks FROM batches WHERE state='pending' ORDER BY id").fetchall()
            flags=[c!='Camera Viewpoints' for s,c,ids in rows]
            self.assertEqual(flags,sorted(flags))
            self.assertTrue(all(len(json.loads(ids))<=4 for s,c,ids in rows if c=='Camera Viewpoints'))
            keys=[(s,i) for s,c,ids in rows for i in json.loads(ids)]
            self.assertEqual(len(keys),len(set(keys)))
            allkeys={(s,r['task_id']) for s,values in m['final'].items() for r in values}
            self.assertEqual(set(keys),allkeys-{(running[1],i) for i in running[3]})
            self.assertEqual(report['pending_tasks'],len(keys))

    def test_difficulty_order_and_live_pending_replacement(self):
        import sqlite3
        m = build_manifest(self.fixture())
        for rows in m['final'].values():
            for i,r in enumerate(rows):
                r['difficulty_level'] = DIFFICULTY_ORDER[i % 6]
        lookup = {(s,r['task_id']):r for s,rows in m['final'].items() for r in rows}
        batches = make_batches(m,{})
        ranks = [DIFFICULTY_ORDER.index(lookup[s,i]['difficulty_level']) for s,c,ids in batches for i in ids]
        self.assertEqual(ranks,sorted(ranks))
        for s,c,ids in batches:
            self.assertEqual(len({lookup[s,i]['difficulty_level'] for i in ids}),1)
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'queue.sqlite'
            create_queue(p,list(reversed(batches)))
            done=claim(p,0); finish(p,done[0]); running=claim(p,1)
            with closing(sqlite3.connect(p)) as db:
                before=db.execute("SELECT * FROM batches WHERE state!='pending' ORDER BY id").fetchall()
            report=reprioritize_pending(p,m)
            with closing(sqlite3.connect(p)) as db:
                self.assertEqual(before,db.execute("SELECT * FROM batches WHERE state!='pending' ORDER BY id").fetchall())
            keys=[]; ranks=[]
            while (row := claim(p,2)) is not None:
                _,s,c,ids=row
                keys.extend((s,i) for i in ids)
                ranks.extend(DIFFICULTY_ORDER.index(lookup[s,i]['difficulty_level']) for i in ids)
                finish(p,row[0])
            self.assertEqual(ranks,sorted(ranks))
            self.assertEqual(len(keys),report['pending_tasks'])
            self.assertEqual(len(keys),len(set(keys)))
            excluded={(r[1],i) for r in [done,running] for i in r[3]}
            self.assertEqual(set(keys),set(lookup)-excluded)

    def test_old_new_journals_deduplicated_and_contract_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'journals').mkdir()
            target = dict(task_id=0, name='task', category='Sensor Noise', difficulty_level=3)
            manifest = dict(final={'libero_goal':[target]}, final_episode_seed=743)
            header = dict(record_type='run', run_fingerprint='same', run_config=dict(
                run_id='10k', task_suite_name='libero_goal', benchmark_mode='plus', seed=743,
                explicit_task_ids=[0], replan_steps=5))
            row = dict(record_type='episode', run_fingerprint='same', task_suite_name='libero_goal',
                task_id=0, episode_idx=0, seed=743, episode_seed=123, task_name='task',
                category='Sensor Noise', difficulty_level=3, status='success', max_steps=300,
                task_description='task', num_steps=100)
            for name in ['plus_libero_goal', 'plus_libero_goal_gpu0_dynamic']:
                (root/'journals'/f'{name}.jsonl').write_text(json.dumps(header)+'\n'+json.dumps(row)+'\n')
            records,headers = collect(root,manifest)
            self.assertEqual(len(records),1)
            export_canonical(root,records,headers)
            self.assertEqual(len((root/'canonical/plus_libero_goal.jsonl').read_text().splitlines()),2)
            header['run_config']['replan_steps']=10
            (root/'journals/plus_libero_goal_gpu0_dynamic.jsonl').write_text(json.dumps(header)+'\n'+json.dumps(row)+'\n')
            with self.assertRaises(ValueError):
                collect(root,manifest)

    def test_same_gpu_sixteen_slots_claim_unique_batches(self):
        from concurrent.futures import ThreadPoolExecutor
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'queue.sqlite'
            create_queue(p,[('libero_goal','Camera Viewpoints',[i]) for i in range(16)])
            with ThreadPoolExecutor(max_workers=16) as pool:
                values=list(pool.map(lambda slot: claim(p,0,slot),range(16)))
            self.assertEqual(len({v[0] for v in values}),16)
            self.assertEqual({r['slot'] for r in queue_status(p)},set(range(16)))

    def test_hot_concurrency_and_trial_expiry(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'limits.json'
            p.write_text(json.dumps(dict(slots_per_gpu=[2,2,2,2],trial=dict(gpu=0,slots=4,until_unix=100))))
            self.assertEqual(effective_limits(p,now=99),[4,2,2,2])
            self.assertEqual(effective_limits(p,now=101),[2]*4)
            p.write_text(json.dumps(dict(slots_per_gpu=[4]*4)))
            self.assertEqual(effective_limits(p),[4]*4)
            p.write_text(json.dumps(dict(slots_per_gpu=[8]*4)))
            self.assertEqual(effective_limits(p),[8]*4)
            p.write_text(json.dumps(dict(slots_per_gpu=[16,8,8,8])))
            self.assertEqual(effective_limits(p),[16,8,8,8])
            p.write_text(json.dumps(dict(slots_per_gpu=[17]*4)))
            with self.assertRaises(ValueError):
                effective_limits(p)


if __name__ == '__main__':
    unittest.main()
