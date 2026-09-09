import hashlib
from pathlib import Path
import tempfile
import unittest
from run_pure_jepawam_robot_sweep import ORDER, PREFIXES, evaluation_command, inference_file, verify_file, paired, summarize


class RobotSweepTest(unittest.TestCase):
    def test_only_author_jepa_variants_and_order(self):
        self.assertEqual(ORDER, ('50k','40k','30k'))
        self.assertTrue(PREFIXES['50k'].endswith('/50000'))
        self.assertTrue(PREFIXES['40k'].endswith('/40000'))
        self.assertTrue(PREFIXES['30k'].endswith('/29999'))
        for prefix in PREFIXES.values():
            self.assertIn('pi05_libero_vjepa_aux/', prefix)
            self.assertTrue(inference_file(prefix+'/params/ocdbt.process_0/d/a', prefix))
            self.assertFalse(inference_file(prefix+'/train_state/d/a', prefix))
            self.assertFalse(inference_file(prefix+'0/params/a', prefix))

    def test_isolated_robot_commands(self):
        commands = [evaluation_command(label) for label in ORDER]
        self.assertEqual(len({c[c.index('--output')+1] for c in commands}),3)
        for c in commands:
            self.assertEqual(c[c.index('--category')+1], 'Robot Initial States')
            self.assertEqual(c[c.index('--config')+1], 'pi05_libero_paper_reference')
            self.assertNotIn('--rapr-runtime-gate', c)

    def test_official_checksum_and_partial_file_rejection(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'chunk'
            path.write_bytes(b'test')
            verify_file(path, dict(size=4,lfs=dict(oid=hashlib.sha256(b'test').hexdigest())))
            verify_file(path, dict(size=4,oid=hashlib.sha1(b'blob 4\0test').hexdigest()))
            with self.assertRaises(ValueError):
                verify_file(path,dict(size=5,oid='invalid'))
            with self.assertRaises(ValueError):
                verify_file(path,dict(size=4,lfs=dict(oid='invalid')))

    def test_paired_identity_and_outcomes(self):
        row = dict(seed=743,episode_seed=123,max_steps=300,task_name='test',
                   category='Robot Initial States',difficulty_level=5,task_suite_name='libero_goal',status='success')
        reference = {('libero_goal',0,0):dict(row,status='failure')}
        rows = {('libero_goal',0,0):row}
        self.assertEqual(paired(rows,reference), dict(wins=1,losses=0))
        self.assertEqual(summarize(rows)['by_level']['5']['success_rate'],1)
        reference[('libero_goal',0,0)]['episode_seed'] = 124
        with self.assertRaises(ValueError):
            paired(rows,reference)


if __name__=='__main__':
    unittest.main()
