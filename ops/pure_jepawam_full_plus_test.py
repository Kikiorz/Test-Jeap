import unittest
from run_con1_dynamic_eval import policy_command, selected_batches, selected_records, category_suffix
from run_pure_jepawam_full_plus import audit_metadata


class PureBaselineTest(unittest.TestCase):
    def test_robot_only_reuses_completed_and_preserves_other_results(self):
        manifest = dict(final={'libero_goal':[
            dict(task_id=i, category='Robot Initial States' if i<10 else 'Sensor Noise',
                 difficulty_level=5) for i in range(12)]})
        records = {('libero_goal',0,0):dict(status='failure', category='Robot Initial States'),
                   ('libero_goal',10,0):dict(status='success', category='Sensor Noise')}
        batches = selected_batches(manifest, records, 'Robot Initial States')
        self.assertEqual(sorted(i for _,_,ids in batches for i in ids), list(range(1,10)))
        self.assertTrue(all(c=='Robot Initial States' and len(ids)<=4 for _,c,ids in batches))
        self.assertEqual(len(selected_records(records, 'Robot Initial States')), 1)
        self.assertEqual(len(records), 2)
        self.assertEqual(category_suffix('Robot Initial States'), '_robot_initial_states')
        self.assertEqual(category_suffix(None), '')

    def test_pure_server_has_no_con1_override(self):
        command = policy_command('pi05_libero_paper_reference', '/baseline/59999', 8900)
        self.assertNotIn('--rapr-runtime-gate', command)
        self.assertIn('pi05_libero_paper_reference', command)
        self.assertEqual(command[-1], '/baseline/59999')

    def test_existing_con1_server_unchanged(self):
        command = policy_command('pi05_libero_paper_con1_direct_delta_stage2', '/con1/4999', 8800)
        self.assertEqual(command[command.index('--rapr-runtime-gate')+1], '1.0')

    def test_reject_adapter_and_wrong_leaf_count(self):
        metadata = dict(tree_metadata={f'base_parameter_{i}': {} for i in range(58)})
        self.assertEqual(audit_metadata(metadata)['parameter_leaves'], 58)
        metadata['tree_metadata']['rapr_router'] = {}
        with self.assertRaises(ValueError):
            audit_metadata(metadata)
        del metadata['tree_metadata']['base_parameter_0']
        with self.assertRaises(ValueError):
            audit_metadata(metadata)


if __name__ == '__main__':
    unittest.main()
