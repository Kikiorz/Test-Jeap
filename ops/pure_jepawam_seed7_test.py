import copy
import unittest
from unittest import mock

from run_pure_jepawam_full_plus import seeded_manifest
from run_con1_dynamic_eval import selected_batches


class Seed7Test(unittest.TestCase):
    def test_only_seed_changes_no_old_manifest_mutation(self):
        original = dict(final_episode_seed=743, final={'libero_goal':[
            dict(task_id=0,category='Robot Initial States',difficulty_level=5),
            dict(task_id=1,category='Sensor Noise',difficulty_level=5)]},expected_episodes=2)
        with mock.patch('run_pure_jepawam_full_plus.build_manifest',side_effect=lambda _:copy.deepcopy(original)):
            changed = seeded_manifest({},7)
            self.assertEqual(changed['final_episode_seed'],7)
            self.assertEqual(original['final_episode_seed'],743)
            self.assertEqual(changed['final'],original['final'])
            changed['final_episode_seed']=743
            self.assertEqual(changed,original)
            self.assertEqual(selected_batches(changed,{},'Robot Initial States'),
                             [('libero_goal','Robot Initial States',[0])])

    def test_seed_validation(self):
        for seed in (-1,2**32,True,7.5):
            with self.assertRaises(ValueError):
                seeded_manifest({},seed)


if __name__=='__main__':
    unittest.main()
