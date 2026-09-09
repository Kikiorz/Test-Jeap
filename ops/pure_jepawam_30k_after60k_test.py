import unittest
from run_pure_jepawam_30k_after60k import predecessor_complete,command


class SequencingTest(unittest.TestCase):
    def test_wait_until_model_servers_have_exited(self):
        status=dict(state='complete',evaluation_seed=7,category='Robot Initial States')
        summary=dict(complete=True,completed_episodes=1550,errors=0)
        self.assertFalse(predecessor_complete(status,summary,'RUNNING'))
        self.assertTrue(predecessor_complete(status,summary,'EXITED'))
        for service in ('STOPPED','FATAL','BACKOFF'):
            with self.assertRaises(RuntimeError):
                predecessor_complete(status,summary,service)
        for bad in (dict(summary,errors=1),dict(summary,completed_episodes=1549)):
            with self.assertRaises(RuntimeError):
                predecessor_complete(status,bad,'EXITED')
        with self.assertRaises(RuntimeError):
            predecessor_complete(dict(status,evaluation_seed=743),summary,'EXITED')

    def test_only_30k_robot_in_new_directory(self):
        args=command()
        self.assertTrue(args[args.index('--checkpoint')+1].endswith('/29999'))
        self.assertIn('pi05_libero_vjepa_aux/',args[args.index('--checkpoint')+1])
        self.assertEqual(args[args.index('--category')+1],'Robot Initial States')
        self.assertIn('30k_robot_seed7',args[args.index('--output')+1])
        self.assertEqual(args[args.index('--config')+1],'pi05_libero_paper_reference')


if __name__=='__main__':
    unittest.main()
