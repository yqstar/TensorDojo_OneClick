"""Integration coverage for the LLM and reinforcement-learning expansion."""
from collections import Counter
import json
from pathlib import Path
import re
import sys
import unittest

APP = Path(__file__).resolve().parents[1] / 'llm_code_lab'
sys.path.insert(0, str(APP))
import build_bank
from cases import case_modules, make_cases
from judge import copy_tree, leaves


class AdvancedTrackTests(unittest.TestCase):
    def test_catalog_and_dispatch_cover_the_same_ninety_problems(self):
        bank = build_bank.P
        self.assertEqual(Counter(p['track'] for p in bank),
                         {'llm': 42, 'recsys': 32, 'rl': 16})
        registered = [pid for module in case_modules() for pid in module.IDS]
        self.assertEqual(len(registered), len(set(registered)))
        self.assertEqual(set(registered), {p['id'] for p in bank[18:]})
        self.assertEqual([p['number'] for p in bank], list(range(1, 91)))
        self.assertTrue(all(p['priority'] in ('P0', 'P1', 'P2') for p in bank[50:]))

    def test_learning_routes_cover_every_added_problem_without_dead_links(self):
        html = (APP / 'ui.template.html').read_text(encoding='utf-8')
        match = re.search(r'const LEARNING_PATHS=(\[.*?\]);\s*const BYID', html, re.S)
        self.assertIsNotNone(match)
        routes = json.loads(match.group(1))
        bank = {p['id']: p for p in build_bank.P}
        routed = set()
        for route in routes:
            ids = [pid for _, step in route['steps'] for pid in step]
            self.assertEqual(len(ids), len(set(ids)), route['track'])
            self.assertTrue(set(ids) <= bank.keys(), route['track'])
            routed.update(ids)
            if route['track'] == 'rl':
                self.assertEqual(set(ids), {pid for pid, p in bank.items() if p['track'] == 'rl'})
        self.assertTrue({p['id'] for p in build_bank.P[50:]} <= routed)
        self.assertIn('<option value="rl">强化学习</option>', html)
        self.assertIn("const STORAGE='tensor-dojo.v1'", html)

    def test_every_added_problem_has_float32_and_float64_cases(self):
        for p in build_bank.P[50:]:
            with self.subTest(pid=p['id']):
                cases = make_cases(p['id'])
                self.assertGreaterEqual(len(cases), 12)
                self.assertEqual(len({c['name'] for c in cases}), len(cases))
                dtypes = {str(t.dtype) for c in cases for _, t in leaves(c['args'])
                          if t.is_floating_point()}
                self.assertEqual(dtypes, {'torch.float32', 'torch.float64'})

    def test_frozen_targets_and_old_policies_are_not_treated_as_parameters(self):
        expected = {
            'lora_linear': {'input[0]', 'input[2]', 'input[3]'},
            'causal_lm_loss': {'input[0]'},
            'sequence_log_probs': {'input[0]'},
            'kl_distillation_loss': {'input[0]'},
            'dpo_loss': {'input[0]', 'input[1]'},
            'reward_pairwise_loss': {'input[0]', 'input[1]'},
            'reinforce_loss': {'input[0]'},
            'importance_sampling_ratio': {'input[0]'},
            'ppo_clipped_loss': {'input[0]'},
            'ppo_value_loss': {'input[0]'},
            'dqn_loss': {'input[0]'},
            'grpo_loss': {'input[0]'},
        }
        for pid, paths in expected.items():
            for case in make_cases(pid):
                with self.subTest(pid=pid, case=case['name']):
                    inputs = copy_tree(case['args'], case['grad'], case.get('grad_paths'))
                    actual = {path for path, tensor in leaves(inputs) if tensor.requires_grad}
                    self.assertEqual(actual, paths)


if __name__ == '__main__':
    unittest.main()
