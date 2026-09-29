"""Catalog integrity, evaluation contracts, and compatibility with saved drafts."""
import ast
from collections import Counter
import json
from pathlib import Path
import sys
import unittest

APP = Path(__file__).resolve().parents[1] / 'llm_code_lab'
sys.path.insert(0, str(APP))
import build_bank
from cases import make_cases
from judge import copy_tree, leaves

ORIGINAL_IDS = (
    'relu', 'relu_backward', 'sigmoid', 'silu', 'gelu', 'softmax', 'linear',
    'layernorm', 'rmsnorm', 'swiglu', 'attention', 'masked_attention', 'multihead',
    'rope', 'gqa', 'kv_cache', 'cross_entropy', 'decoder',
)


class RecommendationTests(unittest.TestCase):
    def test_catalog_preserves_existing_ids_and_adds_all_priorities(self):
        bank = build_bank.P
        self.assertEqual(tuple(p['id'] for p in bank[:18]), ORIGINAL_IDS)
        self.assertEqual(len(bank), 90)
        self.assertEqual(Counter(p['priority'] for p in bank[18:50]),
                         {'P0': 16, 'P1': 8, 'P2': 8})
        self.assertTrue(all(p['track'] == 'recsys' for p in bank[18:50]))
        self.assertEqual(len({p['id'] for p in bank}), len(bank))

    def test_generated_bank_and_html_match_authoring_sources(self):
        raw = json.dumps(build_bank.P, ensure_ascii=False, indent=2)
        self.assertEqual((APP / 'problems.json').read_text(encoding='utf-8'), raw)
        expected = (APP / 'ui.template.html').read_text(encoding='utf-8').replace(
            '__PROBLEMS_JSON__', raw.replace('</', '<\\/'))
        self.assertEqual((APP / 'index.html').read_text(encoding='utf-8'), expected)

    def test_every_new_problem_has_complete_content_and_executable_templates(self):
        for p in build_bank.P[18:]:
            with self.subTest(id=p['id']):
                for key in ('summary', 'formula', 'contract', 'example', 'hints',
                            'pitfalls', 'followups', 'sources'):
                    self.assertTrue(p[key], key)
                for key in ('starter', 'solution'):
                    tree = ast.parse(p[key])
                    self.assertIn(p['id'], [node.name for node in tree.body
                                          if isinstance(node, ast.FunctionDef)])
                self.assertIn(p['check_mode'], ('autograd', 'numeric'))

    def test_case_gradient_selection_matches_problem_modes(self):
        for p in build_bank.P[18:]:
            cases = make_cases(p['id'])
            self.assertGreaterEqual(len(cases), 6)
            for case in cases:
                with self.subTest(id=p['id'], case=case['name']):
                    self.assertEqual(case['grad'], p['check_mode'] == 'autograd')
                    args = copy_tree(case['args'], case['grad'], case.get('grad_paths'))
                    tensors = leaves(args)
                    if case['grad']:
                        self.assertTrue(any(t.requires_grad for _, t in tensors))
                    for path in case.get('grad_paths', ()):
                        self.assertTrue(any(name == path or name.startswith(path + '.')
                                            or name.startswith(path + '[')
                                            for name, _ in tensors), path)

    def test_loss_labels_are_not_selected_for_gradient_checks(self):
        for pid in ('mse_loss', 'bce_with_logits', 'cross_entropy_extended', 'esmm_loss'):
            case = make_cases(pid)[0]
            args = copy_tree(case['args'], case['grad'], case.get('grad_paths'))
            if pid == 'esmm_loss':
                self.assertTrue(args[0].requires_grad)
                self.assertTrue(args[1].requires_grad)
                self.assertFalse(args[2].requires_grad)
                self.assertFalse(args[3].requires_grad)
            else:
                self.assertTrue(args[0].requires_grad)
                self.assertFalse(args[1].requires_grad)


if __name__ == '__main__':
    unittest.main()
