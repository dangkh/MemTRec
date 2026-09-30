"""CPU checks without torch/transformers or downloads.
Loads the real protocol components; substitutes model calls and graph pruning only.
Run: python tests/test_frozen_protocol.py
"""
import ast
import importlib.util
import json
from pathlib import Path
import random
import sys
import tempfile
import types
import typing
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT/relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

Dataset = load('frozen', 'src/data/dataset_frozen.py').FrozenRecDataset
Graph = load('graph', 'src/memory/graph.py').UserItemGraph
Storage = load('storage', 'src/memory/storage.py').MemoryStorage
Packer = load('packer', 'src/memory/packer.py').SnippetPacker
Manager = load('manager', 'src/memory/manager.py').MemRecManager
Ranker = load('ranker', 'src/models/reranker_llm.py').LLMReranker

class Pruner:
    def __init__(self, **kwargs): pass
    def prune(self, user, graph, candidates):
        neighbors = [{'id': i, 'type': 'item', 'score': 1.0}
                     for i in graph.get_user_items(user)]
        return {'user_id': user, 'neighbors': neighbors, 'n_items': len(neighbors), 'n_users': 0}

class LLM:
    model = 'mock'
    api_endpoint = None
    def __init__(self): self.prompts = []
    def generate_json(self, messages, properties, **kwargs):
        self.prompts.append(json.dumps(messages))
        assert 'SECRET_DESCRIPTION' not in self.prompts[-1]
        if 'ranking' in properties:
            return {'ranking': ['C02', 'C02', 'FAKE'], 'reasoning': 'A match.'}
        if 'facets' in properties:
            return {'facets': [], 'support_edges': [], 'vector_profile': []}
        return {'user_memory': 'Learned from observed train interactions.',
                'item_memory': 'Train item preference.', 'neighbor_updates': []}

# Execute full, unchanged agent class with injected dependencies (no GPU libraries).
def class_from_source(relative, name, namespace):
    tree = ast.parse((ROOT/relative).read_text())
    node = next(x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == name)
    namespace.update(vars(typing))
    namespace.update(json=json, np=np, time=__import__('time'), Path=Path, random=random)
    exec(compile(ast.Module(body=[node], type_ignores=[]), relative, 'exec'), namespace)
    return namespace[name]

Agent = class_from_source('src/models/memrec_agent.py', 'MemRecAgent', {
    'UserItemGraph': Graph, 'MemoryStorage': Storage, 'SnippetPacker': Packer,
    'MemRecManager': Manager, 'NeighborPruner': Pruner, 'LLMRulePruner': Pruner,
    'LLMReranker': Ranker})

# Load trainer class, keeping all method bodies, with import-time placeholders.
Trainer = class_from_source('src/train/trainer_memrec.py', 'MemRecTrainer', {
    'RecDataset': object, 'torch': types.SimpleNamespace(device=object),
    'threading': __import__('threading'), 'MemRecAgent': Agent,
    'LLMClient': lambda **kwargs: LLM(),
    'tqdm': lambda seq, **kwargs: seq})
graph_module = types.ModuleType('src.memory.graph'); graph_module.UserItemGraph = Graph
sys.modules['src.memory.graph'] = graph_module

class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.seq = {'uB': {'train': ['0', '1', '2'], 'val': '3', 'test': '4'},
                    'uA': {'train': ['1', '2', '0'], 'val': '3', 'test': '5'}}
        self.put('user_sequences.json', self.seq)
        self.put('items.json', {str(i): {'title': f'Title {i}', 'category': ['Drama'],
                                        'description': 'SECRET_DESCRIPTION'} for i in range(9)})
        self.put('user_negatives.json', {u: {'val_neg': ['6', '7', '8']} for u in self.seq})
        self.put('train_users_1.json', ['uB']); self.put('eval_users_1.json', ['uA'])
        self.put('user_candidates_testpool1000_seed42.json', {'uA': {'candidates': ['8', '5', '7'], 'target': '5'}})
        self.config = {'num_train_users': 1, 'num_test_users': 1, 'history_size': 2,
                       'n_eval_candidates': 3, 'seed': 42, 'instruction_mode': 'history',
                       'provider': {'name': 'local_hf', 'model': 'mock'},
                       'warmup': {'enabled': True, 'rounds': 1, 'n_candidates': 3},
                       'memrec': {'reranker_mode': 'llm', 'reranker_output': 'list',
                                  'pruner': {'mode': 'hybrid_rule'}}, 'topk': [1, 3, 5, 10]}
    def put(self, name, value): (self.base/name).write_text(json.dumps(value))
    def dataset(self): return Dataset(self.base, self.config)

    def test_order_mapping_metadata_and_graph(self):
        ds = self.dataset()
        self.assertEqual(list(ds.user_id_map), ['uB', 'uA'])
        u = ds.eval_user_ids[0]
        self.assertEqual([ds.raw_item_ids[i] for i in ds.frozen_candidates[u]], ['8', '5', '7'])
        self.assertEqual([ds.raw_item_ids[i] for i in ds.get_user_train_items(u)], ['2', '0'])
        self.assertNotIn('SECRET_DESCRIPTION', json.dumps(ds.item_metadata))
        graph = Graph(ds)
        self.assertTrue(graph.get_user_items(u))
        self.assertNotIn(u, [x for values in graph.users_by_item.values() for x in values])

    def test_validation(self):
        self.put('user_candidates_testpool1000_seed42.json', {'uA': {'candidates': ['8', '5', '7'], 'targets': ['4']}})
        with self.assertRaisesRegex(ValueError, 'cached target'): self.dataset()
        self.put('eval_users_1.json', ['uB'])
        with self.assertRaisesRegex(ValueError, 'overlap'): self.dataset()

    def test_clabel_repair_and_fallback(self):
        r = Ranker(LLM(), output_mode='list')
        candidates = [{'id': x, 'title': 'title', 'category': 'Drama'} for x in (101, 7, 309)]
        scores = r.rerank(0, {}, candidates)
        self.assertEqual([x['item_id'] for x in scores], [7, 101, 309])
        self.assertEqual(r.last_diagnostics['duplicates'], ['C02'])
        self.assertEqual(r.last_diagnostics['unknown'], ['FAKE'])
        r.llm.generate_json = lambda **kwargs: {'ranking': 'C02'}
        self.assertEqual([x['item_id'] for x in r.rerank(0, {}, candidates)], [101, 7, 309])

    def test_train_only_and_frozen_eval(self):
        ds = self.dataset()
        trainer = Trainer(None, ds, self.config, None)
        trainer.train(str(self.base/'output'))
        calls = []
        original = trainer.agent.write
        def tracked(user, feedback, *args, **kwargs):
            self.assertIn(user, ds.train_user_ids)
            self.assertIn(feedback['item_id'], ds.full_train_data[user])
            self.assertNotIn(feedback['item_id'], ds.train_data[user])
            calls.append(feedback['item_id'])
            return original(user, feedback, *args, **kwargs)
        trainer.agent.write = tracked
        result = trainer.test()
        self.assertEqual(len(calls), 1)
        self.assertEqual(result['n_stage_w_calls'], 0)
        self.assertTrue(result['memory_unchanged'])
        self.assertEqual(result['Hit@1'], 1.0)
        with self.assertRaisesRegex(RuntimeError, 'Stage-W'):
            original(ds.eval_user_ids[0], {'item_id': 1}, [])
        with self.assertRaisesRegex(RuntimeError, 'frozen'):
            trainer.agent.storage.update_user_memory(0, 'forbidden')
        with self.assertRaisesRegex(RuntimeError, 'frozen'):
            trainer.agent.storage.update_item_memory(0, 'forbidden')
        all_prompts = ''.join(trainer.llm_client.prompts)
        self.assertNotIn('SECRET_DESCRIPTION', all_prompts)
        self.assertIn('Drama', all_prompts)

    def test_cli_flags_declared_before_parse(self):
        source = (ROOT/'scripts/run_train.py').read_text()
        self.assertLess(source.index("parser.add_argument('--train_users_file')"), source.index('args = parser.parse_args()'))

if __name__ == '__main__': unittest.main()
