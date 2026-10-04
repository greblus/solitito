"""Take7 routing, frozen base, real optimizer resume, and standalone packaging."""
import ast
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from build_model_trainer import build_model_trainer
from chord_training import chord_runtime
from take7_training import SnapshotStore, choose_chord_start, prepare_chords, run_take7, weights_digest
from test_onset_preparation import fixture
from train_short_onset import run_pipeline, onset_sources, cache_onset_features, train_onset_experiment

HAS_TRAINING = all(importlib.util.find_spec(m) for m in ('torch', 'onnx', 'onnxruntime'))


class Take7Tests(unittest.TestCase):
    def test_start_modes_and_no_overwriting_parent(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = SnapshotStore(tmp)
            self.assertEqual(choose_chord_start(store, 'v2_take7', 'v2_take6', 'auto'), ('fresh', None))
            with self.assertRaisesRegex(ValueError, 'needs a chord checkpoint'):
                choose_chord_start(store, 'v2_take7', 'v2_take6', 'onset_only')
            parent = Path(tmp)/'checkpoint_v2_take6_best.pth'
            parent.write_bytes(b'base')
            self.assertEqual(choose_chord_start(store, 'v2_take7', 'v2_take6', 'auto'), ('base', parent))
            self.assertEqual(choose_chord_start(store, 'v2_take7', 'v2_take6', 'full'), ('fresh', None))
            own = Path(tmp)/'checkpoint_v2_take7_best.pth';own.write_bytes(b'own')
            self.assertEqual(choose_chord_start(store, 'v2_take7', 'v2_take6', 'full'), ('resume', own))
            with self.assertRaisesRegex(ValueError, 'never overwrite'):
                run_take7(dict(work_dir=tmp, run_tag='v2_take6', base_run='v2_take6'), store)
            self.assertEqual(parent.read_bytes(), b'base')

    def test_access_error_never_becomes_fresh_training(self):
        from types import SimpleNamespace
        store = SimpleNamespace(fetch=lambda name: (_ for _ in ()).throw(ConnectionError('HF down')))
        with self.assertRaisesRegex(ConnectionError, 'HF down'):
            choose_chord_start(store, 'v2_take7', 'v2_take6', 'auto')

    def test_standalone_matches_sources_and_has_no_repo_imports(self):
        root = Path(__file__).parent
        generated = build_model_trainer(root)
        self.assertEqual((root/'model_trainer.py').read_text(), generated)
        tree = ast.parse(generated)
        modules = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
        self.assertFalse(modules & {'train_short_onset', 'chord_training', 'take7_training', 'onset_rise'})
        self.assertNotIn('exec(', generated)

    @unittest.skipUnless(HAS_TRAINING, 'temporary training dependencies needed')
    def test_take6_chord_outputs_preserved_and_bad_base_rejected(self):
        import torch
        from torch import nn
        torch.set_num_threads(2)
        # Read the historical class definitions without running its Kaggle bootstrap.
        source = subprocess.check_output(['git','show','HEAD:dist/model_trainer.py'], text=True)
        nodes = [n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and
                 n.name in ('SEBlock','ConvBlockSE','ChordTransformer')]
        namespace = dict(torch=torch, nn=nn, DROPOUT_RATE=.2, CTX_FRAMES=48,
                         QUALITIES=list(range(11)), ONSET_LOOKBACK=6)
        exec(compile(ast.Module(body=nodes, type_ignores=[]), 'take6_classes', 'exec'), namespace)
        old = namespace['ChordTransformer']().eval()
        with tempfile.TemporaryDirectory() as tmp:
            cfg=dict(run_tag='v2_take7', input_dir=tmp, work_dir=tmp, device='cpu')
            runtime=chord_runtime(cfg, SnapshotStore(tmp));new=runtime.model().eval()
            runtime.load_weights(new, old.state_dict())
            x=torch.randn(2,48,168)
            with torch.no_grad():
                a=old(x);b=new(x)
            self.assertEqual(len(b),3)
            for first,second in zip(a,b):torch.testing.assert_close(first,second,rtol=0,atol=0)
            partial=dict(new.state_dict());partial.pop('proj.weight')
            with self.assertRaises(RuntimeError):runtime.load_weights(new,partial)
            base=Path(tmp)/'checkpoint_v2_take6_best.pth'
            torch.save(dict(model_state_dict=new.state_dict(), best_threshold=.7),base)
            before=weights_digest(new.state_dict());original=base.read_bytes()
            cfg.update(mode='onset_only', base_run='v2_take6')
            result=prepare_chords(cfg,SnapshotStore(tmp))
            self.assertEqual(result['weights_sha256'],before)
            self.assertEqual(base.read_bytes(),original)
            import onnxruntime as ort
            session=ort.InferenceSession(result['path'], providers=['CPUExecutionProvider'])
            self.assertEqual([o.name for o in session.get_outputs()],result['outputs'])
            actual=session.run(None,{'features':x.numpy()})
            for expected, exported in zip(b,actual):np.testing.assert_allclose(expected.numpy(),exported,rtol=2e-4,atol=2e-5)

    @unittest.skipUnless(HAS_TRAINING, 'temporary training dependencies needed')
    def test_rise_interrupted_resume_equals_uninterrupted_and_export_only(self):
        import torch
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);inputs=root/'input';inputs.mkdir();fixture(inputs)
            prepared=run_pipeline(inputs,root/'prepared','auto',groups=(1,1,1))
            feat=root/'features';sources=cache_onset_features(onset_sources(Path(prepared['prepared_directory'])),feat)
            full=root/'full';resumed=root/'resumed';full.mkdir();resumed.mkdir()
            kwargs=dict(epochs=2,batch_size=16,device_name='cpu',feature_directory=feat,spectral_rise=True,resume=True)
            reference=train_onset_experiment(sources,full,**kwargs)
            def interrupt(path):raise RuntimeError('simulated interruption after checkpoint')
            with self.assertRaisesRegex(RuntimeError,'simulated interruption'):
                train_onset_experiment(sources,resumed,checkpoint_callback=interrupt,**kwargs)
            recovered=train_onset_experiment(sources,resumed,**kwargs)
            a=torch.load(full/'short_onset_last.pt',weights_only=False)
            b=torch.load(resumed/'short_onset_last.pt',weights_only=False)
            for key in a['state_dict']:torch.testing.assert_close(a['state_dict'][key],b['state_dict'][key],rtol=0,atol=0)
            self.assertEqual(reference['training_batches_sha256'],recovered['training_batches_sha256'])
            self.assertEqual(reference['test'],recovered['test'])
            with patch.object(torch.optim.AdamW,'step',side_effect=AssertionError('unexpected retraining')):
                train_onset_experiment(sources,resumed,**kwargs)
            bad=dict(sources[0]);bad['sha256']='different'
            with self.assertRaisesRegex(ValueError,'changed data'):
                train_onset_experiment([bad]+sources[1:],resumed,**kwargs)

    @unittest.skipUnless(HAS_TRAINING, 'temporary training dependencies needed')
    def test_full_chord_training_and_best_checkpoint_export(self):
        import torch
        torch.set_num_threads(2)
        class Tiny(torch.utils.data.Dataset):
            def __init__(self):self.x=torch.randn(4,48,168)
            def set_epoch(self,epoch):pass
            def __len__(self):return 4
            def __getitem__(self,i):return self.x[i],torch.tensor(0),torch.tensor(0),torch.ones(12),torch.tensor(True),torch.tensor(True)
        with tempfile.TemporaryDirectory() as tmp:
            config=dict(run_tag='fresh',input_dir=tmp,work_dir=tmp,device='cpu',chord_epochs=1)
            store=SnapshotStore(tmp);runtime=chord_runtime(config,store);model=runtime.model()
            before=weights_digest(model.state_dict());loader=torch.utils.data.DataLoader(Tiny(),batch_size=2)
            runtime.phase1(model,loader,loader)
            saved=torch.load(store.fetch('checkpoint_fresh_best.pth'),weights_only=False)
            self.assertTrue(saved['phase1_done'])
            self.assertNotEqual(before,weights_digest(saved['model_state_dict']))
            self.assertFalse(any(k.startswith('fc_onset') for k in saved['model_state_dict']))
            runtime.phase2(model,loader)
            self.assertTrue(torch.load(store.fetch('checkpoint_fresh_best.pth'),weights_only=False)['phase2_done'])

    @unittest.skipUnless(HAS_TRAINING, 'temporary training dependencies needed')
    def test_complete_standalone_run_without_repository_or_hf(self):
        import torch
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);inputs=root/'input';inputs.mkdir();fixture(inputs)
            work=root/'v2_take7';work.mkdir()
            config=dict(run_tag='v2_take7',base_run='v2_take6',mode='auto',
                        input_dir=str(inputs),work_dir=str(work),device='cpu',
                        groups=[1,1,1],onset_epochs=1,onset_batch_size=16)
            store=SnapshotStore(work);runtime=chord_runtime(config,store)
            base=work/'checkpoint_v2_take6_best.pth'
            model=runtime.model()
            torch.save(dict(model_state_dict=model.state_dict(),best_threshold=.7),base)
            original=base.read_bytes()
            script=root/'standalone.py';script.write_text(build_model_trainer(Path(__file__).parent))
            runner=root/'run.py'
            runner.write_text("import torch; torch.set_num_threads(2)\nimport standalone as m\n"
                              + "c="+repr(config)+"\nr=m.run_take7(c,m.SnapshotStore(c['work_dir']))\nassert r['ok']\n")
            env=dict(os.environ);env['PYTHONPATH']=os.pathsep.join(p for p in sys.path if p and 'solitito-take7' in p)
            done=subprocess.run([sys.executable,str(runner)],cwd=root,env=env,capture_output=True,text=True,timeout=180)
            self.assertEqual(done.returncode,0,done.stdout+done.stderr)
            report=json.loads((work/'training_summary_v2_take7.json').read_text())
            self.assertTrue(report['ok'])
            self.assertTrue(report['onset']['spectral_rise'])
            self.assertEqual(Path(report['model']['path']).name,'best_model_v2_take7.onnx')
            self.assertEqual(report['model']['outputs'],['root_logits','quality_logits','pitch_logits','onset_logits'])
            self.assertEqual(report['chords']['initialized_from'],'base')
            self.assertEqual(base.read_bytes(),original)
            # Run again with the same run: no optimizer step should happen.
            runner.write_text("import torch; torch.set_num_threads(2)\nimport standalone as m\n"
                              + "from unittest.mock import patch\nc="+repr(config)+"\n"
                              + "with patch.object(torch.optim.AdamW,'step',side_effect=AssertionError('retraining')):\n"
                              + " r=m.run_take7(c,m.SnapshotStore(c['work_dir']))\nassert r['ok']\n")
            again=subprocess.run([sys.executable,str(runner)],cwd=root,env=env,capture_output=True,text=True,timeout=180)
            self.assertEqual(again.returncode,0,again.stdout+again.stderr)


if __name__=='__main__':unittest.main()
