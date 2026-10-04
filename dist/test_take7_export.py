"""A single four-output graph, including recovery without Torch or datasets."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto
import onnxruntime as ort

from build_model_trainer import build_model_trainer
from take7_training import SnapshotStore, export_combined_model, run_take7, sha256


def model_pair(root):
    chord = root/'best_model_v2_take7_chords.onnx'
    onset = root/'best_model_v2_take7_onset.onnx'
    nodes=[helper.make_node('ReduceMean',['features'],['mean'],axes=[1],keepdims=0)]
    weights=[];outputs=[]
    for name,n in [('root_logits',13),('quality_logits',11),('pitch_logits',12)]:
        weight=name+'_weight'
        weights.append(numpy_helper.from_array(np.full((168,n),.01,np.float32),weight))
        nodes.append(helper.make_node('MatMul',['mean',weight],[name],name=name))
        outputs.append(helper.make_tensor_value_info(name,TensorProto.FLOAT,['batch',n]))
    graph=helper.make_graph(nodes,'chords',[helper.make_tensor_value_info('features',TensorProto.FLOAT,['batch',48,168])],outputs,weights)
    m=helper.make_model(graph,opset_imports=[helper.make_opsetid('',14)],ir_version=8)
    helper.set_model_props(m,{'pitch_threshold':'.7'});onnx.save(m,str(chord))
    weights=[numpy_helper.from_array(np.array([v],np.int64),name) for name,v in [('start',0),('end',12),('axis',1)]]
    graph=helper.make_graph([helper.make_node('Slice',['short_features','start','end','axis'],['onset_logits'])],
        'rise',[helper.make_tensor_value_info('short_features',TensorProto.FLOAT,['batch',770,'time'])],
        [helper.make_tensor_value_info('onset_logits',TensorProto.FLOAT,['batch',12,'time'])],weights)
    onnx.save(helper.make_model(graph,opset_imports=[helper.make_opsetid('',17)],ir_version=8),str(onset))
    return chord,onset


class SingleExportTests(unittest.TestCase):
    def test_combined_graph_contract_parity_metadata_and_independent_branches(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);chord,onset=model_pair(root);before=(sha256(chord),sha256(onset))
            result=export_combined_model(chord,onset,root/'one.onnx',.8)
            self.assertEqual(result['parity_max_absolute_error'],dict.fromkeys(result['outputs'],0.))
            self.assertEqual((sha256(chord),sha256(onset)),before)
            model=onnx.load(result['path']);onnx.checker.check_model(model)
            self.assertEqual([v.name for v in model.graph.output],['root_logits','quality_logits','pitch_logits','onset_logits'])
            metadata={v.key:v.value for v in model.metadata_props}
            self.assertEqual(float(metadata['onset_threshold']),.8)
            self.assertEqual(float(metadata['pitch_threshold']),.7)
            self.assertTrue(all(not i.external_data for i in model.graph.initializer))
            options=ort.SessionOptions();options.intra_op_num_threads=2
            session=ort.InferenceSession(result['path'],sess_options=options,providers=['CPUExecutionProvider'])
            rng=np.random.default_rng(1);x=rng.random((1,48,168),dtype=np.float32);s=rng.random((1,770,65),dtype=np.float32)
            a=session.run(None,{'features':x,'short_features':s})
            b=session.run(None,{'features':x*2,'short_features':s})
            np.testing.assert_array_equal(a[3],b[3])
            s[:,:,35:]=0
            c=session.run(None,{'features':x,'short_features':s})
            for before,after in zip(a[:3],c[:3]):np.testing.assert_array_equal(before,after)
            np.testing.assert_array_equal(a[3][:,:,:35],c[3][:,:,:35])

    def test_export_only_preserves_report_and_never_calls_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);model_pair(root)
            summary=root/'training_summary_v2_take7.json'
            summary.write_text(json.dumps(dict(training_complete=True,onset=dict(threshold=.8,validation={'keep':'all metrics'}))))
            config=dict(work_dir=tmp,run_tag='v2_take7',base_run='v2_take6',mode='export_only')
            with patch('take7_training.prepare_chords',side_effect=AssertionError('training forbidden')), patch('take7_training.prepare_features',side_effect=AssertionError('features forbidden')), patch('take7_training.train_onset_experiment',side_effect=AssertionError('training forbidden')):
                store=SnapshotStore(tmp)
                with patch.object(store,'publish',wraps=store.publish) as publish:
                    report=run_take7(config,store)
                    self.assertEqual([c.args[1] for c in publish.call_args_list if c.args[1].endswith('.onnx')],['best_model_v2_take7.onnx'])
            self.assertEqual(report['onset']['validation'],{'keep':'all metrics'})
            self.assertFalse(report['training_performed'])
            self.assertFalse(report['app_ready'])

    def test_missing_sources_or_threshold_never_restart_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);config=dict(work_dir=tmp,run_tag='v2_take7',base_run='v2_take6',mode='export_only')
            with patch('take7_training.train_onset_experiment',side_effect=AssertionError('training forbidden')):
                with self.assertRaisesRegex(FileNotFoundError,'No training'):
                    run_take7(config,SnapshotStore(tmp))
                model_pair(root)
                with self.assertRaisesRegex(ValueError,'Missing onset threshold'):
                    run_take7(config,SnapshotStore(tmp))
                config['export_onset_threshold']=.8
                self.assertTrue(run_take7(config,SnapshotStore(tmp))['ok'])

    def test_invalid_contract_or_output_cannot_overwrite_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);chord,onset=model_pair(root)
            with self.assertRaisesRegex(ValueError,'overwrite'):
                export_combined_model(chord,onset,chord,.8)
            with self.assertRaisesRegex(ValueError,'threshold'):
                export_combined_model(chord,onset,root/'one.onnx',0)
            m=onnx.load(str(onset));m.graph.input[0].type.tensor_type.shape.dim[1].dim_value=168;onnx.save(m,str(onset))
            with self.assertRaisesRegex(ValueError,'770'):
                export_combined_model(chord,onset,root/'one.onnx',.8)
            self.assertFalse((root/'one.onnx').exists())

    def test_entire_generated_script_exports_without_torch_repository_or_dataset(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);work=root/'v2_take7';work.mkdir();model_pair(work)
            code=build_model_trainer(Path(__file__).parent)
            script=root/'trainer.py';script.write_text(code)
            env=dict(os.environ);env['PYTHONPATH']=os.pathsep.join(p for p in sys.path if p and 'take7-merge-deps' in p)
            done=subprocess.run([sys.executable,str(script),'--mode','export_only','--no-hf','--output-root',str(root),'--export-onset-threshold','.8'],cwd=root,env=env,capture_output=True,text=True,timeout=60)
            self.assertEqual(done.returncode,0,done.stdout+done.stderr)
            report=json.loads((work/'training_summary_v2_take7.json').read_text())
            self.assertEqual(Path(report['model']['path']).name,'best_model_v2_take7.onnx')
            self.assertFalse(report['training_performed'])


if __name__=='__main__':unittest.main()
