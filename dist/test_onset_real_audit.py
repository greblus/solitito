"""Exercise cached replay, ambiguous labels and a pasted notebook without Torch."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import soundfile as sf

from audit_onset_real import (AuditDiscoveryError, EXPECTED_PAIRED_MODEL, annotation_context, classify_miss,
                             find_audit_run, load_audit_inputs, replay_audit, save_json, spectral_evidence)
from build_audit_onset_real_kaggle import build
from onset_events import sha256, latch_events
from train_short_onset import FEATURE_SPEC, onset_features, onset_metrics


def fixture(root):
    run=root/'onset-prepared-fixture'
    features_dir=run/'features'; features_dir.mkdir(parents=True)
    audio_path=root/'fixture.wav'
    t=np.arange(48000)/16000
    audio=(.1*np.sin(2*np.pi*130.8128*t)).astype(np.float32)
    sf.write(audio_path,audio,16000,subtype='FLOAT')
    features=onset_features(audio)
    fp=features_dir/'0000.npy'; np.save(fp,features)
    source={'id':'04_fixture_comp','domain':'guitarset','case':'comp','split':'validation',
            'group':'player04','wav':str(audio_path),'sha256':sha256(audio_path),'duration':3.,
            'features':str(fp),'feature_sha256':sha256(fp),'frames':len(features),
            'events':[{'id':'old','t':.2,'end':2.8,'pc':0,'midi':48,'string':0},
                      {'id':'other','t':1.,'end':1.8,'pc':7,'midi':55,'string':1},
                      {'id':'repeat','t':2.,'end':2.8,'pc':0,'midi':48,'string':0}]}
    index=features_dir/'index.json';save_json(index,{'feature_spec':FEATURE_SPEC,'sources':[source]})
    summary={'ok':True,'experiment':'same-background-ranking-v1','arms':{}}
    for arm in ('control','paired'):
        folder=run/arm; (folder/'probabilities').mkdir(parents=True)
        values=np.zeros((len(features),12),dtype=np.float32)
        values[14:18,0]=.95; values[65:69,7]=.95
        values[127:131,0]=.95 if arm=='control' else .85
        if arm=='paired':values[66:70,0]=.86
        np.save(folder/'probabilities/0000-validation.npy',values)
        threshold=.8 if arm=='control' else .7
        metrics,details=onset_metrics([(source,values)],threshold)
        save_json(folder/'validation_events.json',details)
        save_json(folder/'validation_thresholds.json',{'table':[onset_metrics([(source,values)],x)[0] for x in (.7,.8,.9)]})
        save_json(folder/'contract.json',{'data_index_sha256':sha256(index)})
        summary['arms'][arm]={'threshold':threshold,'validation':metrics,'model_sha256':EXPECTED_PAIRED_MODEL if arm=='paired' else 'fixture-control'}
    save_json(run/'training_summary.json',summary)
    return run,source


class RealAuditTests(unittest.TestCase):
    def test_generated_script_has_only_external_or_standard_imports(self):
        directory=Path(__file__).resolve().parent
        source=build(directory)
        for name in ('train_short_onset','onset_events','onset_ringing'):
            self.assertNotIn('from '+name+' import',source)
        self.assertNotIn('import torch',source)
        compile(source,'audit.py','exec')

    def test_missing_or_wrong_run_never_retrains_or_selects_other_experiment(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            with self.assertRaisesRegex(ValueError,'Do NOT retrain'):
                find_audit_run('auto',root)
            run,_=fixture(root)
            self.assertEqual(find_audit_run('auto',root),run)
            summary=json.loads((run/'training_summary.json').read_text())
            summary['arms']['paired']['model_sha256']='wrong'
            save_json(run/'training_summary.json',summary)
            with self.assertRaises(AuditDiscoveryError) as caught:
                find_audit_run(run,root)
            self.assertIn('Different paired model',caught.exception.inventory['rejected_summaries'][0]['reason'])

    def test_renamed_nested_attached_run_and_txt_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            working=root/'working';working.mkdir()
            attached=root/'input'/'notebook-output'/'nested'
            attached.mkdir(parents=True)
            run,_=fixture(working)
            relocated=attached/'renamed-results'
            shutil.move(run,relocated)
            (relocated/'training_summary.json').rename(relocated/'training_summary.json.txt')
            self.assertEqual(find_audit_run('auto',working,root/'input'),relocated)
            self.assertEqual(find_audit_run(relocated/'training_summary.json.txt',working),relocated)
            _,sources,predictions,_=load_audit_inputs(relocated)
            self.assertEqual(len(sources),1)
            self.assertEqual(len(predictions['paired']),1)

    def test_correct_summary_reports_exact_missing_cache_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);run,_=fixture(root)
            (run/'paired/probabilities/0000-validation.npy').unlink()
            with self.assertRaises(AuditDiscoveryError) as caught:
                find_audit_run('auto',root)
            candidate=caught.exception.inventory['candidate_runs'][0]
            self.assertEqual(candidate['missing_file_count'],1)
            self.assertEqual(candidate['missing_files_first_20'],['paired/probabilities/0000-validation.npy'])

    def test_orphan_model_is_reported_without_claiming_predictions_can_be_recovered(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/'short_onset_paired.onnx').write_bytes(b'fixture-not-a-model')
            with self.assertRaises(AuditDiscoveryError) as caught:
                find_audit_run('auto',root)
            self.assertEqual(caught.exception.inventory['surviving_models'],[str(root/'short_onset_paired.onnx')])
            self.assertIn('summary JSON alone cannot reconstruct',str(caught.exception))

    def test_partial_probabilities_and_changed_index_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            run,_=fixture(Path(tmp))
            p=run/'paired/probabilities/0000-validation.npy'
            values=np.load(p);np.save(p,values[:-1])
            with self.assertRaisesRegex(ValueError,'incomplete probabilities'):load_audit_inputs(run)
            np.save(p,values)
            index=run/'features/index.json';index.write_text(index.read_text()+' ')
            with self.assertRaisesRegex(ValueError,'Changed feature index'):load_audit_inputs(run)

    def test_miss_distinguishes_weak_response_latch_and_reference_competition(self):
        event={'id':'repeat','t':1.,'pc':0}
        source={'duration':2.,'events':[event,{'id':'same','t':1.01,'pc':0}]}
        values=np.zeros((125,12),np.float32);values[65,0]=.85
        row={'predicted':[]}
        self.assertEqual(classify_miss(event,source,values,row,.9)['cause'],'below_threshold')
        # A real plateau emits once at startup and blocks a later reference.
        values[:,0]=.95
        events=latch_events((((i+1)*.016,100,p) for i,p in enumerate(values)),threshold=.9,fill_min=0)
        row['predicted']=[{'t':e.t,'pc':e.pc} for e in events]
        self.assertEqual(classify_miss(event,source,values,row,.9)['cause'],'latch_suppressed')
        row['predicted']=[{'t':1.056,'pc':0}]
        self.assertEqual(classify_miss(event,source,values,row,.9)['cause'],'event_claimed_by_other_reference')

    def test_unselected_threshold_difference_is_reported_not_hidden(self):
        with tempfile.TemporaryDirectory() as tmp:
            run,_=fixture(Path(tmp))
            path=run/'paired/probabilities/0000-validation.npy'
            values=np.load(path)
            # Both sides exceed .7/.8; only the unselected .9 crossing changes.
            values[66:70,0]=.91
            np.save(path,values)
            summary,_,predictions,_=load_audit_inputs(run)
            _,_,differences=replay_audit(run,summary,predictions)
            self.assertEqual([(d['arm'],d['threshold']) for d in differences],[('paired',.9)])

    def test_annotation_conflicts_are_flags_not_removed_references(self):
        source={'events':[{'id':'old','pc':0,'t':0.,'end':1.51,'midi':48,'string':0},
                          {'id':'conflict','pc':7,'t':1.4,'end':2.,'midi':55,'string':0}]}
        context=annotation_context(source,1.5,0)
        self.assertTrue(context['flags']['conflicting_pitch_on_old_string'])
        self.assertTrue(context['flags']['old_note_end_within_32ms'])
        self.assertEqual(len(source['events']),2)
        source['events'].append({'id':'near','pc':0,'t':1.35,'end':2.,'midi':48,'string':2})
        self.assertTrue(annotation_context(source,1.5,0)['flags']['same_pc_start_near_scoring_boundary'])

    def test_spectral_rise_ignores_future_and_marks_shared_harmonics(self):
        f=np.zeros((125,770),np.float16);f[63:68,:]=.1
        a=spectral_evidence(f,1.,48,[48])
        for w in a['windows'].values():
            self.assertEqual(w['unshared_harmonic_count'],0)
            self.assertEqual(w['unshared_harmonics_rise_fraction'],0)
        f[80:]=.9
        self.assertEqual(a,spectral_evidence(f,1.,48,[48]))

    def test_full_pasted_notebook_replay_review_audio_and_failure_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);run,_=fixture(root)
            source=build(Path(__file__).resolve().parent)
            source=source.replace('WORK_ROOT = "/kaggle/working"',f'WORK_ROOT = {str(root)!r}')
            source=source.replace('INPUT_ROOT = "/kaggle/input"',f'INPUT_ROOT = {str(root)!r}')
            script=root/'pasted.py'
            script.write_text('import sys\nsys.modules["ipykernel"] = object()\n'+source)
            def execute():
                return subprocess.run([sys.executable,str(script),'-f','kernel.json'],cwd=root,capture_output=True,text=True,timeout=60,env=dict(os.environ))
            completed=execute()
            self.assertEqual(completed.returncode,0,completed.stdout+completed.stderr)
            path=next(root.glob('onset-real-audit-*/audit_summary.json'))
            report=json.loads(path.read_text())
            self.assertTrue(report['ok'],completed.stdout)
            self.assertEqual(report['comp_held_errors_at_08'],1)
            self.assertEqual(report['loss_causes'],{'below_threshold':1})
            self.assertEqual(report['audio_failures'],[])
            self.assertEqual(report['review_cases'],2)
            self.assertEqual(len(list(path.parent.glob('*.wav'))),2)
            self.assertTrue((path.parent/'review.zip').is_file())
            # Corrupt probabilities without changing shape: replay must catch it.
            values=np.load(run/'paired/probabilities/0000-validation.npy');values[:]=0
            np.save(run/'paired/probabilities/0000-validation.npy',values)
            completed=execute()
            reports=[json.loads(p.read_text()) for p in root.glob('onset-real-audit-*/audit_summary.json')]
            failed=[r for r in reports if not r['ok']]
            self.assertEqual(len(failed),1,completed.stdout+completed.stderr)
            self.assertIn('does not reproduce',failed[0]['error'])


if __name__=='__main__':
    unittest.main()
