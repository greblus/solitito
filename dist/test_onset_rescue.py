import unittest
import numpy as np
from onset_rescue import decode,evidence,confirm_pitch,decode_confirmed,HOP

class RescueTests(unittest.TestCase):
    def test_weak_then_strong_is_one_attack(self):
        p=np.zeros((12,12));p[2:10,4]=.7;p[5:8,4]=.95
        eligible=np.zeros_like(p,dtype=bool);eligible[2,4]=True
        self.assertEqual([(e['frame'],e['pc']) for e in decode(p,eligible)],[(3,4)])
        self.assertEqual([(e['frame'],e['pc']) for e in decode(p)],[(6,4)])
    def test_fresh_repluck_can_follow_weak_attack(self):
        p=np.zeros((12,12));p[2:5,4]=.7;p[7:10,4]=.95
        eligible=np.zeros_like(p,dtype=bool);eligible[2,4]=True
        self.assertEqual([e['frame'] for e in decode(p,eligible)],[3,8])
    def test_future_audio_cannot_change_past_evidence(self):
        rng=np.random.default_rng(11);a=rng.normal(0,.01,24*HOP)
        p=np.full((24,12),.65)
        short=evidence(a[:12*HOP],p[:12]);full=evidence(a,p)
        for x,y in zip(short,full):np.testing.assert_array_equal(x,y[:12])
    def test_evidence_is_gain_invariant_and_silence_is_not_attack(self):
        a=np.sin(np.arange(32*HOP)*2*np.pi*220/16000)*.1
        a[:8*HOP]=0
        p=np.full((32,12),.65)
        x=evidence(a,p);y=evidence(a*.25,p)
        np.testing.assert_array_equal(x[0],y[0]);np.testing.assert_allclose(x[1],y[1],atol=1e-10)
        self.assertFalse(evidence(np.zeros_like(a),p)[0].any())
    def test_confirmation_is_causal_and_requires_the_same_pitch(self):
        a=np.sin(np.arange(24*HOP)*2*np.pi*220/16000)*.1
        eligible=np.ones((24,12),dtype=bool)
        full=confirm_pitch(a,eligible)
        np.testing.assert_array_equal(full[:12],confirm_pitch(a[:12*HOP],eligible[:12]))
        self.assertFalse(full[0].any())
        self.assertTrue(full[16:,9].all())
        self.assertFalse(full[16:,:9].any())
        self.assertFalse(full[16:,10:].any())
    def test_confirmation_consumes_the_same_latch_as_strong_response(self):
        p=np.zeros((12,12));p[2:5,4]=.7;p[5:9,4]=.95
        confirm=np.zeros_like(p,dtype=bool);confirm[3,4]=True
        self.assertEqual([e['frame'] for e in decode_confirmed(p,confirm)],[4])
    def test_new_note_is_confirmed_even_when_the_old_note_is_louder(self):
        # A sustained E followed by a quieter G. Eligible onset still needs
        # real novelty; merely setting model confidence cannot rescue a tail.
        frames=64
        t=np.arange(frames*HOP)/16000
        old=np.sin(2*np.pi*164.813778*t)
        new=.5*np.sin(2*np.pi*195.997718*t)
        new[:32*HOP]=0
        p=np.zeros((frames,12));p[32:40,7]=.7
        eligible=evidence(old+new,p)[0]
        confirmed=confirm_pitch(old+new,eligible)
        self.assertTrue(eligible[:,7].any())
        self.assertTrue(confirmed[:,7].any())
        self.assertFalse(evidence(old,p)[0][:,7].any())
        self.assertEqual(len(decode_confirmed(p,confirmed)),1)

if __name__=='__main__':unittest.main()
