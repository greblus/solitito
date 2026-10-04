"""Offline experiment: confirm weak Rise outputs with causal pitch-specific novelty.

Enabled only by SOLITITO_ONSET_RESCUE=1 in the app. No target-note or reference-label input.
Strong model crossings use the existing path. Both paths share one latch.
"""
from functools import lru_cache
import numpy as np
from scipy.optimize import nnls
from short_onset_features import harmonic_dictionary, MIDIS

SR = 16000
HOP = 256
WINDOW = 2048
PAST = 4
SPEC = dict(model_threshold=.8, weak_threshold=.6, novelty_fraction=.15,
            pitch_share=.6, window=WINDOW, past_frames=PAST,
            midi_range=[int(MIDIS[0]),int(MIDIS[-1])],
            future_samples=0, fitted_on_user_audio=False,
            confirmation="raw share >0.5 or fixed-background residual share >=0.6")

@lru_cache(maxsize=1)
def dictionary():
    # Competing harmonic templates allocate overlapping partials jointly.
    d=harmonic_dictionary(WINDOW)[:513].copy()
    d/=np.linalg.norm(d,axis=0)
    return d

def evidence(audio, probabilities):
    audio=np.asarray(audio,dtype=np.float64)
    p=np.asarray(probabilities,dtype=np.float64)
    if audio.ndim!=1 or p.shape!=(len(audio)//HOP,12) or not np.isfinite(audio).all() or not np.isfinite(p).all():
        raise ValueError('Expected finite mono 16kHz audio and aligned frame probabilities')
    padded=np.pad(audio,(WINDOW,0))
    window=np.hanning(WINDOW)
    previous=[]
    shares=np.zeros_like(p)
    fractions=np.zeros(len(p))
    for i in range(len(p)):
        end=(i+1)*HOP
        current=np.abs(np.fft.rfft(padded[end:end+WINDOW]*window))[:513]
        if np.any((p[i]>=SPEC['weak_threshold']) & (p[i]<SPEC['model_threshold'])):
            baseline=sum(previous,np.zeros(513))/PAST
            novelty=np.maximum(current-baseline,0)
            fractions[i]=novelty.sum()/max(current.sum(),1e-12)
            if fractions[i]>=SPEC['novelty_fraction']:
                amplitudes,_=nnls(dictionary(),novelty)
                pc=np.bincount(MIDIS%12,weights=amplitudes,minlength=12)
                shares[i]=pc/max(pc.sum(),1e-12)
        previous.append(current)
        if len(previous)>PAST:previous.pop(0)
    eligible=(p>=SPEC['weak_threshold']) & (shares>=SPEC['pitch_share']) & (fractions[:,None]>=SPEC['novelty_fraction'])
    return eligible,shares,fractions

def decode(probabilities, eligible=None):
    """Original .8 hysteresis plus optional weak evidence; never two latches."""
    p=np.asarray(probabilities,dtype=np.float64)
    eligible=np.zeros_like(p,dtype=bool) if eligible is None else np.asarray(eligible,dtype=bool)
    if p.ndim!=2 or p.shape[1]!=12 or eligible.shape!=p.shape:raise ValueError('Invalid frame shapes')
    armed=np.ones(12,dtype=bool);peaks=np.zeros(12);events=[]
    for frame,row in enumerate(p):
        for pc,value in enumerate(row):
            if value<max(.3*peaks[pc],.1):armed[pc]=True
            elif armed[pc] and (value>=.8 or eligible[frame,pc]):
                events.append(dict(frame=frame+1,t=(frame+1)*.016,pc=pc,weak=bool(value<.8)))
                armed[pc]=False;peaks[pc]=value
    return events

def confirm_pitch(audio, eligible):
    """Confirm a weak candidate one hop later, including under a louder old note.

Keep raw-pitch confirmation. If the old note dominates it, require the same
60% pitch share in the positive residual against the candidate's FIXED past
spectrum. Moving the baseline forward would absorb the new attack itself.
"""
    audio=np.asarray(audio,dtype=np.float64)
    eligible=np.asarray(eligible,dtype=bool)
    padded=np.pad(audio,(WINDOW,0));window=np.hanning(WINDOW)
    confirmed=np.zeros_like(eligible)
    def spectrum(frame):
        end=(frame+1)*HOP
        if frame<0:return np.zeros(513)
        return np.abs(np.fft.rfft(padded[end:end+WINDOW]*window))[:513]
    def shares(magnitude):
        amplitudes,_=nnls(dictionary(),magnitude)
        pc=np.bincount(MIDIS%12,weights=amplitudes,minlength=12)
        return pc/max(pc.sum(),1e-12)
    for i in range(1,len(eligible)):
        if not eligible[i-1].any():continue
        magnitude=spectrum(i)
        confirmed[i]=eligible[i-1] & (shares(magnitude)>.5)
        remaining=eligible[i-1] & ~confirmed[i]
        if remaining.any():
            baseline=sum((spectrum(j) for j in range(i-1-PAST,i-1)),np.zeros(513))/PAST
            residual=np.maximum(magnitude-baseline,0)
            confirmed[i] |= remaining & (shares(residual)>=SPEC['pitch_share'])
    return confirmed

def decode_confirmed(probabilities, confirmed):
    p=np.asarray(probabilities,dtype=np.float64)
    confirmed=np.asarray(confirmed,dtype=bool)
    armed=np.ones(12,dtype=bool);peaks=np.zeros(12);events=[]
    for i,row in enumerate(p):
        for pc,value in enumerate(row):
            if armed[pc] and (value>=.8 or confirmed[i,pc]):
                events.append(dict(frame=i+1,t=(i+1)*.016,pc=pc,weak=bool(value<.8)))
                armed[pc]=False
                peaks[pc]=max(value,p[i-1,pc] if i and confirmed[i,pc] else value)
            elif value<max(.3*peaks[pc],.1):armed[pc]=True
    return events
