# Self Note

The note for myself or I'll forget it.

## Input/Output Format
```
At training:
src:
<sos> e1 e2 e3 ... eL <eos>

tgt:
if e has phoneme representation:
<cot> p1 p2 p3 ... pM <sos> k1 k2 k3 ... kM <eos>
else:
<sos> k1 k2 k3 ... kN <eos>
```
At inference, whether the cot is enabled is determined by feeding `<cot>` token to the decoder or not.
It's nothing like chain of thought, I use this name just for chasing the fashion.
Better name would be corase-to-fine or auxiliary loss, whatever.
