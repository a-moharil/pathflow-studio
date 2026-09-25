"""NumPy-only independent path enumeration check. Run python tests/input_fields_numerical.py."""
import ast, math
from pathlib import Path
import numpy as np
# Load the pure numerical functions without requiring the training runtime.
source=ast.parse((Path(__file__).parents[1]/'lab/core/pathmath.py').read_text())
names={'group_logsum','measure','input_origin_field'}
module=ast.Module(body=[n for n in source.body if isinstance(n,ast.FunctionDef) and n.name in names],type_ignores=[])
ns={'np':np,'math':math};exec(compile(module,'pathmath.py','exec'),ns)
rng=np.random.default_rng(71)
for trial in range(100):
    # Four input pixels, three hidden units, two outputs, one bookkeeping sink.
    sizes=[4,3,2,1];layers=[]
    for a,b in zip(sizes[:-1],sizes[1:]):
        v,u=np.indices((b,a));layers.append((u.ravel(),v.ravel(),rng.normal(size=a*b)))
    start=rng.random(4);sec=[rng.random(3),np.ones(2),np.ones(1)]
    stats,flows=ns['measure'](sizes,layers,start,sec)
    expected=np.zeros(4)
    for i in range(4):
        for j in range(3):
            for k in range(2):
                amplitude=layers[0][2][j*4+i]*sec[0][j]*layers[1][2][k*3+j]*layers[2][2][k]
                expected[i]+=start[i]*amplitude**2
    expected/=expected.sum()
    field=ns['input_origin_field'](layers[0][0],flows[0],(1,2,2),stats['defined'])
    np.testing.assert_allclose(np.array(field).ravel(),expected,rtol=1e-12,atol=1e-14)
    # Multiple input channels must sum at corresponding spatial pixels.
    channel=ns['input_origin_field'](layers[0][0],flows[0],(2,1,2),True)
    np.testing.assert_allclose(channel,[expected[:2]+expected[2:]])
    zero,Fzero=ns['measure'](sizes,layers,np.zeros(4),sec)
    assert not zero['defined'] and ns['input_origin_field'](layers[0][0],Fzero[0],(1,2,2),False) is None
print('100 independently enumerated path-field checks passed; channel reduction and undefined measures passed.')
