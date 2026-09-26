import copy
import numpy as np
import torch
from lab.core.metricfit import pair_design,refine,fit_euclidean,euclidean_distances,deformation,spectral_initial
from lab.core.hyperbolic import compute,fit,distances
from test_hyperbolic import example

def test_pair_coverage_and_reproducibility():
    data={'nodes':[{'layer':0}]*784+[{'layer':1}]*4+[{'layer':2}]*2}
    a=pair_design(data);b=pair_design(data)
    assert a['degree'].min()>=4
    assert len(set(a['train'].ravel()))==790
    assert not set(map(tuple,a['train']))&set(map(tuple,a['validation']))
    np.testing.assert_array_equal(a['train'],b['train']);np.testing.assert_array_equal(a['validation'],b['validation'])
    assert {data['nodes'][i]['layer'] for i in a['sources']}=={0,1,2}

def test_known_euclidean_refinement_and_dimension():
    rng=np.random.default_rng(21);X=rng.normal(size=(22,3));pairs=np.array([(i,j) for i in range(22) for j in range(i+1,22)]);truth=euclidean_distances(X,pairs)
    initial=X+rng.normal(size=X.shape)*.08
    fitted,report=fit_euclidean(initial,pairs,truth,200)
    assert report['best_objective']<report['initial_objective']*1e-5
    assert np.linalg.norm(euclidean_distances(fitted,pairs)-truth)/np.linalg.norm(truth)<1e-4
    D=np.linalg.norm(X[:,None]-X[None,:],axis=-1);two,_=spectral_initial(D,np.arange(22),2);three,_=spectral_initial(D,np.arange(22),3)
    assert two.shape==(22,2) and three.shape==(22,3)

def test_known_lorentz_refinement():
    rng=np.random.default_rng(12);space=rng.normal(size=(15,3))*.35;z=np.column_stack((np.sqrt(1+(space**2).sum(1)),space));pairs=np.array([(i,j) for i in range(15) for j in range(i+1,15)])
    truth=distances(torch.tensor(z),pairs[:,0],pairs[:,1],1).numpy()
    fitted,report=fit(15,3,1,pairs,truth,space+rng.normal(size=space.shape)*.08,200)
    assert report['best_objective']<report['initial_objective']*1e-5
    assert report['best_objective']<1e-8

def test_independent_fits_comparable_pairs_and_invariants():
    data=example();original=copy.deepcopy(data);design,T=refine(data,50,True)
    for key in ['edges','metric','conditional']:assert data[key]==original[key]
    assert set(data['euclidean']['dimensions'])=={'2','3','8'}
    for dim,entries in data['euclidean']['dimensions'].items():
        assert np.shape(entries[0]['positions'])==(len(data['nodes']),int(dim))
        np.testing.assert_allclose(entries[0]['positions'],entries[1]['positions'],atol=1e-10)
        assert entries[0]['optimization']['best_objective']<=entries[0]['optimization']['initial_objective']
    h=compute(data,.7,50,design,T,True)
    assert h['training_pairs']==data['euclidean']['pairs']['training']
    assert h['validation_pairs']==data['euclidean']['pairs']['validation']
    assert h['unconstrained_nodes']==0
    for side in ['s','c']:assert set(data['distance_evidence']['sides'][side]['hyperbolic'])=={'3','8'}
    assert data['euclidean']['deformation']['2']['rms_intrinsic_change']==0

def test_deformation_error_not_hidden_by_alignment():
    a=np.array([1.,2,3,4]);b=a*2
    good=deformation(a,b,a,b);assert good['relative_change_error']==0 and abs(good['change_correlation']-1)<1e-12
    bad=deformation(a,b,a,a);assert bad['relative_change_error']==1 and bad['change_correlation'] is None
    zero=deformation(a,a,a,a);assert zero['relative_change_error'] is None

def test_intrinsic_matrix_export_is_exact_and_budgeted(tmp_path):
    import json
    from lab import create_app,storage as st
    from lab.core.hyperbolic import metric_graph
    from scipy.sparse.csgraph import dijkstra
    app=create_app(tmp_path);data=example()
    for i,node in enumerate(data['nodes']):node['label']=str(i)
    data['measurement_settings']={'limit':50};file=tmp_path/'saved.json';file.write_text(json.dumps(data));jid=st.job(tmp_path,{'kind':'analysis'})
    with st.db(tmp_path) as db:db.execute("UPDATE jobs SET status='completed',result=? WHERE id=?",(str(file),jid))
    client=app.test_client();response=client.get(f'/api/job/{jid}/metric-matrix?side=s&full=1')
    assert response.status_code==200
    actual=np.array(response.json['distances']);truth=dijkstra(metric_graph(data,'s'),directed=False)/data['metric']['display_reference']['unit']
    np.testing.assert_allclose(actual,truth);assert response.json['kind']=='full_intrinsic_matrix'
    data['measurement_settings']['limit']=1;file.write_text(json.dumps(data));assert client.get(f'/api/job/{jid}/metric-matrix?full=1').status_code==400
    assert client.get(f'/api/job/{jid}/metric-matrix').status_code==200


def test_lorentz_gradient_matches_loss_at_small_separations():
    from lab.core.hyperbolic import lorentz_objective
    from lab.core.metricfit import objective_weights
    pairs=np.array([[0,1]]);truth=np.array([1.]);weights,_=objective_weights(truth)
    for curvature in [.01,1.,4.]:
        for separation,step in [(1e-7,1e-9),(.3,1e-6),(1e-14,1e-15)]:
            x=np.array([[0.,0.,0.],[separation,0.,0.]]).ravel()
            loss,gradient=lorentz_objective(x,2,3,curvature,pairs,truth,weights)
            numerical=[]
            for j in range(len(x)):
                plus=x.copy();minus=x.copy();plus[j]+=step;minus[j]-=step
                numerical.append((lorentz_objective(plus,2,3,curvature,pairs,truth,weights)[0]-lorentz_objective(minus,2,3,curvature,pairs,truth,weights)[0])/(2*step))
            np.testing.assert_allclose(gradient,numerical,atol=2e-6,rtol=2e-6)
    # Near-coincident points above the numerical floor must still separate.
    fitted,report=fit(2,3,1.,pairs,truth,np.array([[0.,0.,0.],[1e-7,0.,0.]]),100)
    assert report['best_objective']<1e-8


def test_lorentz_distance_zero_and_numerical_rejection():
    from lab.core.hyperbolic import numerical_validity,move_origin
    space=np.array([[0.,0.,0.],[.2,.3,.4]])
    z=np.column_stack((np.sqrt(1+(space**2).sum(1)),space))
    assert distances(torch.tensor(z),np.array([0,1]),np.array([0,1]),1).tolist()==[0.,0.]
    moved=move_origin(z,z[-1],1.)
    assert numerical_validity(moved,1.,before=z,pairs=np.array([[0,1]]))['valid']
    corrupt=z.copy();corrupt[1,0]+=.01
    assert not numerical_validity(corrupt,1.)['valid']
    corrupt=z.copy();corrupt[1,1]=np.nan
    assert not numerical_validity(corrupt,1.)['valid']
    assert not numerical_validity(z,1.,before=z*2,pairs=np.array([[0,1]]))['valid']


def test_neighborhood_includes_late_layer_sources():
    from lab.core.metricfit import neighborhood
    n=40;X=np.arange(n,dtype=float)[:,None];D=np.abs(X-X.T);sources=np.arange(n)
    assert neighborhood(X,D,sources)==1.
    # Disagree only after the former 12-source prefix. Old code returned 1.
    changed=D.copy()
    for i in range(12,n):changed[i]=D[i][::-1]
    assert neighborhood(X,changed,sources)<.8
    data=example();refine(data,25)
    for entries in data['euclidean']['dimensions'].values():
        for entry in entries:
            assert entry['neighbor_sample_count']==len(entry['neighbor_source_ids'])
            assert {data['nodes'][i]['layer'] for i in entry['neighbor_source_ids']}=={q['layer'] for q in data['nodes']}


def test_invalid_lorentz_fit_is_not_renderable():
    from unittest.mock import patch
    data=example();design,T=refine(data,25)
    def invalid(n,dim,*args,**kwargs):
        z=np.zeros((n,dim+1));z[:,0]=2.
        return z,dict(best_objective=0.)
    with patch('lab.core.hyperbolic.fit',invalid):
        result=compute(data,1.,25,design,T)
    assert result['status']=='numerical_invalid'
    assert result['dimensions']=={}
    assert 'constraint' in result['reason']


def test_hyperbolic_budget_preserves_euclidean_payload():
    data={'nodes':[{}]*4001,'euclidean':{'sentinel':'preserved'}}
    result=compute(data)
    assert result['status']=='budget_exceeded'
    assert data['euclidean']=={'sentinel':'preserved'}
