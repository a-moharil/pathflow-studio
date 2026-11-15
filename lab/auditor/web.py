"""Read committed evidence; remeasurement never changes training or saved jobs."""
import io,json
from flask import Blueprint,current_app,request,render_template,jsonify,send_file
from .evidence import catalogue,report
from .statistics import population
from .geometry import remeasure

web=Blueprint('auditor',__name__)

@web.get('/auditor')
def page():return render_template('auditor.html',assets='charts')

@web.get('/api/auditor/catalogue')
def records():return jsonify(catalogue(current_app.config['ROOT'],request.args.get('run') or None))

@web.get('/api/auditor/report')
def evidence():
    value=report(current_app.config['ROOT'],request.args.get('record',''))
    if request.args.get('download')=='1':
        return send_file(io.BytesIO(json.dumps(value,indent=2,allow_nan=False).encode()),mimetype='application/json',as_attachment=True,download_name='pathflow-audit.json')
    return jsonify(value)

@web.post('/api/auditor/population')
def infer():
    data=request.get_json();ids=data.get('records',[])
    if not isinstance(ids,list) or not 1<=len(ids)<=100:raise ValueError('Select 1 to 100 saved records.')
    reports=[report(current_app.config['ROOT'],i) for i in ids]
    return jsonify(population(reports,data.get('independent') is True))

@web.post('/api/auditor/geometry')
def geometry():
    value=report(current_app.config['ROOT'],request.get_json().get('record',''),include_graph=True)
    if value['fidelity']['status']=='failed':raise ValueError('Numerical audit failed. Resolve it before interpreting remeasured geometry.')
    measured=remeasure(value['_graph'])
    measured['source_sha256']=value['provenance']['sha256'];measured['record_id']=value['record_id']
    return jsonify(measured)
