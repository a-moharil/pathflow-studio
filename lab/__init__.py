from pathlib import Path
import os,secrets
from flask import Flask,session,request,jsonify,abort
from . import storage

def create_app(root=None):
    root=Path(root or os.environ.get('PATHFLOW_HOME',Path(__file__).resolve().parents[1]/'workspace')).resolve();storage.init(root)
    app=Flask(__name__);app.config.update(ROOT=root,MAX_CONTENT_LENGTH=2*1024*1024)
    key=root/'secret.key'
    if not key.exists():key.write_text(secrets.token_hex(32))
    app.secret_key=key.read_text()
    @app.before_request
    def csrf():
        session.setdefault('csrf',secrets.token_hex(24))
        if request.method=='POST' and request.headers.get('X-CSRF-Token')!=session['csrf']:abort(403)
    @app.errorhandler(ValueError)
    @app.errorhandler(AssertionError)
    @app.errorhandler(RuntimeError)
    def bad(e):return jsonify(error=str(e)),400
    @app.errorhandler(KeyError)
    def missing(e):return jsonify(error=str(e)),404
    @app.context_processor
    def version_context():
        return {'studio_version':(Path(__file__).resolve().parents[1]/'VERSION').read_text().strip()}
    from .web import web
    app.register_blueprint(web)
    from .joint import engine as joint_engine
    from .joint.web import web as joint_web
    joint_engine.init(root)
    app.register_blueprint(joint_web)
    return app
