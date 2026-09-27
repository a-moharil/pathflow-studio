"""Read-only dependency diagnostics. Run with the interpreter used to launch run.py."""
import importlib,sys
print('Interpreter:',sys.executable)
print('Python:',sys.version.split()[0])
failed=[]
for name in ['flask','waitress','numpy','pandas','scipy','torch','torchvision','medmnist']:
    try:
        m=importlib.import_module(name)
        from importlib.metadata import version
        print(name,version(name),'OK')
    except Exception as e:
        failed.append(name);print(name,'FAILED:',repr(e))
if not failed:print('All required imports passed. Launch run.py using this same interpreter.')
else:print('Repair this interpreter using:',sys.executable,'bootstrap.py')
sys.exit(bool(failed))
