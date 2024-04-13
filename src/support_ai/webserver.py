import pkgutil
import yaml
from flask import Flask, jsonify, request, Response
from .lib.const import CONFIG_FILE
from .lib.chain import Chain

app = Flask(__name__)
data = pkgutil.get_data(__package__, CONFIG_FILE)
if data is None:
    raise Exception(f'{CONFIG_FILE} doesn\'t exist in {__package__}')
config = yaml.safe_load(data.decode('utf-8'))
chain = Chain(config)

@app.route('/ask_ai', methods=['POST'])
def ask_ai():
    query = request.form.get('query')
    datasource = request.form.get('datasource')
    session = request.form.get('session')

    if query is None:
        return 'Query not specified', 400
    try:
        return Response(chain.ask(query, ds_type=datasource, session=session), mimetype='text/plain')
    except ValueError:
        return 'Service unavailable', 400

@app.route('/clear_history', methods=['POST'])
def clear_history():
    session = request.form.get('session')

    if session is None:
        return 'Session not specified', 400
    chain.clear_history(session)
    return jsonify(success=True)

def main():
    app.run(debug=True)


if __name__ == '__main__':
    main()
