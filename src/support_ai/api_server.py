import argparse
from flask import Blueprint, Flask, jsonify, request, Response
from .lib.chain import Chain
from .utils import get_config
from .lib import const as const


app = Flask(__name__)
chain = None
api_blueprint = Blueprint('api', __name__)

@api_blueprint.route('/ask_ai', methods=['POST'])
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

@api_blueprint.route('/summarize_case', methods=['POST'])
def summarize_case():
    if 'case_number' not in request.form:
        return 'Case number not specified', 400

    data = {
            const.CASE_NUMBER: request.form.get('case_number')
            }
    try:
        return Response(chain.custom_api(const.CONFIG_SF, const.SUMMARIZE_CASE, data), mimetype='text/plain')
    except ValueError:
        return 'Service unavailable', 400

@api_blueprint.route('/clear_history', methods=['POST'])
def clear_history():
    session = request.form.get('session')

    if session is None:
        return 'Session not specified', 400
    chain.clear_history(session)
    return jsonify(success=True)

def parse_args():
    parser = argparse.ArgumentParser(description='Command line tool for support-ai')
    parser.add_argument('--config', type=str, default=None, help='Config path')
    return parser.parse_args()

def main():
    global chain
    args = parse_args()
    config = get_config(args.config)
    chain = Chain(config)

    app.register_blueprint(api_blueprint, url_prefix='/api')
    app.run(host='0.0.0.0', port=8080, debug=False, use_reloader=False)


if __name__ == '__main__':
    main()
