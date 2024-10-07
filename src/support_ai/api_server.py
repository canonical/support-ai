import argparse
from flask import Flask, jsonify, request, Response
from .lib.chain import Chain
from .utils import get_config


app = Flask(__name__)
chain = None

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

@app.route('/summarize_case', methods=['POST'])
def summarize_case():
    case_number = request.form.get('case_number')

    if case_number is None:
        return 'Case number not specified', 400
    try:
        return Response(chain.summarize_case(case_number), mimetype='text/plain')
    except ValueError:
        return 'Service unavailable', 400

@app.route('/clear_history', methods=['POST'])
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

    app.run(host='0.0.0.0', port=8080, debug=True)


if __name__ == '__main__':
    main()
