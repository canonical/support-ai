import argparse

from flask import Blueprint, Flask, jsonify, request, Response
from flask_restful import Api, Resource

from .lib import const
from .lib.chain import Chain
from .utils import get_config


app = Flask(__name__)
chain = None
api_blueprint = Blueprint('api', __name__)
api = Api(api_blueprint)


class AI(Resource):
    def get(self):
        query = request.args.get('query')
        datasource = request.args.get('datasource')
        session = request.args.get('session')

        if query is None:
            return {'message': 'Query not specified'}, 400
        try:
            return Response(chain.ask(query, ds_type=datasource,
                                      session=session),
                            mimetype='text/plain')
        except ValueError:
            return {'message': 'Service unavailable'}, 400


class Salesforce(Resource):
    def get(self, case_number):
        if case_number is None:
            return {'message': 'Case number not specified'}, 400

        data = {const.CASE_NUMBER: case_number}
        try:
            return Response(chain.custom_api(const.CONFIG_SF,
                                             const.SUMMARIZE_CASE, data),
                            mimetype='text/plain')
        except ValueError:
            return {'message': 'Service unavailable'}, 400


class History(Resource):
    def delete(self):
        session = request.args.get('session')

        if session is None:
            return {'message': 'Session not specified'}, 400
        chain.clear_history(session)
        return jsonify(success=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Command line tool for support-ai')
    parser.add_argument('--config', type=str, default=None, help='Config path')
    return parser.parse_args()


def main():
    global chain
    args = parse_args()
    config = get_config(args.config)
    chain = Chain(config)

    api.add_resource(AI, '/ai')
    api.add_resource(Salesforce, '/salesforce/<string:case_number>/summary')
    api.add_resource(History, '/history')

    app.register_blueprint(api_blueprint, url_prefix='/api')
    app.run(host='0.0.0.0', port=8080, debug=False, use_reloader=False)


if __name__ == '__main__':
    main()
