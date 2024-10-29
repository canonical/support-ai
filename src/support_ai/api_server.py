"""
Support-AI API Server
"""

import argparse

from flask import Blueprint, Flask, jsonify, request, Response
from flask_restful import Api, Resource

from .lib import const
from .lib.chain import Chain
from .utils import get_config


app = Flask(__name__)
chain = None  # pylint: disable=invalid-name
api_blueprint = Blueprint('api', __name__)
api = Api(api_blueprint)


class AI(Resource):  # pylint: disable=too-few-public-methods
    """
    Resource class for the AI endpoint.
    """

    def get(self):  # pylint: disable=no-self-use
        """
        Handles GET requests to the /api/ai endpoint. Queries the AI model
        with the provided query and optional datasource and session arguments.

        Returns:
            Response: A text/plain response with the model's response or
            a JSON error message if the query parameter is missing.
        """
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


class Salesforce(Resource):  # pylint: disable=too-few-public-methods
    """
    Resource class for the Salesforce endpoint.
    """

    def get(self, case_number):  # pylint: disable=no-self-use
        """
        Handles GET requests to the /api/salesforce/<case_number>/summary
        endpoint.

        Args:
            case_number: The case number for which to retrieve a summary.

        Returns:
            Response: A text/plain response with the case summary or a JSON
            error message if the case number is missing or service is
            unavailable.
        """
        if case_number is None:
            return {'message': 'Case number not specified'}, 400

        data = {const.CASE_NUMBER: case_number}
        try:
            return Response(chain.custom_api(const.CONFIG_SF,
                                             const.SUMMARIZE_CASE, data),
                            mimetype='text/plain')
        except ValueError:
            return {'message': 'Service unavailable'}, 400


class History(Resource):  # pylint: disable=too-few-public-methods
    """
    Resource class for the History endpoint.
    """

    def delete(self):  # pylint: disable=no-self-use
        """
        Handles DELETE requests to the /api/history endpoint. Clears the
        history for the specified session.

        Returns:
            Response: A JSON response with success status, or an error
            message if the session parameter is missing.
        """
        session = request.args.get('session')

        if session is None:
            return {'message': 'Session not specified'}, 400
        chain.clear_history(session)
        return jsonify(success=True)


def parse_args():
    """
    Parses command-line arguments for the support-ai API server.

    Returns:
        argparse.Namespace: Parsed arguments, including the config file path.
    """
    parser = argparse.ArgumentParser(
        description='Command line tool for support-ai')
    parser.add_argument('--config', type=str, default=None, help='Config path')
    return parser.parse_args()


def main():
    """
    Main function to initialize the support-ai server. Loads configuration,
    initializes the Chain, and sets up API routes.

    The server listens on all interfaces at port 8080.
    """
    global chain  # pylint: disable=global-statement
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
