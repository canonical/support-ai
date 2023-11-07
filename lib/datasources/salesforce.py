import simple_salesforce
from lib.const import CONFIG_AUTHENTICATION, CONFIG_LLM, \
        CONFIG_SF_PASSWORD, CONFIG_SF_USERNAME, CONFIG_SF_TOKEN
from lib.datasources.ds import Data, Datasource
from lib.llm import LLM


def get_authentication(auth_config):
    if CONFIG_SF_USERNAME not in auth_config:
        raise ValueError(f'The auth config doesn\'t contain {CONFIG_SF_USERNAME}')
    if CONFIG_SF_PASSWORD not in auth_config:
        raise ValueError(f'The auth config doesn\'t contain {CONFIG_SF_PASSWORD}')
    if CONFIG_SF_TOKEN not in auth_config:
        raise ValueError(f'The auth config doesn\'t contain {CONFIG_SF_TOKEN}')
    return {
            'username': auth_config[CONFIG_SF_USERNAME],
            'password': auth_config[CONFIG_SF_PASSWORD],
            'security_token': auth_config[CONFIG_SF_TOKEN]
            }

class SalesforceSource(Datasource):
    def __init__(self, config):
        if CONFIG_AUTHENTICATION not in config:
            raise ValueError(f'The config doesn\'t contain {CONFIG_AUTHENTICATION}')
        if CONFIG_LLM not in config:
            raise ValueError(f'The config doesn\'t contain {CONFIG_LLM}')
        auth = get_authentication(config[CONFIG_AUTHENTICATION])
        self.sf = simple_salesforce.Salesforce(**auth)
        self.llm = LLM(config[CONFIG_LLM])

    def _get_cases(self, start_date=None, end_date=None):
        clause = ''
        conditions = []
        if start_date is not None:
            conditions.append(f'LastModifiedDate >= {start_date.isoformat()}T00:00:00Z')
        if end_date is not None:
            conditions.append(f'LastModifiedDate < {end_date.isoformat()}T00:00:00Z')

        for condition in conditions:
            if clause:
                clause += ' AND '
            clause += condition

        sql_cmd = 'SELECT Id, CaseNumber, Subject, Description, Case_Categories_2016__c ' + \
                'FROM Case' + (f' WHERE {clause}' if clause else '')
        cases = self.sf.query_all(sql_cmd)

        for case in cases['records']:
            yield Data(
                    case['Case_Categories_2016__c'].lower(),
                    case['Description'],
                    {'id': case['Id'], 'subject': case['Subject']},
                    case['CaseNumber']
            )

    def get_initial_data(self, start_date):
        return self._get_cases(start_date)

    def get_update_data(self, start_date, end_date):
        return self._get_cases(start_date, end_date)

    def get_summary_prompt(self):
        return """Write a concise summary of the following:
            "{context}"
            CONCISE SUMMARY:"""

    def get_content(self, doc):
        content = ''
        comments = self.sf.query_all(f'SELECT CommentBody FROM CaseComment ' +
                                     f'WHERE ParentId = \'{doc.metadata["id"]}\' ' +
                                     f'ORDER BY LastModifiedDate')
        for comment in comments['records']:
            if content:
                content += '\n'
            content += comment['CommentBody']
        return [content]
