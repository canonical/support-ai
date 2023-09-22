import simple_salesforce
from lib.const import CONFIG_SF, CONFIG_SF_PASSWORD, CONFIG_SF_USERNAME, CONFIG_SF_TOKEN
from lib.datasources.ds import Data, Datasource


class SalesforceSource(Datasource):
    def __init__(self, config):
        if CONFIG_SF_USERNAME not in config:
            raise ValueError(f'The configuration\'s {CONFIG_SF} ' +
                             f'section doesn\'t contain {CONFIG_SF_USERNAME}')
        if CONFIG_SF_PASSWORD not in config:
            raise ValueError(f'The configuration\'s {CONFIG_SF} ' +
                             f'section doesn\'t contain {CONFIG_SF_PASSWORD}')
        if CONFIG_SF_PASSWORD not in config:
            raise ValueError(f'The configuration\'s {CONFIG_SF} ' +
                             f'section doesn\'t contain {CONFIG_SF_TOKEN}')
        auth = {
            'username': config[CONFIG_SF_USERNAME],
            'password': config[CONFIG_SF_PASSWORD],
            'security_token': config[CONFIG_SF_TOKEN]
        }
        self.sf = simple_salesforce.Salesforce(**auth)

    def _get_cases(self, start_time=None, end_time=None):
        clause = ''
        conditions = []
        if start_time is not None:
            conditions.append(f'LastModifiedDate >= {start_time.date().isoformat()}T00:00:00Z')
        if end_time is not None:
            conditions.append(f'LastModifiedDate < {end_time.date().isoformat()}T00:00:00Z')

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

    def get_initial_data(self):
        return self._get_cases()

    def get_update_data(self, start_time, end_time):
        return self._get_cases(start_time, end_time)

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
