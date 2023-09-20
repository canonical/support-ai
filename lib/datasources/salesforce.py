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

    def get_update_data(self, start_time, end_time):
        start_date = start_time.date()
        end_date = end_time.date()
        cases = self.sf.query_all(f'SELECT Id, CaseNumber, Subject, Description, ' +
                                  f'Case_Categories_2016__c FROM Case WHERE ' + 
                                  f'LastModifiedDate >= {start_date.isoformat()}T00:00:00Z AND ' + 
                                  f'LastModifiedDate < {end_date.isoformat()}T00:00:00Z LIMIT 1')

        for case in cases['records']:
            yield Data(
                    case['Case_Categories_2016__c'].lower(),
                    case['Description'],
                    {'id': case['Id'], 'subject': case['Subject']},
                    case['CaseNumber']
            )

    def get_content(self, doc):
        content = ''
        comments = self.sf.query_all(f'SELECT CommentBody FROM CaseComment ' +
                                     f'WHERE ParentId = \'{doc.metadata["id"]}\' ' +
                                     f'ORDER BY LastModifiedDate')
        for comment in comments['records']:
            if content:
                content += '\n'
            content += comment['CommentBody']
        return content
