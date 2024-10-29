import json
import os
import sys
from datetime import datetime as dt

import yaml
from launchpadlib.launchpad import Launchpad

CONFIG_PATH = "config.yaml"
PROJECT_NAME = "linux"


with open(CONFIG_PATH, 'r', encoding="utf8") as f:
    data = yaml.load(f, Loader=yaml.FullLoader)


if 'data_dir' not in data or 'model_path' not in data:
    sys.exit(f'Unexpected config format [{data}]')


if not os.path.exists(data['data_dir']):
    os.mkdir(data['data_dir'])


launchpad = Launchpad.login_anonymously('test', 'production',
                                        version='devel')

project = launchpad.projects[PROJECT_NAME]
bugs = project.searchTasks(status=['Fix Committed', 'Fix Released'])
# bugs = project.searchTasks(status = ['New', 'Incomplete', 'Triaged',
#                                      'Opinion',
#                                      'Invalid', 'Won\'t Fix', 'Confirmed',
#                                      'In Progress',
#                                      'Fix Committed', 'Fix Released'])


for bug in bugs:
    browser = launchpad._browser
    bugInfoUndecode = browser.get(bug.bug_link)
    bugInfoJson = bugInfoUndecode.decode('utf-8')
    bugInfo = json.loads(bugInfoJson)
    bugDate = bugInfo["date_created"].split('T')[0]
    bugDateTime = dt.strptime(bugDate, "%Y-%m-%d")
    limitDateTime = dt.strptime("2020-01-01", "%Y-%m-%d")
    if bugDateTime > limitDateTime:
        print(bug.web_link)
        if os.path.exists(data['data_dir'] + "/lp-" + PROJECT_NAME +
                          "-" + str(bugInfo['id']) + ".data"):
            os.remove(data['data_dir'] + "/lp-" + PROJECT_NAME +
                      "-" + str(bugInfo['id']) + ".data")
        with open(data['data_dir'] + "/lp-" + PROJECT_NAME + "-" +
                  str(bugInfo['id']) + ".data", "a", encoding="utf8") as f:
            f.write("bug ID: " + str(bugInfo['id']) + "\n")
            f.write("web link: " + bug.web_link + "\n")
            f.write("bug title: " + bugInfo['title'] + "\n")
            commentsUndecode = browser.get(bugInfo["messages_collection_link"])
            commentsJson = commentsUndecode.decode('utf-8')
            comments = json.loads(commentsJson)
            f.write("bug description:\n")
            f.write("###\n")
            f.write(comments['entries'][0]['content'] + "\n")
            f.write("###\n")
            # f.write("steps to solve the bug:\n")
            commentLen = len(comments['entries'])
            for commentID in range(1, commentLen):
                auther = comments['entries'][commentID]['owner_link']
                auther = auther.rsplit('/', 1)[1][1:]
                f.write("auther: " + auther + "\n")
                f.write("create date: " +
                        comments['entries'][commentID]['date_created'] +
                        "\n")
                f.write("comment " + str(commentID) + ":\n")
                f.write("###\n")
                f.write(comments['entries'][commentID]['content'])
                f.write("\n###\n")
