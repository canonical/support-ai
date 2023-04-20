
import launchpadlib
from launchpadlib.launchpad import Launchpad
import json
import sys
import yaml
import os
from datetime import datetime as dt

CONFIG_PATH = "config.yaml"
project_name = "linux"

with open(CONFIG_PATH, 'r') as f:
    data = yaml.load(f, Loader=yaml.FullLoader)

if 'data_dir' not in data or 'model_path' not in data:
    sys.exit("Unexpected config format [{}]".format(data))

if not os.path.exists(data['data_dir']):
    os.mkdir(data['data_dir'])


launchpad = Launchpad.login_anonymously('test', 'production', version = 'devel')

project = launchpad.projects[project_name]
bugs = project.searchTasks(status = ['Fix Committed', 'Fix Released'])
#bugs = project.searchTasks(status = ['New', 'Incomplete', 'Triaged', 'Opinion', 'Invalid', 'Won\'t Fix', 'Confirmed', 'In Progress', 'Fix Committed', 'Fix Released'])


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
        if os.path.exists(data['data_dir'] + "/lp-" + project_name + "-" + str(bugInfo['id']) + ".data"):
            os.remove(data['data_dir'] + "/lp-" + project_name + "-" + str(bugInfo['id']) + ".data")
        lpfile = open(data['data_dir'] + "/lp-" + project_name + "-" + str(bugInfo['id']) + ".data", "a")
        lpfile.write("bug ID: " + str(bugInfo['id']) + "\n")
        lpfile.write("web link: " + bug.web_link + "\n")
        lpfile.write("bug title: " + bugInfo['title'] + "\n")
        commentsUndecode = browser.get(bugInfo["messages_collection_link"])
        commentsJson = commentsUndecode.decode('utf-8')
        comments = json.loads(commentsJson)
        lpfile.write("bug description:\n")
        lpfile.write("###\n")
        lpfile.write(comments['entries'][0]['content'] + "\n")
        lpfile.write("###\n")
        #lpfile.write("steps to solve the bug:\n")
        commentLen = len(comments['entries'])
        for commentID in range(1, commentLen):
            auther = comments['entries'][commentID]['owner_link']
            auther = auther.rsplit('/', 1)[1][1:]
            lpfile.write("auther: " + auther + "\n")
            lpfile.write("create date: " + comments['entries'][commentID]['date_created'] + "\n")
            lpfile.write("comment " + str(commentID) + ":\n")
            lpfile.write("###\n")
            lpfile.write(comments['entries'][commentID]['content'])
            lpfile.write("\n###\n")
