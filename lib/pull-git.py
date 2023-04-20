import subprocess
import os

origdir = os.getcwd()
gitdir = origdir + '/git-data'
os.chdir("linux")

cmd = os.popen('git log --since="2023-01-01" --grep="^mm:" --grep="^mm/.*:" --oneline')
output = cmd.read()
output_list = output.splitlines()

if os.path.exists(gitdir):
    for f in os.listdir(gitdir):
        fpath = os.path.join(gitdir, f)
        if os.path.isfile(fpath):
            os.remove(fpath)
else:
    os.mkdir(gitdir)

for commit in output_list:
    commit_id = commit.split()[0]
    print('write commit ' + commit_id)
    with open(gitdir + '/' + commit_id + '.data', 'w') as cfile:
        cmd = os.popen('git show ' + commit.split()[0])
        commit_log = cmd.read()
        cfile.write(commit_log.split('diff --git')[0])

