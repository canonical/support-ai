import subprocess
import os

origdir = os.getcwd()
gitdir = origdir + '/git-data'
os.chdir("linux")

#cmd = os.popen('git log --since="2023-01-01" --grep="^mm:" --grep="^mm/.*:" --oneline')
cmd = os.popen('git log --since="2023-01-01" --grep="^net:" --grep="^net/.*:" --oneline | grep -v " Merge branch "')
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
        commit_log = commit_log.split('diff --git')[0]
        commit_log_post = commit_log.split('\n')
        commit_log_post.insert(3, 'content:')
        for i, line in enumerate(commit_log_post):
            if i < 4:
                cfile.write(line + '\n')
            if i >= 4 and i < 50:
                cfile.write(line)

        """
        people = 0
        for i, line in enumerate(commit_log_post):
            if 'Fixes: ' in line:
                people = i
                break
            if 'Signed-off-by:' in line:
                people = i
                break
        commit_log_post.insert(people, 'people:')
        for i, line in enumerate(commit_log_post):
            cfile.write(line + '\n')
        """
