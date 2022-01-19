import csv
from tkinter.constants import COMMAND

def submitScore(name='guest',score=0):
    if name == "":
        name = 'guest'
    submission = str(score)+','+name+'\n'
    with open('Game/Scores.csv','a') as fd:
        fd.write(submission)

def getLeaderboard():
    file = open('Game/Scores.csv')
    csvreader = csv.reader(file)
    header = []
    header = next(csvreader)
    data = []
    for row in csvreader:
        data.append(row)
    for row in data:
        row[0] = int(row[0])
    data.sort()
    file.close()
    return data


def commmandLineTest():
    #print('name')
    #name = str(input())
    #print('score')
    #score = str(input())

    data = getLeaderboard()
    buildLeaderboard = ""
    for score in data:
        for i in score:
            buildLeaderboard+=str(i)+' '
        buildLeaderboard+='\n'
    print(buildLeaderboard)
