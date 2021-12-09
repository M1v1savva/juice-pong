import json

def createLogin(username,password,stat1=0,stat2=0,stat3=0):
    loginInfo = {
        "userID":0,
        "username":username,
        "password":password,
        "stat1":stat1,
        "stat2":stat2,
        "stat3":stat3,
    }

    with open('Game/logins.json','r+') as loginFile:
        data = json.load(loginFile)
        loginInfo["userID"] = len(data["loginInfo"])
        data["loginInfo"].append(loginInfo)
        loginFile.seek(0)
        json.dump(data,loginFile,indent = 4)

def printData():
    logins = open('Game/logins.json')
    data = json.load(logins)

    for i in data['loginInfo']:
        print(i)

def attemptSession(username,password):
    logins = open('Game/logins.json')
    data = json.load(logins)

    for i in data['loginInfo']:
        if i["username"]==username and i["password"] == password:
            return i["userID"]
    return False

def getStatistics(userID):
    logins = open('Game/logins.json')
    data = json.load(logins)

    for i in data['loginInfo']:
        if i["userID"] == userID:
            return i["stat1"],i["stat2"],i["stat3"]


def commmandLineTest():
    print('username')
    username = str(input())
    print('password')
    password = str(input())

    print(attemptSession(username,password))

