def main():
    if __name__=="__main__":
        gameActive = True
        playerTurn = False
        botID = 0
        humanID = 1
        while gameActive:
            if playerTurn:
                performAction(playerID=humanID)
            else:
                performAction(playerID=botID)
            playerTurn = not playerTurn
            print(playerTurn)

def performAction(playerID):
    if playerID==0:
        performRobotAction()
    else:
        performHumanAction(playerID)

def performRobotAction():
    return

def performHumanAction(playerID):
    return

main()

