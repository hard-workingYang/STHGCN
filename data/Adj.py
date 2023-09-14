
import numpy as np

def adj_matrix(rowNum, colNum):
    adj = np.zeros((rowNum*colNum,rowNum*colNum))
    dirRow = [0,0,1,-1,-1,-1,1,1]
    dirCol = [1,-1,0,0,-1,1,-1,1]
    for row in range(0,rowNum):
        for col in range(0,colNum):
            for t in range(0,8):
                if row + dirRow[t] >= 0 and row + dirRow[t] < rowNum and \
                col + dirCol[t] >= 0 and col + dirCol[t] < colNum:
                    curPos = row*colNum + col
                    targetPos = (row + dirRow[t])*colNum + col + dirCol[t]
                    if t > 3 :
                        adj[curPos][targetPos] = 1
                        adj[targetPos][curPos] = 1
                    else:
                        adj[curPos][targetPos] = 1
                        adj[targetPos][curPos] = 1
                curPos = row*colNum + col
                adj[curPos][curPos] = 1
    return adj.astype(np.float32)