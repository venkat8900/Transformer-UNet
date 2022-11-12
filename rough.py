import os

if not os.path.exists("./lists/lists_RTS_binary/train.txt"):
    f = open("./lists/lists_RTS_binary/train.txt", "w")
    print("[INFO] train text file created for binary RTS")
else:
    print("[INFO] train text file exists for binary RTS")