def printHelp():
    print('train - train neural network;\nidentify - identify JPEG images in current folder;\nexit - end script')

printHelp()
while True:
    cmd = input()
    if cmd.lower() == 'train':
        exec(open("TrainNeuralNetwork.py").read())
    elif cmd.lower() == 'identify':
        exec(open("IdentifyImages.py").read())
    elif cmd.lower() == 'exit':
        exit()
    else:
        printHelp()