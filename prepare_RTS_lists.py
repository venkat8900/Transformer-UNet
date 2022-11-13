import os

# TODO
# modify code for test set

def prep_list(data_dir, save_dir, split):
    '''
    args:
    - data_dir: path to directory where images and corresponding masks are present
    - save_dir: path to directory where image path is added to text files
    - split: train or test
    '''
    # read all the image names in every dataset
    names_list = []
    for i in range(1, 9):
        
        data_dir_i = data_dir + split + '/' + 'instrument_dataset_' + str(i) + '/images/'
        print(data_dir_i)
        for image in os.listdir(data_dir_i):
            names_list.append(data_dir_i + image)
        
    with open(save_dir + split, 'w') as fp:
        for n in names_list:
            fp.write("%s\n" % n)
        print("[INFO] Added instrument_dataset_{} {} into {}", format(i), format(split), format(save_dir))




if __name__ == "__main__":

    # create save directory and list text files
    if not os.path.exists("./lists/lists_RTS_binary/train.txt"):
        f = open("./lists/lists_RTS_binary/train.txt", "w")
        print("[INFO] train text file created for binary RTS")
    else:
        print("[INFO] train text file exists for binary RTS")
    if not os.path.exists("./lists/lists_RTS_binary/test.txt"):
        f = open("./lists/lists_RTS_binary/test.txt", "w")
        print("[INFO] test text file created for binary RTS")
    else:
        print("[INFO] test text file exists for binary RTS")

    if not os.path.exists("./lists/lists_RTS_instrument/train.txt"):
        f = open("./lists/lists_RTS_instrument/train.txt", "w")
        print("[INFO] train text file created for instrument RTS")
    else:
        print("[INFO] train text file exists for instrument RTS")
    if not os.path.exists("./lists/lists_RTS_instrument/test.txt"):
        f = open("./lists/lists_RTS_instrument/test.txt", "w")
        print("[INFO] test text file created for instrument RTS")
    else:
        print("[INFO] test text file exists for instrument RTS")
    
    if not os.path.exists("./lists/lists_RTS_parts/train.txt"):
        f = open("./lists/lists_RTS_parts/train.txt", "w")
        print("[INFO] train text file created for parts RTS")
    else:
        print("[INFO] train text file exists for parts RTS")
    if not os.path.exists("./lists/lists_RTS_parts/test.txt"):
        f = open("./lists/lists_RTS_parts/test.txt", "w")
        print("[INFO] test text file created for parts RTS")
    else:
        print("[INFO] test text file exists for parts RTS")

    
    task_list = ['binary', 'instrument', 'parts']

    for t in task_list:
        data_dir = './data/RTS_' + t + '/'
        save_dir = './lists/lists_RTS_' + t + '/'

        prep_list(data_dir = data_dir, save_dir = save_dir, split = "train")
        # prep_list(data_dir = data_dir, save_dir = save_dir, split = "test") 
    