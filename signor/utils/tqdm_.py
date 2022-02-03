from tqdm import tqdm

if __name__ == '__main__':
    import time
    for i,j in enumerate(tqdm(range(100))):
        time.sleep(0.01)
