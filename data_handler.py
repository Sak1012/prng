import pandas as pd

def save_to_csv(data, filename):
    df = pd.DataFrame(data, columns=['x', 'y', 'z', 'w'])
    df.to_csv(filename, index=False)
