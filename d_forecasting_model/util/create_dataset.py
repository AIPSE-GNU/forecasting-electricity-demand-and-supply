import pandas as pd

def concat_data(data, n_in, n_out):
    immed = []
    name = []

    df = pd.DataFrame(data)

    if type(data) ==list:
        n_var = 1
    else:
        n_var = data.shape[1]

    for i in range(n_in, 0, -1):
        immed.append(df.shift(i))
        name += ['var{}(t-{})'.format(j+1, i) for j in range(n_var)]

    for i in range(0, n_out):
        immed.append(df.shift(-i))
        if i==0:
            name += ['var{}(t)'.format(j+1) for j in range(n_var)]
        else:
            name += ['var{}(t+{})'.format(j+1, i) for j in range(n_var)]

    # print(immed)
    concat = pd.concat(immed, axis=1)
    concat.columns = name
    try:
        concat.dropna(axis=0, inplace=True)
    except:
        pass

    concat = concat.values

    return concat