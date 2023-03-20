import pickle
from helpers.preprocessing import read_all_data


data_dict = read_all_data()

imu = data_dict['imu'].to_numpy()
ann = data_dict['ann'].to_numpy().flatten()
print(sum(imu[-1000:]))
print(sum(ann[-10000:]))
truthys = [bool(x != 0 ) for x in ann]
print(len(imu))
print(len(imu[truthys]))

filename = 'DECISION_TREE/dtc_model.sav'
model = pickle.load(open(filename, 'rb'))

out = model.predict(imu[truthys][-2000].reshape(1, -1))
print(out)
print(ann[truthys][-2000])