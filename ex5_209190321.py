import numpy as np
import pandas as pd

import imageio.v3 as iio


#######################################################################
# Question 1 - numpy
def get_highest_weight_loss_participant(training_data, participant_names):
    loss = training_data[:,0] - training_data[:,-1]
    return participant_names[np.argmax(loss)]

def get_diff_data(training_data):
    return training_data[:,1:] - training_data[:,:-1]

def get_distance_from_linear_change(training_data):
    initial = training_data[:,0]
    final = training_data[:,-1]
    num = training_data.shape[1]
    linear = np.linspace(initial,final,num,axis=1)
    return training_data[:,1:-1] - linear[:,1:-1]


#######################################################################
# Question 2 - image processing
def np_array_to_ascii(darr):
    return ''.join([chr(item) for item in darr])


def ascii_to_np_array(s):
    return np.frombuffer(s.encode(), dtype=np.uint8)


def arr_dist(a1, a2):
    return np.sum(np.abs(np.int_(a1) - np.int_(a2)))

def find_best_place(im, np_msg):
    k = len(np_msg)
    best = float('inf')
    best_row = 0
    best_col = 0
    for i in range(im.shape[0]):
        for j in range(im.shape[1]-k+1):
            image_array = im[i,j:j+k]
            if arr_dist(np_msg,image_array) < best:
                best = arr_dist(np_msg,image_array)
                best_row = i
                best_col = j
    return (best_row,best_col)

def create_image_with_msg(im, img_idx, np_msg):
    k = len(np_msg)
    a = im.copy()
    a[img_idx[0],img_idx[1]:img_idx[1]+k] = np_msg
    a[0,0] = img_idx[0]
    a[0,1] = img_idx[1]
    a[0,2] = k
    return a

def put_message(im, msg):
    arr = ascii_to_np_array(msg)
    tup = find_best_place(im,arr)
    enc_photo = create_image_with_msg(im,tup,arr)
    return enc_photo

def get_message(im):
    row = im[0,0]
    col = im[0,1]
    k = im[0,2]
    en_text = im[row,col:col+k]
    return np_array_to_ascii(en_text)


#######################################################################
# Question 3 - pandas
def load_weather_csv(file_name):
    return pd.read_csv(file_name)

def impute_to_mean(weather_data):
    copy = weather_data.fillna(weather_data.mean())
    return copy

def fix_to_celsius(weather_data):
    weather_data[weather_data > 40] = (weather_data[weather_data > 40] - 32)*5/9
    return weather_data

# Helper function to run all Question 2 steps in one line.
# feel free to use, but don't change!
def clean_data(weather_data):
    return fix_to_celsius(impute_to_mean(weather_data))


def add_week_index(weather_data):
    a = clean_data(weather_data)
    a["week"] = a.index // 7 + 1
    return a

def get_weekly_mean(weather_data):
    b = add_week_index(weather_data).groupby(['week']).mean()
    return b.iloc[:,1:]


def get_temperature_range(weather_data):
    a = clean_data(weather_data).iloc[:,1:]
    range = a.max() - a.min()
    cities = weather_data.columns[1:].tolist()
    return pd.DataFrame({'City': cities, 'Temperature range': range})

def find_coastal_effect(weather_data, coastal_cities):
    all_data = get_temperature_range(weather_data)
    coastal = all_data[all_data['City'].isin(coastal_cities)]
    nocoast = all_data[~all_data['City'].isin(coastal_cities)]
    mean_coast = coastal['Temperature range'].mean()
    mean_nocoast = nocoast['Temperature range'].mean()
    coastal_effect = mean_coast - mean_nocoast
    return coastal_effect

def add_rainy_days(weather_data):
    city_columns = weather_data.columns
    rainy_days = {city: (weather_data[city] < 20).sum() for city in city_columns}
    rainy_days['Day'] = 0
    rainy_days_df = pd.DataFrame(rainy_days, index=['Rainy Days'])
    weather_data = pd.concat([weather_data, rainy_days_df])
    return weather_data


if __name__ == '__main__':
    def array_compare(a, b, threshold=1e-10):
        if a.shape != b.shape:
            return False
        return np.abs(a - b).max() < threshold

    def df_compare(a, b, threshold=1e-2):
        return array_compare(a.values, b.values, threshold=threshold)

    # Q1 checks
    training_data = np.loadtxt('training_data.csv', delimiter=',')
    participant_names = ['Tali', 'Avi', 'Naomi', 'Shlomi']
    study_months = ['November', 'December', 'January', 'February', 'March', 'April']

    diff_data_expected = np.array([[-2.7, 1.5, -2.7, -2.7, -2.2],
                                   [-4.4, -0.2, -0.7, -1.5, -1.4],
                                   [-1.0, -1.2, 0.6, -0.3, -1.6],
                                   [-2.5, -4.1, -3.1, -2.7, -2.8]])
    distance_from_linear_expected = np.array([[-0.94, 2.32, 1.38, 0.44],
                                              [-2.76, -1.32, -0.38, -0.24],
                                              [-0.3, -0.8, 0.5, 0.9],
                                              [0.54, -0.52, -0.58, -0.24]])

    print(get_highest_weight_loss_participant(training_data, participant_names) == 'Shlomi')
    print(array_compare(get_diff_data(training_data), diff_data_expected))
    print(array_compare(get_distance_from_linear_change(training_data), distance_from_linear_expected))

    # Q2 checks
    sent_message = "thats all folks"
    image_raw = iio.imread('parrot.png')  # read unencrypted image from file

    print(arr_dist(ascii_to_np_array(sent_message), ascii_to_np_array("gettin schwifty")) == 320)
    
    print(find_best_place(image_raw, ascii_to_np_array("show me what you got")) == (126, 92))

    image_enc = put_message(image_raw, sent_message)
    iio.imwrite('parrot_enc.png', image_enc)  # write encrypted image to file
    retrieved_message = get_message(iio.imread('parrot_enc.png'))
    print(retrieved_message == sent_message)

    # Q3 checks
    coastal_cities = ['haifa', 'zichron_yaakov', 'hadera', 'hakfar_hayarok', 'tel_aviv', 'ashdod', 'ashkelon', 'netanya']
    weather_data = load_weather_csv('weather_data_2023.csv')

    print(df_compare(impute_to_mean(weather_data), pd.read_csv('pandas_results/post_impute.csv')))
    print(df_compare(clean_data(weather_data), pd.read_csv('pandas_results/post_clean_data.csv')))

    weekly_mean = get_weekly_mean(weather_data)
    print(df_compare(weekly_mean, pd.read_csv('pandas_results/weekly_mean.csv', index_col=0)))
    print(weekly_mean.index.name == 'week')

    print(find_coastal_effect(weather_data, coastal_cities) - (-2.79714) < 1e-5)
    
    with_rainy_days = add_rainy_days(weather_data)
    print(df_compare(with_rainy_days, pd.read_csv('pandas_results/with_rainy_days.csv', index_col=0)))
    print(with_rainy_days.index[-1] == 'Rainy Days')
