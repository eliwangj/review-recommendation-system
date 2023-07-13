"""
Method Description:

This model started off as a hybrid recommendation system with a weighted average of item-based and model-based
models. The idea is that if a user has rated a lot of businesses, the approach of using similar businesses to
predict ratings for new businesses (given the same user) would be more trustworthy. However, after experimenting
different weight parameter (alpha), I found out that a pure model-based approach performs the best.
I continued using the XGBRegressor in xgboost from HW3 Task 2.3. The main improvement is realized in two areas:

(1) Feature Engineering
Since our training set is big enough (455k rows), I tried incorporating as many features (that make sense) as
possible in the hope to contain more useful information. Compared with HW3, I exploit business.json as much as
I could here in the competition. Not only did I use all the numeric values, but I also included all the boolean
features in the 'attributes' field of the businesses, such as BikeParking, BusinessAcceptsCreditCards, etc. This
makes sense as most of these attributes do influence customer experience. Here is the full list of the features 
selected:

user.json:      review_count, useful, funny, cool, fans, average_stars
business.json:  stars, review_count, latitude, longitude, is_open, attributes
tip.json:       tip_count (the amount of tips a user/business gives/has)
checkin.json:   average amount of checkins per business


(2) Fine-tuning
A significant amount of effort has been spent on tuning the parameters of this XGBRegressor. My experiments showed
it does have a non-trivial impact on the performance since we are using a pure model-based approach. Here's the full list
of the tuned parameters:
verbosity       = 0
learning_rate   = 0.1            
subsample       = 0.7
gamma           = 0.5
n_estimators    = 200
max_depth       = 7

A few comments:
- I used max_depth=3 in HW3 and found out that the default value is 6. Apparently we need it to be higher.
- n_estimators is a tricky one. Technically speaking the higher it gets, the more likely more accurate it is. 
  But there's a time factor. 200 is appropriate in this case as it improves accuracy the most and the time is still
  acceptable.
- gamma is useful as it specifies the minimum loss reduction required to make a split. It makes the algorithm conservative.


Error Distribution:


RMSE:
0.9792935252391215

Execution Time:
695.7668347358704

"""



#####--------- Import Libraries  ---------#####
from pyspark import SparkContext, SparkConf
import time
import os
import pandas as pd
import json
import numpy as np
import xgboost
import math
import sys


#####--------- Set Values For Parameters  ---------#####
folder_path         = sys.argv[1]
test_file_name      = sys.argv[2]
output_file_name    = sys.argv[3]

train_file_name     = os.path.join(folder_path, "yelp_train.csv")
user_json_name      = os.path.join(folder_path, 'user.json')             # only apply to users
business_json_name  = os.path.join(folder_path, 'business.json')     # only apply to businesses
tip_json_name       = os.path.join(folder_path, "tip.json")               # apply to both users and buinesses
checkin_json_name   = os.path.join(folder_path, "checkin.json")       # only apply to businesses


## Features selected
user_features               = ['review_count', 'useful', 'funny', 'cool', 'fans', 'average_stars', 'tip_count']
business_features           = ['stars', 'review_count', 'latitude', 'longitude', 'is_open', 'tip_count', 'checkin_count']
business_attributes_bool    = ['BusinessAcceptsCreditCards', 'BikeParking', 'GoodForKids', 'RestaurantsTakeOut', 'OutdoorSeating', 'RestaurantsGoodForGroups', 'WheelchairAccessible', 'RestaurantsDelivery', 'RestaurantsReservations', 'HasTV', 'ByAppointmentOnly']
business_attributes_num     = ['RestaurantsPriceRange2']
restaurant_attire_dict      = {'casual' : 1, 'formal' : 2, 'dressy' : 3}


#####--------- Define Helper Functions  ---------#####
def calc_pearson(business_id_i, business_id_j):
    """Calculate the Pearson Correlation between business i(current), j(to-be-compared).
        Essentially to find the similarities between them based on co-rated users' ratings.
        We need:
            r_{u, i}: rating by user u on business i
            r_i bar: average rating of business i by co-rated users
            r_{u, j}: rating by user u on business j
            r_j bar: average rating of business j by co-rated users
    Returns
    -------
    _type_
        _description_
    """
    ## Find co-rated users
    users_rated_i = business_record[business_id_i].keys()
    users_rated_j = business_record[business_id_j].keys()
    co_rated_users = set(users_rated_i).intersection(set(users_rated_j))

    ### **************** This part can be improved **************** ###
    ## If there's no user that rated both business i and j
    if len(co_rated_users) == 0:
        ratings_i_avg = business_avgs[business_id_i]
        ratings_j_avg = business_avgs[business_id_j]
        return float(min(ratings_i_avg, ratings_j_avg) / max(ratings_i_avg, ratings_j_avg))
    

    ## Find ratings from co-rated users
    ratings_i = [business_record[business_id_i][user] for user in co_rated_users]
    ratings_j = [business_record[business_id_j][user] for user in co_rated_users]

    
    ## Method 1: Average rating by co-rated users
    ratings_i_avg = sum(ratings_i) / len(ratings_i)
    ratings_j_avg = sum(ratings_j) / len(ratings_j)

    numerator = 0
    denominator_i = 0
    denominator_j = 0

    for n in range(len(co_rated_users)):
        numerator += (ratings_i[n] - ratings_i_avg) * (ratings_j[n] - ratings_j_avg)
        denominator_i += (ratings_i[n] - ratings_i_avg) * (ratings_i[n] - ratings_i_avg)
        denominator_j += (ratings_j[n] - ratings_j_avg) * (ratings_j[n] - ratings_j_avg)

    denominator = math.sqrt(denominator_i * denominator_j)

    if denominator == 0:
        if numerator == 0:
            pearson = 1
        else:
            pearson = -1
    else:
        pearson = numerator / denominator
    
    return pearson

def check_case(user_id, business_id):
    """ Check which situation this user_id, business_id combination belongs to"""
    ## Default Rating Mechanism 1
    if user_id not in user_avgs and business_id in business_avgs:
        case = "USER_UNKNOWN"
    ## Default Rating Mechanism 2
    elif user_id in user_avgs and business_id not in business_avgs:
        case = "BIZ_UNKNOWN"
    elif user_id not in user_avgs and business_id not in business_avgs:
        case = "BOTH_UNKNOWN"
    elif user_id in user_avgs and business_id in business_avgs:
        case = "NORMAL"
    return case

def calc_prediction(weights):
    """
    Calculate the weighted average rating given a weight vector (and the ratings).
    Given tuples of (w_{i, n}, r_{u, n}), compute the final predicted rating
    """
    limit = min(30, len(weights))
    weights.sort(key = lambda x: x[0], reverse = True)
    weights = weights[:limit]
    numerator = 0
    denominator = 0

    for i in range(limit):
        weight = weights[i][0]
        rating = weights[i][1]
        numerator += weight * rating
        denominator += abs(weight)

    prediction = numerator / denominator

    return prediction

def calc_weight(num_neighbors, min_num, max_num):
    """Calculate the alpha in computing the weighted average using Min-Max scaling method"""
    return float((num_neighbors - min_num) / (max_num - min_num))

def item_cf(user_id, business_id):
    which_case = check_case(user_id, business_id)
    ## Default Rating Mechanism 1
    if which_case == "USER_UNKNOWN":
        pred_rating = business_avgs[business_id]
        alpha = 0.2

    ## Default Rating Mechanism 2
    elif which_case == "BIZ_UNKNOWN":
        pred_rating = user_avgs[user_id]
        alpha = 0.2

    ## Default Rating Mechanism 3
    elif which_case == "BOTH_UNKNOWN":
        pred_rating = global_avg
        alpha = 0

    ## Normal case: Use Item-based CF
    elif which_case == "NORMAL":
        weights = []        # store the pearson correlation weights between this business and other businesses that this user has rated
        rated_businesses = user_record[user_id].keys()
        alpha = calc_weight(len(rated_businesses), min_num, max_num)
        for to_comp_biz in rated_businesses:    # to-be-compared business
            if to_comp_biz == business_id:
                continue
            pearson_weight = calc_pearson(business_id, to_comp_biz)
            to_comp_biz_rating = user_record[user_id][to_comp_biz]
            if pearson_weight > 1:
                pearson_weight = 1 / pearson_weight
            if pearson_weight > 0:
                weights.append((pearson_weight, to_comp_biz_rating))

        pred_rating = calc_prediction(weights)

    return user_id, business_id, pred_rating, alpha

def get_sentiment_score(text_list, st_analyzer):
    sentiment_score = 0

    for text in text_list:
        sentiment_score_dict = st_analyzer.polarity_scores(text)
        sentiment_score += sentiment_score_dict['compound']

    avg_sentiment_score = sentiment_score / len(text_list)
    
    return avg_sentiment_score

def add_tip_features(profile_dict, tip_count_dict):
    for id in profile_dict:
        if id in tip_count_dict:        # if an id is in tip_count_dict, then it has to be in tip_score dict as well
            profile_dict[id] = profile_dict[id] + (tip_count_dict[id],)

        else:
            profile_dict[id] = profile_dict[id] + (0,)

    return profile_dict

def add_checkin_features(bid_profile_dict, checkin_dict):
    for bid in bid_profile_dict:
        if bid in checkin_dict:
            bid_profile_dict[bid] = bid_profile_dict[bid] + (checkin_dict[bid],)
        else:
            bid_profile_dict[bid] = bid_profile_dict[bid] + (0,)
    return bid_profile_dict

def combine_business_attributes(bid, business_attr_dict, features_row):
    if bid in business_attr_dict:
        attr = business_attr_dict.get(bid)
        if attr:
            for key in business_attributes_bool:
                features_row.append(int(bool(attr[key]))) if key in attr else features_row.append(0)
            for key in business_attributes_num:
                features_row.append(int(attr[key])) if key in attr else features_row.append(0)

            #Business Parking
            features_row.append(attr.get('BusinessParking').count('True')) if attr.get('BusinessParking') else features_row.append(0)
            #Alcohol
            features_row.append(1) if attr.get('Alcohol') and attr.get('Alcohol') != 'none' else features_row.append(0)
            #Ambience
            features_row.append(attr.get('Ambience').count('True')) if attr.get('Ambience') else features_row.append(0)
            #Noise Level
            features_row.append(1) if attr.get('NoiseLevel') and attr.get('NoiseLevel') == 'average' else features_row.append(0)
            # Restaurants Attire
            features_row.append(restaurant_attire_dict.get(attr.get('RestaurantAttire'))) if attr.get('RestaurantAttire') and attr.get('RestaurantAttire') != 'none' else features_row.append(0)
            # Wifi
            features_row.append(1) if attr.get('WiFi') == 'Free' else features_row.append(0)
            #GoodForMeal
            features_row.append(attr.get('GoodForMeal').count('True')) if attr.get('GoodForMeal') else features_row.append(0)

        else:
            features_row.extend([None] * (len(business_attributes_bool) + len(business_attributes_num) + 7))
                        
    else:
        features_row.extend([None] * (len(business_attributes_bool) + len(business_attributes_num) + 7))

    return features_row

def combine_features(df, uid_profile_dict, bid_profile_dict):
    """
    Return numpy array of features, the formatted data to feed XGBRegressor.
    The multi-dimensional numpy array looks like:
    > train_x
    > array([[ 8.71745216,  5.39772091,  0.31337483, ...,  3.60076903,
              27.57968709,  3.25497734],
              ...,
             [ 8.71745216,  5.39772091,  0.31337483, ...,  3.60076903,
              27.57968709,  3.25497734]])
    > train_x.shape
    > (9117, 8)
    9117: number of rows
    8: number of dimensions
    """
    num_rows = len(df)
    all_features_values = [[] for _ in range(num_rows)]  # store feature values for each row of entry

    for i in range(num_rows):
        uid = df['user_id'][i]
        bid = df['business_id'][i]

        ## Append user features values first
        if uid in uid_profile_dict.keys():
            for j in range(len(user_features)):
                all_features_values[i].append(uid_profile_dict[uid][j])
        else:
            for j in range(len(user_features)):
                feature_name = user_features[j]
                all_features_values[i].append(default_user[feature_name])

        ## Then append business features values (numeric)
        if bid in bid_profile_dict.keys():
            for j in range(len(business_features)):
                all_features_values[i].append(bid_profile_dict[bid][j])
        else:
            for j in range(len(business_features)):
                feature_name = business_features[j]
                all_features_values[i].append(default_business[feature_name])

        ## Lastly append business attributes
        all_features_values[i] = combine_business_attributes(bid, business_attr_dict, all_features_values[i])

    return np.array(all_features_values)



if __name__ == "__main__":
    #####--------- Configure PySpark  ---------#####
    ## Create a SparkContext to connect the spark cluster
    conf = (
        SparkConf()
        .setMaster("local[*]")
        .setAppName("competition-v5")
        .set("spark.executor.memory", "4g")
        .set("spark.driver.memory", "4g")
        .set("spark.default.parallelism", "4")
    )
    sc = SparkContext(conf=conf)
    sc.setLogLevel('ERROR')

    ## Start the timer
    start_time = time.time()

    rdd_train = sc.textFile(train_file_name)
    header_train = rdd_train.first()
    rdd_train = rdd_train.filter(lambda x: x != header_train).map(lambda x: x.split(',')) \
                         .map(lambda x: (x[0], x[1], float(x[2]))).cache()

    
    #####--------- Extract Useful Info  ---------#####
    ## Default Rating Mechanism 1
    ## Average rating for each business. For predicting a new user's rating on a known business.
    business_avgs = rdd_train.map(lambda x: (x[1], x[2])).groupByKey().mapValues(lambda x: sum(x) / len(x)).collectAsMap()
    
    ## Default Rating Mechanism 2
    ## Average rating done by each user. For predicting rating on a new business by a known user.
    user_avgs = rdd_train.map(lambda x: (x[0], x[2])).groupByKey().mapValues(lambda x: sum(x) / len(x)).collectAsMap()

    ## Default Rating Mechanism 3
    ## Average rating of the entire database. Both user and business are unknown.
    global_avg = rdd_train.map(lambda x: x[2]).mean()

    ## Each user and all the businesses & corresponding ratings this user has rated
    user_record = rdd_train.map(lambda x: (x[0], (x[1], x[2]))).groupByKey().mapValues(dict).collectAsMap()
    dist_neighbors = [len(sub_dict) for sub_dict in user_record.values()] #改了
    min_num = min(dist_neighbors)
    max_num = max(dist_neighbors)

    ## Each business and all the users & corresponding ratings on this business
    business_record = rdd_train.map(lambda x: ((x[1]), ((x[0]), x[2]))).groupByKey().mapValues(dict).collectAsMap()

    ## Use broadcast variables to reduce overhead
    business_avgs_bc = sc.broadcast(business_avgs)
    user_avgs_bc = sc.broadcast(user_avgs)
    user_record_bc = sc.broadcast(user_record)
    business_record_bc = sc.broadcast(business_record)


    #####----------------------------------------------#####
    #####------------ Perform Item-based CF -----------#####
    #####----------------------------------------------#####
    """
    Here 'item-based' is essentially 'business-based'.
    Which means we match a customer's rated businesses to target business
    """
    ## Load Validation Data
    rdd_val = sc.textFile(test_file_name)
    header_val = rdd_val.first()
    rdd_val = rdd_val.filter(lambda x: x != header_val).map(lambda x: x.split(',')) \
                     .map(lambda x: (x[0], x[1], float(x[2]))).cache()

    ## Map predicting function to every row in the validation/test dataset
    itembased_predictions = rdd_val.sortBy(lambda x: (x[0], x[1])).map(lambda x: item_cf(x[0], x[1]))


    #####----------------------------------------------#####
    #####-------- Perform Model-based Approach --------#####
    #####----------------------------------------------#####

    #####--------- Load training and test data for XGBRegressor  ---------#####
    train_df = pd.read_csv(train_file_name)
    test_df = pd.read_csv(test_file_name)

    rdd_user_profile = sc.textFile(user_json_name).map(json.loads) \
                    .map(lambda x: (x['user_id'], 
                                    (x['review_count'], x['useful'], x['funny'], x['cool'], x['fans'], x['average_stars']))).cache()

    rdd_business_profile = sc.textFile(business_json_name).map(json.loads) \
                    .map(lambda x: (x['business_id'], 
                                    (x['stars'], x['review_count'], x['latitude'], x['longitude'], x['is_open']))).cache()
    
    # uid sample: 'L1MYSq2IH3hkY_eCkBHCnw'
    uid_profile_dict = rdd_user_profile.collectAsMap()
    bid_profile_dict = rdd_business_profile.collectAsMap()



    #####--------- Process tip data and add tip features  ---------#####
    rdd_tip = sc.textFile(tip_json_name).map(json.loads)
    ## Tip Feature 1: count
    user_tip_count = rdd_tip.map(lambda x: (x['user_id'], 1)).reduceByKey(lambda x, y: x + y)
    business_tip_count = rdd_tip.map(lambda x: (x['business_id'], 1)).reduceByKey(lambda x, y: x + y)
    user_tip_count_dict = user_tip_count.collectAsMap()
    business_tip_count_dict = business_tip_count.collectAsMap()

    ## Add tip features: count and sentiment score
    uid_profile_dict = add_tip_features(uid_profile_dict, user_tip_count_dict)
    bid_profile_dict = add_tip_features(bid_profile_dict, business_tip_count_dict)
    

    #####--------- Process Checkin Data And Add Checkin Features  ---------#####
    rdd_checkin = sc.textFile(checkin_json_name).map(json.loads)
    business_checkin_count = rdd_checkin.map(lambda x: (x['business_id'], (sum(x['time'].values()) / len(x['time'].values()))))
    business_checkin_count_dict = business_checkin_count.collectAsMap()

    ## Add checkin feature to business profiles
    bid_profile_dict = add_checkin_features(bid_profile_dict, business_checkin_count_dict)


    #####--------- Process business attributes data and add business attributes as features  ---------#####
    ## Gotta put this one at last because all the previous features are added in combine_features() in order
    rdd_business_attr = sc.textFile(business_json_name).map(json.loads) \
                        .map(lambda x: (x['business_id'], x['attributes']))
    business_attr_dict = rdd_business_attr.collectAsMap()


    #####--------- Define Default Mechanism For User And Business Features  ---------#####
    ## Reconsider your default mechanism, averge or simply 0 (maybe better when it's sparse)?
    default_user = {'review_count': rdd_user_profile.map(lambda x: x[1][0]).mean(),
                    'useful': rdd_user_profile.map(lambda x: x[1][1]).mean(),
                    'funny': rdd_user_profile.map(lambda x: x[1][2]).mean(),
                    'cool': rdd_user_profile.map(lambda x: x[1][3]).mean(),
                    'fans': rdd_user_profile.map(lambda x: x[1][4]).mean(),
                    'average_stars': rdd_user_profile.map(lambda x: x[1][5]).mean(),
                    'tip_count': user_tip_count.map(lambda x: x[1]).mean(),
                    }

    default_business = {'stars': rdd_business_profile.map(lambda x: x[1][0]).mean(),
                        'review_count': rdd_business_profile.map(lambda x: x[1][1]).mean(),
                        'latitude': rdd_business_profile.filter(lambda x: x[1][2] is not None).map(lambda x: x[1][2]).mean(), # there are none values in 'latitude' and 'longitude'
                        'longitude': rdd_business_profile.filter(lambda x: x[1][3] is not None).map(lambda x: x[1][3]).mean(),
                        'is_open': rdd_business_profile.map(lambda x: x[1][4]).mean(),
                        'tip_count': business_tip_count.map(lambda x: x[1]).mean(),
                        'checkin_count': business_checkin_count.map(lambda x: x[1]).mean()
                        }
    

    #####--------- Format training and test data  ---------#####
    ## Format training data to feed into the regressor
    train_x = combine_features(train_df, uid_profile_dict, bid_profile_dict)
    train_y = np.array(train_df['stars'])

    ## Format testing data for prediction
    test_x = combine_features(test_df, uid_profile_dict, bid_profile_dict)
    test_y = np.array(test_df['stars'])


    #####--------- Fit the regressor and do predictions  ---------#####
    reg = xgboost.XGBRegressor(objective='reg:linear',
                               verbosity=0,
                               learning_rate=0.1, 
                               subsample=0.7,
                               gamma=0.5,
                               n_estimators=200,
                               max_depth=7,
                               seed=553)
    reg.fit(train_x, train_y)
    modelbased_predictions = reg.predict(test_x)

    model_uidbid_pred = {}
    for i in range(len(modelbased_predictions)):
        uid = test_df['user_id'][i]
        bid = test_df['business_id'][i]
        model_uidbid_pred[(uid, bid)] = modelbased_predictions[i]

    alpha = 0
    # weighted_predictions = itembased_predictions.map(lambda x: ((x[0], x[1]), x[3] * x[2] + (1-x[3]) * model_uidbid_pred[(x[0], x[1])])).collect()
    weighted_predictions = itembased_predictions.map(lambda x: ((x[0], x[1]), alpha * x[2] + (1-alpha) * model_uidbid_pred[(x[0], x[1])])).collect()

    ## End the timer
    print("Duration:", time.time() - start_time)


    #####----------------------------------------------#####
    #####----- Performance Evaluation Using RMSE ------#####
    #####----------------------------------------------#####
    num_val = rdd_val.count()
    tuple_rating_dict = rdd_val.map(lambda x: ((x[0], x[1]), x[2])).collectAsMap()
    sum_square = 0
    for (user_id, business_id), pred_rating in weighted_predictions:
        ub_tuple = (user_id, business_id)
        real_rating = tuple_rating_dict.get(ub_tuple)
        sum_square += (float(pred_rating) - real_rating)**2
    rmse = math.sqrt(sum_square/num_val)
    print("RMSE: ", rmse)


    #####----------------------------------------------#####
    #####--------------- Export Output ----------------#####
    #####----------------------------------------------#####
    with open(output_file_name, 'w') as outfile:
        outfile.write("user_id, business_id, prediction\n")
        for line in range(len(weighted_predictions)):
            outfile.write(str(weighted_predictions[line][0][0]) + "," + \
                          str(weighted_predictions[line][0][1]) + "," + \
                          str(weighted_predictions[line][1]) + "\n")

    ## Stop the current SparkContext
    sc.stop()

