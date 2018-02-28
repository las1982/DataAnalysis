class Field:
    review_id = 'review_id'
    review_di_unique = 'review_di_unique'
    review = 'review'
    review_tokenized = 'review_tokenized'
    review_en = 'review_en'
    created_at = 'created_at'
    created_at_day_of_week = 'created_at_day_of_week'
    product_id = 'product_id'
    product_id_unique = 'product_id_unique'
    product_name = 'product_name'
    product_name_tokenized = 'product_name_tokenized'
    product_name_en = 'product_name_en'
    product_category = 'product_category'
    stars = 'stars'


class File:
    working_dir = '/home/alex/work/projects/DataAnalysis/profitero_data_scientist'
    data_dir = '/'.join([working_dir, 'data'])
    models_dir = '/'.join([data_dir, 'models'])
    vectorizers_dir = '/'.join([data_dir, 'vectorizers'])
    jd_reviews = '/'.join([data_dir, 'jd_reviews.csv'])
    product = '/'.join([data_dir, 'product.csv'])
    review = '/'.join([data_dir, 'review.csv'])
    product_en = '/'.join([data_dir, 'product_en.csv'])
    review_en = '/'.join([data_dir, 'review_en.csv'])
    chinese_stop_words = '/'.join([data_dir, 'chinese_stop_words.txt'])
    english_stop_words = '/'.join([data_dir, 'english_stop_words.txt'])


class Language:
    en = 'en'
    ch = 'ch'


class Vectorizers:
    ch_revi_1_5_ngram_count = 'ch_revi_1_5_ngram_count'
    ch_prod_1_2_ngram_count = 'ch_prod_1_2_ngram_count'
    en_revi_1_5_ngram_count = 'en_revi_1_5_ngram_count'
    en_prod_1_2_ngram_count = 'en_prod_1_2_ngram_count'


class Models:
    model_1_sgd = 'model_1_sgd'
    model_2_mpl = 'model_1_mpl'


class Category:
    pet_food = 'pet_food'
    snack = 'snack'
    mobile = 'mobile'
    food = 'food'
    baby_care = 'baby_care'
    other = 'other'


class CategoryWords:
    words = {
        Category.pet_food: {'cat', 'dog', 'pet', 'kitten', 'hills', 'pupp', 'kittie'},
        Category.snack: {'chocolate	candy', 'chocolate', 'candy', 'sugar-free', 'chewing gum', 'mint', 'gum', 'caramel', 'sugar', 'snack',
                         'flavor',
                         'taste'},
        Category.mobile: {'mobile', 'mobile phone', 'cell phone', 'phone'},
        Category.food: {'beef', 'pork', 'meat', 'sausage', 'steak', 'turkey', 'chicken', 'tuna', 'fish', 'salmon'},
        Category.baby_care: {'baby', 'infant', 'children', 'molar', 'heinz'},
        Category.other: {'wedding'}
    }
