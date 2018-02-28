class File:
    working_dir = '/home/alex/work/projects/DataAnalysis/profitero_data_scientist'
    data_dir = '/'.join([working_dir, 'data'])
    models_dir = '/'.join([data_dir, 'models'])
    vectorizers_dir = '/'.join([data_dir, 'vectorizers'])
    jd_reviews = '/'.join([data_dir, 'jd_reviews.csv'])
    train = '/'.join([data_dir, 'train.csv'])
    test = '/'.join([data_dir, 'test.csv'])
    output = '/'.join([data_dir, 'output.csv'])
    product_name = '/'.join([data_dir, 'product_name.csv'])
    review = '/'.join([data_dir, 'review.csv'])
    chinese_stop_words = '/'.join([data_dir, 'chinese_stop_words.txt'])
    english_stop_words = '/'.join([data_dir, 'english_stop_words.txt'])


class Field:
    original = 'original'
    review_id = 'review_id'
    review = 'review'
    created_at = 'created_at'
    product_id = 'product_id'
    product_name = 'product_name'
    product_category = 'product_category'
    stars = 'stars'


class Language:
    en = 'en'
    ch = 'ch'


class Vectorizers:
    ch_revi_1_5_ngram_count = 'ch_revi_1_5_ngram_count'
    ch_prod_1_3_ngram_count = 'ch_prod_1_3_ngram_count'
    en_revi_1_5_ngram_count = 'en_revi_1_5_ngram_count'
    en_prod_1_2_ngram_count = 'en_prod_1_2_ngram_count'


class Models:
    model_ch = 'model_ch'
    model_en = 'model_en'


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
