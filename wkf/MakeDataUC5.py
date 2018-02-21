import pandas as pd


print (pd.__version__)
# common_path = '/media/alex/work/wf/nielsen/UC5/US_FLAVOURED_DRINKS___CARBONATED/data/'
# found_path = common_path + 'H3_FOUNDCHARS_US_FLAVOURED_DRINKS___CARBONATED_20170703134346_deleted_broken.csv'
# train_in_path = common_path + 'H3_TRAINING_US_FLAVOURED_DRINKS___CARBONATED_JAN00_DEC16_20170707130603.csv'
# test_in_path = common_path + 'H3_TEST_US_FLAVOURED_DRINKS___CARBONATED_JAN17_JUN17_20170629134109.csv'
# gold_path = common_path + 'H3_GOLD_US_FLAVOURED_DRINKS___CARBONATED_JAN17_JUN17_20170629134346.csv'

common_path = '/media/alex/work/wf/nielsen/UC5/US_SPORT___ENERGY_DRINKS/data/'
found_path = common_path + 'H3_FOUNDCHARS_US_SPORT___ENERGY_DRINKS_20170703134412.csv'
train_in_path = common_path + 'H3_TRAINING_US_SPORT___ENERGY_DRINKS_JAN00_DEC16_20170707130616.csv'
test_in_path = common_path + 'H3_TEST_US_SPORT___ENERGY_DRINKS_JAN17_JUN17_20170629134045.csv'
gold_path = common_path + 'H3_GOLD_US_SPORT___ENERGY_DRINKS_JAN17_JUN17_20170629134412.csv'

train_out_path = common_path + 'train.csv'
test_out_path = common_path + 'test.csv'

df_found = pd.read_table(found_path, encoding='iso-8859-1', quotechar='"', sep=',', dtype='str', na_values=["nan"], keep_default_na=False)
print ('foundchar shape is', df_found.shape)
df_train = pd.read_table(train_in_path, encoding='iso-8859-1', quotechar='"', sep=',', dtype='str', na_values=["nan"], keep_default_na=False)
print ('train start shape is', df_train.shape)
# df_test = pd.read_table(test_in_path, encoding='iso-8859-1', quotechar='"', sep=',', dtype='str', na_values=["nan"], keep_default_na=False)
df_gold = pd.read_table(gold_path, encoding='iso-8859-1', quotechar='"', sep=',', dtype='str', na_values=["nan"], keep_default_na=False)
print ('test start shape is', df_gold.shape)


train = pd.merge(df_train, df_found, left_on=['External Code'], right_on=['nielsenUPC'], how='inner', validate='m:m')
print ('train shape is', train.shape)
train.to_csv(train_out_path, encoding='iso-8859-1', index=False, quotechar='"', sep=',')

test = pd.merge(df_gold, df_found, left_on=['External Code'], right_on=['nielsenUPC'], how='inner', validate='m:m')
print ('test shape is', test.shape)
test.to_csv(test_out_path, encoding='iso-8859-1', index=False, quotechar='"', sep=',')
