
from data import DataLoader



#%%
dataloader = DataLoader()
DataLoader(train)

df = df
#%%%
df[['sales_channel_id','customer_id','article_id']].groupby(['customer_id','article_id']).count().reset_index()
#%%
#
# submit process

# Write Submission CSV
sub = pd.read_csv('../input/h-and-m-personalized-fashion-recommendations/sample_submission.csv')
sub = sub[['customer_id']]

#%%



df.article_id.value_counts()#%%

#%%


sub
#%%

sub['customer_id_2'] = sub['customer_id'].apply(lambda x :  int( x[-16:],16))


sub = sub.merge(preds.rename({'customer_id':'customer_id_2'},axis=1), on='customer_id_2',how='left').fillna('')

del sub['customer_id_2']


sub.prediction

sub.prediction = sub.prediction + top12

sub.prediction


sub.prediction = sub.prediction.str.strip()
sub.prediction = sub.prediction.str[:131]

sub['prediction'].values[0]

sub.to_csv(f'submission.csv',index=False)
sub.head()