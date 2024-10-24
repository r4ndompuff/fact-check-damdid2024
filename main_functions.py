def get_period_range(start, end):
    real_start = []
    real_end = []
    # Years
    if 'year' in start and 'year' in end:
        start_y = int(start['year'])
        end_y = int(end['year'])
        if start_y < 100 or end_y > 2024:
            return {}
        if start_y > end_y:
            return {}
        real_start.append(start_y)
        real_end.append(end_y)
    else:
        return {}
    # Months
    if 'month' in start and 'month' in end:
        start_m = int(start['month'])
        end_m = int(end['month'])
        if start_m < 1 or end_m > 12:
            return {}
        real_start.append(start_m)
        real_end.append(end_m)
    # Days
    if 'day' in start and 'day' in end:
        start_d = int(start['day'])
        end_d = int(end['day'])  
        if start_d < 1 or end_d > 31:
            return {}
        real_start.append(start_d)
        real_end.append(end_d)
    return custom_date_range(real_start, real_end)

# В данном коде используется функция custom\_date\_range(), которая строит все возможные даты, которые входят в заданный период.
def custom_date_range(start, end):
    # start = [year, month, day]
    if len(start) != len(end):
        return {}
    res = {}
    for i in range(len(start)):
        if i == 0: # year starting
            res = {year:[] for year in list(range(start[0], end[0]+1))}
        if i == 1: # month
            if len(res.keys()) == 1: # only one year
                res[list(res.keys())[0]] = {month:[] for month in 
                list(range(start[1], end[1]+1))}
            else:
                for year_num, year in enumerate(res.keys()):
                    if year_num == 0: # first year
                        res[year] = {month:[] for month in 
                        list(range(start[1], 13))}
                    elif year_num == len(res)-1:
                        res[year] = {month:[] for month in 
                        list(range(1, end[1]+1))}
                    else:
                        res[year] = {month:[] for month in 
                        list(range(1, 13))}
        if i == 2: # days
            if len(res) == 1: # only one year
                year = list(res.keys())[0]
                if len(res[year]) == 1: # and only one month
                    month = list(res[year].keys())[0]
                    res[year][month] = list(range(start[2], end[2]+1))
            else:
                for year_num, year in enumerate(res.keys()):
                    for month_num, month in enumerate(res[year].keys()):
                        if year_num == 0 and month_num == 0:
                            res[year][month] = list(range(start[2], 32))
                        elif year_num == len(res.keys())-1 and 
                            month_num == len(res[year].keys())-1:
                            res[year][month] = list(range(1, end[2]+1))
                        else:
                            res[year][month] = list(range(1, 32))
                
    return res


# Обновление глобального дерева (базы) совершается посредством прохода обоих деревьев и обновления информации в каждом узле.
def update_global_tree(gl_tree, new_tree, id):
    # gl_tree - year: [#ids, [ids], {month:...}]
    # Year
    for year in new_tree:
        if year not in gl_tree: # First ever year in db
            gl_tree[year] = [1, [id], {}]
        else:
            gl_tree[year][0] += 1
            gl_tree[year][1].append(id)
        # Month
        if len(new_tree[year]) > 0:
            for month in new_tree[year]:
                if month not in gl_tree[year][2]:
                    gl_tree[year][2][month] = [1, [id], {}]
                else:
                    gl_tree[year][2][month][0] += 1
                    gl_tree[year][2][month][1].append(id)
                # Day
                if len(new_tree[year][month]) > 0:
                    for day in new_tree[year][month]:
                        if day not in gl_tree[year][2][month][2]:
                            gl_tree[year][2][month][2][day] = [1, [id]]
                        else:
                            gl_tree[year][2][month][2][day][0] += 1
                            gl_tree[year][2][month][2][day][1].append(id)
    return gl_tree


# Код сгущающего контрастного обучения
from sentence_transformers import SentenceTransformer, InputExample, losses, evaluation
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
import pandas as pd
import torch

model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

CFG = {
    'batch_size': 32,
    'seed': 42
}

def run(model, data_path, save_path, epochs=5):
    df = pd.read_csv(data_path)
    train, val = train_test_split(df,
                                 train_size=0.8,
                                 stratify=df['target'],
                                 shuffle=True,
                                 random_state=CFG['seed'])
    train = train.reset_index(drop=True)
    val = val.reset_index(drop=True)

    train_examples = []
    for i in range(train.shape[0]):
        example = train.iloc[i]
        train_examples.append(
            InputExample(texts=[example['text_1'], 
                                example['text_2']], 
            label=example['target'])
        )
    train_loader = torch.utils.data.DataLoader(train_examples, 
        shuffle=True, 
        batch_size=CFG['batch_size'])

    evaluator = evaluation.BinaryClassificationEvaluator(
        sentences1=val['text_1'].values.tolist(),
        sentences2=val['text_2'].values.tolist(),
        labels=val['target'].values.tolist(),
        batch_size=CFG['batch_size'],
        show_progress_bar=False,
        name='eval_wiki',
        write_csv=True
    )
    def cb(score, epoch, steps):
        print(score, epoch, steps)

    criterion = losses.ContrastiveLoss(model=model)
    model.fit(train_objectives=[(train_loader, criterion)],
             evaluator=evaluator,
             epochs=epochs,
             show_progress_bar=True,
             output_path=f'train_model/{save_path}',
             save_best_model=True,
             callback=cb)

# 1000
print('# '*7+'Thousand'+" #"*7)
model = SentenceTransformer(model_name, device='cuda')
run(model, 'train_data/wiki_thousand.csv', 'thousand', epochs=5)
# 100
print('# '*7+'Century'+" #"*7)
model = SentenceTransformer(f'train_model/thousand', device='cuda')
run(model, 'train_data/wiki_hundread.csv', 'hundread', epochs=5)
# 10
print('# '*7+'Decade'+" #"*7)
model = SentenceTransformer(f'train_model/hundread', device='cuda')
run(model, 'train_data/wiki_decade.csv', 'decade', epochs=5)
# 1
print('# '*7+'Year'+" #"*7)
model = SentenceTransformer(f'train_model/decade', device='cuda')
run(model, 'train_data/wiki_year.csv', 'year', epochs=5)
# 1/12
print('# '*7+'Month'+" #"*7)
model = SentenceTransformer(f'train_model/year', device='cuda')
run(model, 'train_data/wiki_month.csv', 'month', epochs=5)
# 1/365
print('# '*7+'Day'+" #"*7)
model = SentenceTransformer(f'train_model/month', device='cuda')
run(model, 'train_data/wiki_day.csv', 'day', epochs=3)