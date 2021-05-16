Dt = {}
user_cnt = 0
users = {}
item_cnt = 0
items = {}

with open("data/rating.csv", "r") as f:
    while True:
        line = f.readline()
        if not line:
            break
        line = line.split('\t')
        user_id, item_id, timestamp = int(line[0]), int(line[1]), int(line[3])
        if user_id not in users:
            users[user_id] = user_cnt
            user_cnt += 1
        if item_id not in items:
            items[item_id] = item_cnt
            item_cnt += 1
        user_id = users[user_id]
        item_id = items[item_id]
        if user_id not in Dt:
            Dt[user_id] = []
        Dt[user_id].append([item_id, timestamp])

for user_id in Dt:
    Dt[user_id].sort(key=lambda x: x[1])

with open("data/train.txt", "w") as f1:
    with open("data/test.txt", "w") as f2:
        for user_id in Dt:
            f1.write("%d" % user_id)
            f2.write("%d" % user_id)
            l = len(Dt[user_id])
            for x in Dt[user_id][:-1]:
                f1.write(" %d" % x[0])
            for x in Dt[user_id][-1:]:
                f2.write(" %d" % x[0])

            f1.write('\n')
            f2.write('\n')

