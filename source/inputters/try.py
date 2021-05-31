import json

data = {'src': [], 'tgt': [], 'cue': []}
with open("train.json", "r", encoding="utf-8") as f:
     datas = json.load(f)
     for data in datas:
        messages = data['messages']
        turn = []
        cue = []
        for message in messages:
            sent = message['message']
            turn.append(sent)

            if 'attrs' in message:
                know=[]
                for j in message['attrs']:
                    knowname = j.get('name')
                    knowrelation = j.get('attrname')
                    knowvalue = j.get('attrvalue')
                    knowfinal = knowname + knowrelation + '是' + knowvalue
                    know.append(knowfinal)
                cue.append(know)
            else:
                cue.append([])
        for i in range(len(turn) - 1):
            posts=''
            for j in range(max(0, (i-9)), i + 1):
                posts = posts+turn[j]
            #prev_post = posts[-1]
            response = turn[i + 1]
            ccue = cue[i+1]
            data.append({'src': posts, 'tgt': response, 'cue':ccue})