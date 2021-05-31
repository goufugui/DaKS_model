import torch
import torch.nn.functional as F
import src.args as args
import src.utils as utils
from src.model import KnowledgeSelector

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
train_dataset, dev_dataset, test_dataset = utils.get_datasets()
test_batch_dataset = utils.get_batch_data(test_dataset, batch_size=1, shuffle=False)
model = KnowledgeSelector().to(device)
model.load_state_dict(torch.load(args.ckpt_dir + 'best_model.ckpt'))


def test():
    model.eval()
    num_success, num_fail = 0, 0
    with torch.no_grad():
        for batch_data in test_batch_dataset:
            output = model(input_ids=batch_data['input_ids'].to(device),
                           token_type_ids=batch_data['token_type_ids'].to(device)
                           , attention_mask=batch_data['attention_mask'].to(device))
            if abs(F.softmax(output.squeeze(0), dim=0)[1].item() - batch_data['label'][0].item()) < 0.5:
                num_success += 1
            else:
                num_fail += 1
            print(abs(F.softmax(output.squeeze(0), dim=0)[1].item() - batch_data['label'][0].item()))
        print('num_success:', num_success)
        print('num_fail:', num_fail)
        print('success_rate', num_success / (num_success + num_fail))


if __name__ == '__main__':
    test()