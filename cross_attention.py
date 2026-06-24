from datasets import load_dataset
from transformers import BertTokenizer
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
from torch.utils.data import TensorDataset, DataLoader

tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
model = BertModel.from_pretrained('bert-base-multilingual-cased')
ds = load_dataset("rajpurkar/squad")
pick=ds['train']['question'][:50000]
get=ds['train']['answers']["text"][:50000]
answers=[get[i]for i in range(len(get))]
quastions=[pick[i] for i in range(len(pick))]
answers=[answers[i][0] for i in range(len(answers))]
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
token=tokenizer(
    quastions,
    padding=True,
    truncation=True,
    max_length=128,
    return_tensors='pt'
)
quastions = token['input_ids'] 
quastions_mask = token['attention_mask']
token2=tokenizer(
    answers,
    padding=True,
    truncation=True,
    max_length=128,
    return_tensors='pt'
)
answers=token2['input_ids'] 
answers_current = answers[:, :-1]
answers_future = answers[:, 1:]
answers_mask = token2['attention_mask']
answers_current_mask = answers_mask[:, :-1]
answers_future_mask = answers_mask[:, 1:]
class encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder= model
    def forward (self , x,mask):
        x=model(input_ids=x,attention_mask=mask)
        x=x.last_hidden_state
        return x
class DecoderLayer(nn.Module):
    def __init__(self, embad_size=768, vocab_size=119547, ff_output=3072, num_heads=12):
        super().__init__()
        self.head = nn.MultiheadAttention(embad_size, num_heads, batch_first=True)
        self.cross_head = nn.MultiheadAttention(embad_size, num_heads, batch_first=True)
        self.agent = nn.Linear(embad_size, ff_output)
        self.boss1 = nn.Linear(ff_output, embad_size)
        self.norm1 = nn.LayerNorm(embad_size)
        self.norm2 = nn.LayerNorm(embad_size)
        self.norm3 = nn.LayerNorm(embad_size)
        self.dropout = nn.Dropout(0.3)
    def forward(self, x, mask, bert_output, encoder_mask):
        x_length = x.size(1)
        training_mask = torch.triu(torch.ones(x_length, x_length), diagonal=1).bool().to(x.device)
        original = x
        x, _ = self.head(x, x, x, attn_mask=training_mask, key_padding_mask=mask)
        x = self.norm1(x + original)
        original = x
        x, _ = self.cross_head(query=x, key=bert_output, value=bert_output, key_padding_mask=encoder_mask)
        x = self.norm2(x + original)
        linear = self.dropout(nn.functional.gelu(self.agent(x)))
        linear = self.boss1(linear)
        x = self.norm3(linear + x)
        return x
class Decoder(nn.Module):
    def __init__(self, num_layers=6, embad_size=768, vocab_size=len(tokenizer)):
        super().__init__()
        self.embading = nn.Embedding(vocab_size, embad_size)
        self.gps = nn.Parameter(torch.randn(1, 127, embad_size))
        self.layers = nn.ModuleList([DecoderLayer() for _ in range(num_layers)])
        self.boss2 = nn.Linear(embad_size, vocab_size)
    def forward(self, x, mask, bert_output, encoder_mask):
        x = self.embading(x)
        x = x + self.gps[:, :x.size(1), :] 
        for layer in self.layers:
            x = layer(x, mask, bert_output, encoder_mask)
        x = self.boss2(x)
        return x
pad_id = tokenizer.pad_token_id
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
brain = encoder().to(device)
main = Decoder().to(device)
creation=nn.CrossEntropyLoss(ignore_index=pad_id)
edit=torch.optim.Adam(main.parameters(),lr=0.0001)
dataset = TensorDataset(quastions, quastions_mask, answers_current, answers_current_mask, answers_future)
loader = DataLoader(dataset, batch_size=64, shuffle=True)
torch.backends.cuda.matmul.allow_tf32 = True
brain.eval()
all_bert_outputs = []
with torch.no_grad():
    for batch in loader:
        q, qm, ac, acm, af = [b.to(device) for b in batch]
        qm_bool = (qm == 0)
        bert_out = brain(q, qm_bool)
        all_bert_outputs.append(bert_out.cpu())
for epoch in range(5):
    for i, batch in enumerate(loader):
        torch.cuda.empty_cache()
        q, qm, ac, acm, af = [b.to(device) for b in batch]
        acm = (acm == 0)
        bert_out = all_bert_outputs[i].to(device)
        edit.zero_grad()
        output = main(ac, acm, bert_out, (q == 0))
        loss = creation(output.view(-1, output.size(-1)), af.reshape(-1))
        loss.backward()
        edit.step()
    print(f"epoch={epoch} loss={loss.item()}")
    torch.save(main.state_dict(), "cross_att.pth")