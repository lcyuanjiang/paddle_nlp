from model.tokenization_bert import BertTokenizer
class Config(object):
    def __init__(self,args):
        self.cuda='True'
        self.model_name='bert'
        self.train_path=args.train_path
        self.dev_path=args.dev_path
        self.test_path=args.test_path
        self.save_path=args.save_path
        self.use_cuda=args.use_cuda
        self.num_class=args.num_class
        self.num_epoch=args.num_epoch
        self.batch_size=args.batch_size
        self.pad_size=args.pad_size
        self.learning_rate=args.lr
        self.bert_pretrain_model=args.pretrain_model_path
        self.tokenizer=BertTokenizer.from_pretrained(self.bert_pretrain_model+
                                      "/bert-base-chinese-vocab.txt")
        self.hidden_size=768
        self.log_path='logs/'+self.model_name
