
import pandas as pd
import numpy as np
import re
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

class CleanData:
    """
    Loads Wikipedia talk page edits data, clean it, tokenize, pad, and returns list of lists of tokenized and padded sequences ready for training
    """
        
    def __init__(self, data_path="train.csv", threshold=0.1, frac=0.7, perc=0.1, seq_length=200, batch_size=50):
        """
        threshold: fraction of how much OOV words appear in the training data
        frac: train-test split fraction
        perc: subsampling percentage for the "neutral" class
        """
        self.threshold  = threshold
        self.frac = frac
        self.perc = perc 
        self.seq_length = seq_length
        self.batch_size = batch_size
        # load data
        self.data = pd.read_csv(data_path) 

        self.comments = self.subsample_train()["comment_text"]
        self.train_comments = None
        self.vocab_unique = None
        self.word2index = None

        # # used for repeated sampling until 41 classes are realized
        # self.recursive_counter = 0
        

    def pre_process_train(self):
        reviews = self.subsample_train()
        comments, labels = reviews["comment_text"], reviews["class"].values
        review_vocab = set()
        # populate review_vocab with all of the words in the given reviews
        #       Remember to split reviews into individual words 
        #       using "split(' ')" instead of "split()".
        for review in comments:
            comment = re.split(r"[' ', \n, $, #, =, \, *, ^, @, ?, !, /]", \
                               review.lower())  # split string by " " and \n
            for word in comment:
                review_vocab.add(word)
        self.vocab_unique = review_vocab
        
        # Convert the vocabulary set to a list so we can access words via indices
        self.review_vocab = list(review_vocab)
                
        # Store the sizes of the review and label vocabularies.
        self.review_vocab_size = len(self.review_vocab)
        
        # Create a dictionary of words in the vocabulary mapped to index positions
        self.word2index = {}
        #  populate self.word2index with indices for all the words in self.review_vocab
        #       like you saw earlier in the notebook
        for i,word in enumerate(self.review_vocab):
            self.word2index[word] = i + 2       # reserve 0 for padding and 1 for OOV words
        
        #  tokenize training reviews
        self.train_comments = self.tokenize(comments)
        # pad training data
        self.train_comments = self.padding(self.train_comments, self.seq_length)
        self.train_labels = labels
    
    def tokenize(self, comments):
        # tokenization - convert words to vector
        training_comments = list()
        for comment in comments:
            choice = np.random.uniform(size=1)
            indices = set()
            _comment = re.split(r"[' ', \n, $, #, =, \, *, ^, @, ?, !, /]", comment.lower())
            for word in _comment:
                if(word in self.word2index.keys()):
                    indices.add(self.word2index[word])
            # randomly introduce out of vocab(OOV) word in a review
            #  with probability = threshold
            if choice < self.threshold:
                oov = np.random.randint(0, high=len(_comment))
                indices = list(indices)
                indices.insert(oov, 1)
            else:
                oov = np.random.randint(0, high=len(_comment))
                indices = list(indices)
                indices.insert(oov, 0)
            training_comments.append(indices)
        return training_comments

    def pre_process_test(self):
        # tokenize test reviews
        _, test_reviews = self.data_split()
        test_reviews, test_labels = test_reviews["comment_text"], test_reviews["class"].values
        test_comments = []
        for review in test_reviews:
            indices = []
            comment = re.split(r"[' ', \n, $, #, =, \, *, ^, @, ?, !, /]", \
                               review.lower())  # split string by " " and \n
            for word in comment:
                if word in self.vocab_unique:
                    indices.append(self.word2index[word])
                else:
                    indices.append(1)            # this is an OOV word
            test_comments.append(indices)
        # pad test data
        test_comments_padded = self.padding(test_comments, self.seq_length)
        # split test data into test and validation sets
        self.test_comments, self.val_comments, self.test_labels, self.val_labels = train_test_split(test_comments_padded, test_labels, test_size=0.33, random_state=9)
        

    def powerset(self):
        """
        Extract unique classes from  the multilabels
        """
        data = self.data
        # assign comments without class a new class 'neutral'
        label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
        data['neutral'] = 1-data[label_cols].max(axis=1)

        #### convert the multilabels to multiclasses #####
        # vectorize the multilabels into a 'combination' column
        data['combination'] = data[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', \
                                    'identity_hate', 'neutral']].agg(tuple, axis=1)
        # Assign a new column 'class' for the new labels by creating unique multi-label classes
        data['class'] = data['combination'].factorize()[0]
        return data

    def data_split(self):
        data = self.powerset()
        # split data into train and test 
        train_data = data.sample(frac=self.frac)
        test_data = data.drop(train_data.index)
        return train_data, test_data
    
    def subsample_train(self):
        """
        Subsampling from the `neutral` class only
        Split the dataframe into two, one containing only the `neutral` class and the other, the rest of the classes
        We subsample the `neutral` class
        Merge the subsample with the dataframe containing the other classes
        """        
        i = 0
        while i < 1000:
           no_of_classes, new_data = self.repeat_sample()
           if no_of_classes == 41:
               return new_data
           i += 1
        print("The fraction for this train-test split is too small to ensure all the 41 unique classes are in the training set")
    
    def repeat_sample(self):
        data, _ = self.data_split()
        neutral_data= data.loc[data['class']==0]
        # subsample the neutral class
        neutral_sub = neutral_data.sample(frac=self.perc)
        # remove neutral class from train data
        train_data_new = data[data['class'] > 0]
        # merge two data frames
        new_data = pd.concat([train_data_new, neutral_sub], axis='rows')
        # ensure number of classes is 41 to avoid missing any class in the training data
        no_of_classes = len(np.unique(new_data['class'].values))
        return no_of_classes, new_data

    def padding(self, reviews_ints, seq_length):
        ''' Return features of review_ints, where each review is padded with 0's or truncated to the input seq_length.
        '''
        
        # getting the correct rows x cols shape
        features = np.zeros((len(reviews_ints), seq_length), dtype=int)

        # for each review, I grab that review and 
        for i, row in enumerate(reviews_ints):
            features[i, -len(row):] = np.array(row)[:seq_length]
        return features
    
    def to_dataloader(self):
        """
        takes in train, test, val data, and their corresponding labels and return pytorch dataloader
        """
        # create Tensor datasets
        train_data = TensorDataset(torch.from_numpy(self.train_comments), torch.from_numpy(self.train_labels))
        valid_data = TensorDataset(torch.from_numpy(self.val_comments), torch.from_numpy(self.val_labels))
        test_data = TensorDataset(torch.from_numpy(self.test_comments), torch.from_numpy(self.test_labels))

        # make sure the SHUFFLE your training data
        train_loader = DataLoader(train_data, shuffle=True, batch_size=self.batch_size, drop_last=True)
        valid_loader = DataLoader(valid_data, shuffle=True, batch_size=self.batch_size, drop_last=True)
        test_loader = DataLoader(test_data, shuffle=True, batch_size=self.batch_size, drop_last=True)
        return train_loader, test_loader, valid_loader
    
if __name__ == '__main__':
    process = CleanData()