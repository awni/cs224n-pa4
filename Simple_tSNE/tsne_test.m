addpath('/Users/awni/Desktop/cs224n/homework/pa4/Simple_tSNE');
addpath('/Users/awni/Desktop/cs224n/homework/pa4/data');

%% Load and process data
wordVecs = load('wordVectors.txt');
words = importdata('vocab.txt', '\n');
train_words = importdata('unique_train_processed', ' ');
wordVecsToMap = [];
wordsMapped = [];

for i=10001:numel(train_words)
   ind = strmatch(train_words(i), words, 'exact');
   if length(ind)==1
     wordVecsToMap = [wordVecsToMap; wordVecs(ind,:)];
     wordsMapped = [wordsMapped;train_words(i)];
   end
end

% add UNK word
wordVecsToMap = [wordVecs(1,:); wordVecsToMap];
wordsMapped = [words(1); wordsMapped];

%% Train



% Set parameters
no_dims = 2;
init_dims = 30;
perplexity = 30;

% Run t_SNE
mappedX = tsne(wordVecs, [], no_dims, init_dims, perplexity);
save 'mapped_L_orig'

%% Plot

load mapped_L_orig;

randperm(numel(train_words));
figure('Color', 'w')        
scatter(mappedX(:,1), mappedX(:,2), 0.001); axis off
text(mappedX(:,1), mappedX(:,2), words(:))

