addpath('/Users/awni/Desktop/cs224n/homework/pa4/Simple_tSNE');
addpath('/Users/awni/Desktop/cs224n/homework/pa4/data');

%% Load and process data
wordVecs = load('wordVectors.txt');
load indices;
indices = indices(1:10000);
wordVecs = wordVecs(indices,:);
%words = importdata('vocab.txt', '\n');
words = importdata('final_train_words.txt', ' ');



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

toPlot = randperm(numel(words));
toPlot = toPlot(1:1000);
figure('Color', 'w')        
scatter(mappedX(toPlot,1), mappedX(toPlot,2), 0.001); axis off
text(mappedX(toPlot,1), mappedX(toPlot,2), words(toPlot))

%%Preprocessing already done

% wordVecsToMap = [];
% wordsMapped = [];
% 
% for i=1:numel(train_words)
%    ind = strmatch(train_words(i), words, 'exact');
%    if mod(i,1000)==0
%        disp(i)
%    end
%    if length(ind)>0
%      wordsMapped = [wordsMapped;train_words(i),ind];
%      wordVecsToMap = [wordVecsToMap; wordVecs(ind,:)];
%    end
% end
