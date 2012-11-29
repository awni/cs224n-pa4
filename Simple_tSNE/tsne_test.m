addpath('/Users/awni/Desktop/cs224n/homework/pa4/Simple_tSNE');
addpath('/Users/awni/Desktop/cs224n/homework/pa4/data');
% Load data
wordVecs = load('wordVectors.txt');
% Set parameters
no_dims = 2;
init_dims = 30;
perplexity = 30;

% Run t?SNE
mappedX = tsne(wordVecs, [], no_dims, init_dims, perplexity);

        
scatter(mappedX(1,1), mappedX(1,2), 10); axis off
